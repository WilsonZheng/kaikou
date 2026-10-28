"""Speech recognizer: Qwen3-ASR (mlx-audio weights) with a cached context prefix.

The vocabulary goes into the system prompt, which comes before the audio, so its KV
cache is computed once and reused; each request only prefills its own audio tokens.
Terms for this one dictation (names from the current conversation) follow the cached
part, so they cost a few dozen tokens per segment and never a rebuild.
The audio embeddings are spliced in with one concatenate instead of mlx-audio's
per-position Python loop.
"""

from __future__ import annotations

import re
import time

import mlx.core as mx
import numpy as np
from mlx_audio.stt import load

_LANG_TAG = re.compile(r"^\s*language\s+[\w-]+\s*<asr_text>", re.I)
_SUFFIX = "<|audio_end|><|im_end|>\n<|im_start|>assistant\n"


class Recognizer:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = load(model_id)
        self.tok = self.model._tokenizer
        self.eos = self.model._eos_token_ids()
        self.audio_token = self.model.config.audio_token_id
        self._head_ids = self.tok.encode("<|im_end|>\n<|im_start|>user\n<|audio_start|>", add_special_tokens=False)
        self._suffix_ids = self.tok.encode(_SUFFIX, add_special_tokens=False)
        self.context = ""
        self._leak = None
        self._orders: list[dict[str, int]] = []  # each list the model was given, in order
        self._hotwords: list[str] = []
        self._leak_extra: tuple = ((), None, [])
        self._prefix_cache = None
        self.set_hotwords([])

    def set_hotwords(self, hotwords: list[str]) -> None:
        self.context = ", ".join(hotwords)
        self._hotwords = list(hotwords)
        # On (near) silence the model sometimes reads its context back: "GitHub, PIMS, Vello, …".
        self._leak = _leak_pattern(hotwords)
        self._orders = [{h: i for i, h in enumerate(hotwords)}]
        self._leak_extra = ((), self._leak, self._orders)
        # Cached part ends right after the hotwords so per-request context can follow.
        prefix = f"<|im_start|>system\n{self.context}\n" if self.context else "<|im_start|>system\n"
        ids = mx.array(self.tok.encode(prefix, add_special_tokens=False))[None]
        cache = self.model.make_cache()
        self.model.model(inputs_embeds=self.model.model.embed_tokens(ids), cache=cache)
        mx.eval([c.state for c in cache])
        self._prefix_cache = cache
        self.prefix_tokens = ids.shape[1]

    def _fresh_cache(self):
        fresh = self.model.make_cache()
        for new, old in zip(fresh, self._prefix_cache):
            new.state = old.state  # trimmed to offset, so the first update copies
        return fresh

    def warmup(self) -> None:
        self.transcribe(np.zeros(16_000, dtype=np.float32))

    beams = 1  # >1 switches decoding to beam search

    def _use_terms(self, terms: list[str]) -> None:
        """Leak detection covers this dictation's terms too; each list keeps its own
        order, since a read-back repeats one list in sequence."""
        key = tuple(terms)
        if key != self._leak_extra[0]:
            orders = [{h: i for i, h in enumerate(self._hotwords)}]
            if terms:
                orders.append({t: i for i, t in enumerate(terms)})
            self._leak_extra = (key, _leak_pattern(list(dict.fromkeys(self._hotwords + list(terms)))), orders)
        self._leak, self._orders = self._leak_extra[1], self._leak_extra[2]

    def transcribe(self, audio: np.ndarray, language: str | None = None, beams: int | None = None,
                   should_stop=None, terms: list[str] | None = None) -> tuple[str, float]:
        """`language` (e.g. "Chinese") skips auto-detection. Chinese mode keeps English
        words in English, so it is the right lock for mixed speech: without it, a
        segment that starts with an English word can come back translated to English.
        `terms` are names and words likely in this dictation (see context.py)."""
        if len(audio) < 1_600:  # under 0.1 s
            return "", 0.0
        t = time.perf_counter()
        feats, mask, n_audio = self.model._preprocess_audio(audio.astype(np.float32))
        audio_emb = self.model.get_audio_features(feats, mask)
        self._use_terms(terms or [])
        head_ids = self._head_ids
        if terms:
            head_ids = self.tok.encode(", ".join(terms) + "\n", add_special_tokens=False) + head_ids
        head_emb = self.model.model.embed_tokens(mx.array(head_ids))
        suffix_ids = self._suffix_ids + (self.tok.encode(f"language {language}<asr_text>", add_special_tokens=False)
                                         if language else [])
        text_emb = self.model.model.embed_tokens(mx.array(suffix_ids))
        embeds = mx.concatenate([head_emb, audio_emb.astype(text_emb.dtype), text_emb], axis=0)
        beams = self.beams if beams is None else beams
        out = self._beam_search(embeds, n_audio, beams, should_stop) if beams > 1 else None
        if out is None:  # greedy, or the beam search gave way to a key release
            out = self._greedy(embeds, n_audio)
        text = _LANG_TAG.sub("", self.tok.decode(out))
        if self._leak:
            text = self._leak.sub(lambda m: "" if self._is_context_run(m.group()) else m.group(), text)
        return text.strip(), time.perf_counter() - t

    def _is_context_run(self, run: str) -> bool:
        """A leaked context run repeats one of the lists it was given (the hotwords or
        this dictation's terms) in that list's order; a real spoken list ("GitHub、Vello
        和 Orca") does not."""
        items = [x.strip() for x in re.split(r"[,，、]", run) if x.strip()]
        for order in self._orders:
            idx = [order.get(x) for x in items]
            if None not in idx and sum(1 for a, b in zip(idx, idx[1:]) if b == a + 1) >= 3:
                return True
        return False

    def _leaked(self, text: str) -> bool:
        return bool(self._leak) and any(self._is_context_run(m.group()) for m in self._leak.finditer(text))

    def _greedy(self, embeds, n_audio: int) -> list[int]:
        m = self.model
        cache = self._fresh_cache()
        hidden = m.model(inputs_embeds=embeds[None], cache=cache)[:, -1:, :]
        logits = m.lm_head(hidden) if m.lm_head is not None else m.model.embed_tokens.as_linear(hidden)
        y = mx.argmax(logits[:, -1, :], axis=-1)
        mx.async_eval(y)
        out: list[int] = []
        for _ in range(64 + n_audio * 2):
            # Queue the next step before reading this token (GPU/Python overlap).
            nxt = mx.argmax(m._forward_with_embeds(m.model.embed_tokens(y[:, None]), cache)[:, -1, :], axis=-1)
            mx.async_eval(nxt)
            tid = int(y.item())
            if tid in self.eos:
                break
            out.append(tid)
            if len(out) % 8 == 0:
                if _looping(out):
                    out = _trim_loop(out)
                    break
                if self._leaked(self.tok.decode(out[-48:])):
                    break
            y = nxt
        return out

    def _beam_search(self, embeds, n_audio: int, beams: int, should_stop=None) -> list[int] | None:
        """Keeps `beams` hypotheses; picks the best by log-probability with a mild
        length normalisation. The prompt is prefilled once, then the cache is repeated
        across beams and reordered every step."""
        m = self.model
        cache = self._fresh_cache()
        hidden = m.model(inputs_embeds=embeds[None], cache=cache)[:, -1:, :]
        logits = (m.lm_head(hidden) if m.lm_head is not None else m.model.embed_tokens.as_linear(hidden))[:, -1, :]
        logp = logits.astype(mx.float32) - mx.logsumexp(logits.astype(mx.float32), axis=-1, keepdims=True)
        first = mx.argsort(-logp[0])[:beams].tolist()
        seqs = [[t] for t in first]
        scores = [float(logp[0, t].item()) for t in first]
        for c in cache:
            c.keys = mx.repeat(c.keys[..., : c.offset, :], len(seqs), axis=0)
            c.values = mx.repeat(c.values[..., : c.offset, :], len(seqs), axis=0)
        finished: list[tuple[float, list[int]]] = []
        norm = lambda sc, n: sc / (((5 + n) / 6) ** 0.6)
        for _ in range(64 + n_audio * 2):
            if should_stop and should_stop():
                return None  # the user is waiting: finish this segment greedily instead
            alive = [i for i, s in enumerate(seqs) if s[-1] not in self.eos]
            for i in range(len(seqs)):
                if seqs[i][-1] in self.eos:
                    finished.append((norm(scores[i], len(seqs[i]) - 1), seqs[i][:-1]))
            if not alive:
                break
            if finished and len(finished) >= beams:
                best_alive = max(norm(scores[i], len(seqs[i])) for i in alive)
                if max(f[0] for f in finished) >= best_alive:
                    break
            seqs = [seqs[i] for i in alive]
            scores = [scores[i] for i in alive]
            idx = mx.array(alive)
            for c in cache:
                c.keys, c.values = c.keys[idx], c.values[idx]
            last = mx.array([[s[-1]] for s in seqs])
            logits = m._forward_with_embeds(m.model.embed_tokens(last), cache)[:, -1, :].astype(mx.float32)
            logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            k = beams * 2
            top = mx.argpartition(-logp, kth=k - 1, axis=-1)[:, :k]
            vals = mx.take_along_axis(logp, top, axis=-1)
            mx.eval(top, vals)
            top_l, vals_l = top.tolist(), vals.tolist()
            cand = sorted(((scores[b] + vals_l[b][j], b, top_l[b][j]) for b in range(len(seqs)) for j in range(k)),
                          reverse=True)[:beams]
            parents = [b for _, b, _ in cand]
            seqs = [seqs[b] + [t] for _, b, t in cand]
            scores = [sc for sc, _, _ in cand]
            pidx = mx.array(parents)
            for c in cache:
                c.keys, c.values = c.keys[pidx], c.values[pidx]
            if len(seqs[0]) % 8 == 0 and _looping(seqs[0]):
                break
        for i, s in enumerate(seqs):
            if s and s[-1] not in self.eos:
                finished.append((norm(scores[i], len(s)), s))
        return max(finished, key=lambda f: f[0])[1] if finished else []


def _leak_pattern(words: list[str]) -> re.Pattern | None:
    if not words:
        return None
    alts = "|".join(re.escape(h) for h in sorted(words, key=len, reverse=True))
    return re.compile(rf"(?:{alts})(?:\s*[,，、]\s*(?:{alts})){{3,}}[,，、]?")


def _looping(out: list[int], min_k: int = 4, max_k: int = 40) -> bool:
    """True when the output ends with the same k tokens three times in a row, the
    classic decoder repetition loop (real speech almost never does that)."""
    for k in range(min_k, min(max_k, len(out) // 3) + 1):
        if out[-k:] == out[-2 * k:-k] == out[-3 * k:-2 * k]:
            return True
    return False


def _trim_loop(out: list[int], min_k: int = 4, max_k: int = 40) -> list[int]:
    for k in range(min_k, min(max_k, len(out) // 3) + 1):
        if out[-k:] == out[-2 * k:-k] == out[-3 * k:-2 * k]:
            end = len(out) - 2 * k
            while end - k >= 0 and out[end - k:end] == out[end:end + k]:
                end -= k
            return out[:end + k]
    return out
