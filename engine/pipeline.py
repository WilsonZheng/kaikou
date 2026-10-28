"""Recognize -> fix vocabulary -> post-edit -> fix vocabulary."""

from __future__ import annotations

import difflib
import logging
import os
import re
import time

import mlx.core as mx
import numpy as np

from asr import Recognizer
from context import ContextSource, is_name_like, sound_alike
from guard import check, novelty, tokens
from lexicon import Lexicon, lexicon_mtime
from polish import Polisher
from prompts import build_system_prompt, user_message
from segmenter import has_speech, join_texts, session_language

log = logging.getLogger("kaikou")

_CJK = re.compile(r"[㐀-鿿]")
_DOTTED = re.compile(r"\b((?:[A-Z]\.){2,})")
_SPACED = re.compile(r"(?<![A-Za-z])([A-Z](?: [A-Z]){1,5})(?![A-Za-z])")


_FILLER = re.compile(r"[呃嗯额]+[，,、]?\s*|啊{2,}[，,、]?\s*")
_FILLER_PHRASE = re.compile(r"(?:呃|嗯)?那啥[，,、]\s*")
# Stutters on single function characters ("又又", "我我"); real reduplication
# (看看, 试试, 刚刚) uses other characters and is left alone.
_STUTTER = re.compile(r"([我你他她它这那的了去给就是在把有要又都也还再会能])\1+")
# Repeated words: "整个整个", "什么什么", "thread thread", "appointment appointment".
# "一个一个" (one by one) is kept.
_REPEAT_CJK = re.compile(r"((?!一个)[\u3400-\u9fff]{2,4})(?:[，,]?\s*\1)+")
_REPEAT_LATIN = re.compile(r"\b([A-Za-z][A-Za-z0-9_-]{1,})(?:\s+\1\b)+", re.I)
_KEEP_DOUBLE = {"看看", "试试", "想想", "谢谢", "刚刚", "常常", "慢慢", "天天", "好好", "人人", "多多", "渐渐", "偷偷", "悄悄"}
_SENTENCE_END = re.compile(r"[。！？!?]|\.(?=\s)")
_CLAUSE_END = re.compile(r"[，,；;]")
_LIST_ITEM = re.compile(r"^(\d+)\.\s")
# Context terms per dictation: more makes the recognizer likelier to read them back on
# silence. The post-editor never sees the list itself (shown one, it started adding
# names nobody said); it only gets hints for words that sound like one of them.
ASR_TERMS = 25


def split_complete(text: str, clause_after: int = 80) -> tuple[str, str, bool]:
    """Splits off the last sentence, which may continue in the next segment (cuts fall on
    pauses, and the recognizer closes every segment with a period even mid-sentence).

    Long run-on speech ("…，然后…，然后…") may have no sentence end at all; past
    `clause_after` characters it is split at the last comma instead, and the third value
    says the next chunk continues the same sentence."""
    body = text.rstrip()
    if body and body[-1] in "。！？!?.":
        body = body[:-1]
    ends = [m.end() for m in _SENTENCE_END.finditer(body)]
    if ends:
        return text[: ends[-1]].strip(), text[ends[-1]:].strip(), False
    clauses = [m.end() for m in _CLAUSE_END.finditer(body)]
    if len(body) >= clause_after and clauses and clauses[-1] >= 15:
        return text[: clauses[-1]].strip(), text[clauses[-1]:].strip(), True
    return "", text.strip(), False


def join_chunks(chunks: list[str], continues: list[bool] | None = None) -> str:
    """Joins separately edited chunks as paragraphs; a numbered list that a chunk boundary
    split in two keeps counting (1. 2. | 1. -> 1. 2. 3.). A chunk cut mid-sentence
    (continues[i]) is joined to the next one with a comma instead of a paragraph break."""
    continues = continues or [False] * len(chunks)
    merged: list[str] = []
    glue = False
    for chunk, cont in zip(chunks, continues + [False] * (len(chunks) - len(continues))):
        chunk = chunk.strip()
        if not chunk:
            continue
        if glue and merged:
            merged[-1] = merged[-1].rstrip("。.") + ("，" if not merged[-1].endswith(("，", ",")) else "") + chunk
        else:
            merged.append(chunk)
        glue = cont
    out: list[str] = []
    last_num = 0
    for chunk in merged:
        lines = chunk.split("\n")
        first = _LIST_ITEM.match(lines[0])
        continuing = last_num > 0 and first is not None and first.group(1) == "1"
        if out:
            out.append("" if not continuing else None)
        for line in lines:
            m = _LIST_ITEM.match(line)
            if m and continuing:
                last_num += 1
                line = f"{last_num}. " + line[m.end():]
            elif m:
                last_num = int(m.group(1))
            elif line.strip():
                last_num = 0
                continuing = False
            out.append(line)
        # keep counting only if the chunk ends on a list item
        if not (lines and _LIST_ITEM.match(lines[-1])):
            last_num = 0
    return "\n".join(l for l in out if l is not None)


_TAG = re.compile(r"</?\s*(context|transcript|alt)\s*>", re.I)


def sanitize_context(context: str) -> str:
    """Context comes from other apps (possibly someone else's message): strip anything
    that could fake the prompt's own tags."""
    return _TAG.sub(" ", context or "")


def context_terms(context: str, transcript: str, limit: int = 20) -> set[str]:
    """Context words the guard may accept as spelling fixes: only Latin tokens that look
    like a misheard version of something actually said (similar spelling), so text near
    the cursor can correct a name but never inject content."""
    said = {t for t in tokens(transcript) if t.isascii() and not t.isdigit()}
    out: set[str] = set()
    for t in {t for t in tokens(context) if t.isascii() and len(t) >= 3 and not t.isdigit()}:
        if t in said or any(difflib.SequenceMatcher(None, t, s).ratio() >= 0.6 for s in said if len(s) >= 2):
            out.add(t)
            if len(out) >= limit:
                break
    return out


def clean_fillers(text: str) -> str:
    """Deterministic last pass on every result: fillers and stutters the post-editor
    missed. Conservative on purpose; anything that can be meaningful is kept."""
    if len(text.strip()) <= 2:
        return text  # a lone "嗯" can be a real answer
    text = _FILLER.sub("", text)
    text = _FILLER_PHRASE.sub("", text)
    text = _REPEAT_CJK.sub(lambda m: m.group(1) if m.group(1) not in _KEEP_DOUBLE else m.group(0), text)
    text = _REPEAT_LATIN.sub(r"\1", text)
    text = _STUTTER.sub(r"\1", text)
    return re.sub(r"^[，,、\s]+", "", text)


_CJK_THEN_LATIN = re.compile(r"([㐀-鿿])([A-Za-z0-9])")
_LATIN_THEN_CJK = re.compile(r"([A-Za-z0-9])([㐀-鿿])")


def space_mixed(text: str) -> str:
    """One space between Chinese and English words or digits ("Ben现在" -> "Ben 现在"),
    the house style the post-editor usually applies but sometimes misses."""
    return _LATIN_THEN_CJK.sub(r"\1 \2", _CJK_THEN_LATIN.sub(r"\1 \2", text))


def collapse_acronyms(text: str) -> str:
    """Chinese mode spells acronyms out ("P.R.", "C I C D"); join them back."""
    text = _DOTTED.sub(lambda m: m.group(1).replace(".", ""), text)
    text = _SPACED.sub(lambda m: m.group(1).replace(" ", ""), text)
    return re.sub(r"\bCICD\b", "CI/CD", text)


def session_terms(session) -> list[str]:
    """Names likely in this dictation, gathered when the recording began (context.py).
    Never holds up recognition: a slow or failed lookup just means no extra terms."""
    fut = getattr(session, "terms_future", None)
    if fut is None:
        return []
    try:
        return fut.result(timeout=1.0)
    except Exception:
        return []


class Pipeline:
    def __init__(self, cfg: dict, lexicon: Lexicon, status=lambda s: None, history_dir: str = ""):
        self.cfg = cfg
        self.lexicon = lexicon
        self.history_dir = history_dir
        self.context = self._context_source(lexicon)
        status("加载识别模型…")
        self.asr = Recognizer(cfg["asr_model"])
        self.asr.set_hotwords(lexicon.hotwords)
        self.allowed = {t for term in lexicon.terms for t in tokens(term)}
        self.polisher = None
        if cfg["polish"]:
            status("加载整理模型…")
            self.polisher = Polisher(cfg["llm_model"], self._system_prompt(),
                                     extra_prompts={"minimal": self._system_prompt(minimal=True)})

    def _context_source(self, lexicon: Lexicon) -> ContextSource | None:
        if not self.cfg.get("context_terms", True):
            return None
        return ContextSource(lexicon.terms, self.history_dir,
                             agent_globs=None if self.cfg.get("agent_context", True) else [],
                             agent_minutes=self.cfg.get("agent_context_minutes", 30))

    def session_terms(self, window_context: str) -> list[str]:
        """Runs off the GPU thread when a recording begins."""
        t = time.perf_counter()
        source = self.context
        terms = source.gather(window_context) if source else []
        log.info("%d context term(s) in %.2fs", len(terms), time.perf_counter() - t)
        return terms

    def _system_prompt(self, minimal: bool = False) -> str:
        return build_system_prompt(self.lexicon.terms, self.lexicon.examples, self.lexicon.misheard, minimal=minimal)

    def reload_lexicon(self, lexicon: Lexicon, should_stop=None) -> None:
        """Only a change to lexicon.json rebuilds the post-editor's cached prompts (tens of
        seconds); learned corrections just refresh rules, hotwords and per-request hints."""
        full = lexicon.base_mtime != self.lexicon.base_mtime
        old_base = self.lexicon.base_mtime
        self.lexicon = lexicon
        self.context = self._context_source(lexicon)
        self.allowed = {t for term in lexicon.terms for t in tokens(term)}
        self.allowed |= {t for _, dst in lexicon.learned_hints for t in tokens(dst)}
        self.asr.set_hotwords(lexicon.hotwords)
        if self.polisher and full:
            done = (self.polisher.set_system_prompt(self._system_prompt(minimal=True), "minimal", should_stop)
                    and self.polisher.set_system_prompt(self._system_prompt(), "default", should_stop))
            if not done:
                # Someone started talking: keep the old prompts and retry when idle.
                self.lexicon.base_mtime = old_base
                self.lexicon.mtime = -1
                log.info("lexicon rebuild paused for a dictation; will retry")
                return
        log.info("lexicon reloaded (%s): %d terms, %d learned", "full" if full else "learned only",
                 len(lexicon.terms), len(lexicon.learned_hints))

    def lexicon_changed(self, path: str) -> bool:
        return lexicon_mtime(path) != self.lexicon.mtime

    def warmup(self) -> None:
        self.asr.warmup()
        if self.polisher:
            self.polisher.polish(user_message("测试"), source="测试")

    def touch(self) -> None:
        """Pages both models back in after idle time. Under memory pressure macOS evicts
        them, and the first dictation after an hour paid ~3 s for it; doing this when the
        key goes down hides that behind the user's speech."""
        t = time.perf_counter()
        self.asr.transcribe(np.zeros(8_000, dtype=np.float32))
        if self.polisher:
            probe = mx.array(self.polisher.tok.encode("你好", add_special_tokens=False))[None]
            mx.eval(self.polisher.model(probe, cache=self.polisher._fresh_cache()))
        took = time.perf_counter() - t
        if took > 1:
            log.info("models touched in %.2fs (they had been paged out)", took)

    # ---------- while recording ----------

    def recognize_segment(self, session, audio: np.ndarray) -> tuple[str, float]:
        """Recognizes one finished segment. Auto language first; if the recording is
        Chinese but this segment came back with no Chinese at all, it was probably
        translated (segments that start with an English word do that), so redo it in
        Chinese mode, which keeps English words in English."""
        # Beam search here costs nothing the user waits for: segments are recognized while
        # they are still talking. The tail after release stays greedy.
        beams = self.cfg.get("background_beams", 1)
        released = lambda: getattr(session, "ended", False)
        terms = session_terms(session)[:ASR_TERMS]
        text, secs = self.asr.transcribe(audio, beams=beams, should_stop=released, terms=terms)
        if session.lang == "Chinese" and text and not _CJK.search(text):
            text, more = self.asr.transcribe(audio, language="Chinese", beams=beams, should_stop=released,
                                             terms=terms)
            secs += more
        if session.lang is None and text.strip():
            session.lang = session_language(text)
        return text, secs

    def edit(self, raw: str, app_name: str, timings: dict | None = None, should_stop=None,
             context: str = "", terms: list[str] | None = None) -> str | None:
        """Post-edits one piece of transcript, with the guard and the minimal-edit retry.
        `terms` are names likely in this dictation. Returns None if `should_stop`
        interrupted it."""
        fixed = collapse_acronyms(self.lexicon.apply(raw))
        if not self.polisher or not fixed.strip():
            return fixed
        t = timings if timings is not None else {}
        # Words around the cursor may name the people and terms being talked about;
        # they are allowed in the output without counting as invented content.
        context = sanitize_context(context)
        terms = terms or []
        allowed = self.allowed | context_terms(context + "\n" + ", ".join(terms), fixed)
        notes = self._notes(fixed, terms)
        for line in notes:  # hinted spellings are fixes, not invented content
            allowed |= {t for t in tokens(line.split("：", 1)[-1])}
        msg = user_message(fixed, app_name, context, notes)
        r = self.polisher.polish(msg, source=fixed, should_stop=should_stop)
        if r.stopped:
            return None
        t["polish_prefill"] = round(t.get("polish_prefill", 0) + r.prefill_s, 3)
        t["polish_gen"] = round(t.get("polish_gen", 0) + r.gen_s, 3)
        t["polish_tokens"] = t.get("polish_tokens", 0) + r.tokens
        why = check(fixed, r.text, allowed) if r.complete else "incomplete"
        if why:
            # Never log dictated text; history (if enabled) has it.
            log.warning("post-edit rejected (%s): %d -> %d chars, novelty %.2f; retrying minimal edit",
                        why, len(fixed), len(r.text), novelty(fixed, r.text, allowed))
            t["polish_rejected"] = t.get("polish_rejected", 0) + 1
            r = self.polisher.polish(msg, source=fixed, prompt="minimal")
            why = check(fixed, r.text, allowed) if r.complete else "incomplete"
            if why:
                t["minimal_rejected"] = t.get("minimal_rejected", 0) + 1
                return space_mixed(clean_fillers(fixed))
            return space_mixed(clean_fillers(self.lexicon.apply(r.text)))
        return space_mixed(clean_fillers(self.lexicon.apply(r.text)))

    def _notes(self, text: str, terms: list[str]) -> list[str]:
        """Per-request <context> line: the likely mishearings that actually occur in this
        transcript (known ones, learned ones, and words that sound like one of the names
        in play or in the vocabulary). Empty when there are none."""
        notes = []
        hints = self.lexicon.hints_for(text)
        names = terms + [t for t in self.lexicon.terms if is_name_like(t)]
        for pair in sound_alike(text, names, self.allowed):
            if pair[0] not in {a for a, _ in hints}:
                hints.append(pair)
        if hints:
            notes.append("转写里可能听错的词（上下文说得通时才改）：" + "；".join(f"{a}→{b}" for a, b in hints[:8]))
        return notes

    def advance_chunks(self, session, should_stop=None) -> None:
        """Edits the finished part of the recording while the user keeps talking: every
        recognized segment except its last (possibly unfinished) sentence. After release
        only the rest is left to edit."""
        while session.chunked < len(session.segments):
            fut = session.segments[session.chunked][2]
            # Runs on the single GPU thread: a segment queued behind this job is not
            # done yet, and waiting for it here would deadlock the engine.
            if not fut.done():
                return
            try:
                text = fut.result()[0]
            except Exception:
                return  # finalize re-recognizes failed segments
            session.chunked += 1
            pending_before = session.pending
            pending = join_texts([session.pending, text]) if session.pending else text
            complete, session.pending, cont = split_complete(pending)
            if len(complete) >= 15:
                edited = self.edit(complete, session.app_name, session.chunk_timings, should_stop,
                                   getattr(session, "context", ""), session_terms(session))
                if edited is None:  # the key was released: finalize takes over from here
                    session.chunked -= 1
                    session.pending = pending_before
                    return
                session.edited.append(edited)
                session.continues.append(cont)
            else:
                session.pending = pending

    # ---------- after release ----------

    def finalize(self, session) -> dict:
        t0 = time.perf_counter()
        audio = session.audio()
        seg_time = 0.0
        start = 0
        for k, (s_start, s_end, fut) in enumerate(session.segments):
            try:
                seg_time += fut.result()[1]
            except Exception:
                log.exception("segment failed; recognizing it again")
                done = _Done(self.recognize_segment(session, audio[s_start:s_end]))
                session.segments[k] = (s_start, s_end, done)
            start = s_end
        tail_audio = audio[start:]
        tail, tail_time = ("", 0.0)
        terms = session_terms(session)
        if not session.segments or has_speech(tail_audio, session.segmenter.levels()):
            tail, tail_time = self.asr.transcribe(tail_audio, language=session.lang, terms=terms[:ASR_TERMS])
        raw = join_texts([f.result()[0] for _, _, f in session.segments] + [tail])
        timings = {"segments": len(session.segments), "asr_bg": round(seg_time, 3), "asr_tail": round(tail_time, 3)}

        if self.cfg.get("chunked", True) and session.segments:
            self.advance_chunks(session)  # anything the background did not get to
            rest = join_texts([session.pending, tail]) if session.pending else tail
            last = (self.edit(rest, session.app_name, timings, context=getattr(session, "context", ""), terms=terms)
                    if rest.strip() else "")
            text = join_chunks(session.edited + [last], session.continues + [False])
            timings["chunks"] = len(session.edited) + 1
            for k, v in session.chunk_timings.items():
                timings["bg_" + k] = v
        else:
            text = self.edit(raw, session.app_name, timings, context=getattr(session, "context", ""), terms=terms)
        timings["total_engine"] = round(time.perf_counter() - t0, 3)
        return {"text": text, "raw": raw, "timings": timings}


class _Done:
    """A finished future holding a value (for segments recognized again in finalize)."""

    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value

    def done(self):
        return True

    def exception(self):
        return None
