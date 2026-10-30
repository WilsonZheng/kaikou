"""Faithfulness checks for the post-edit.

The post-editor must edit the transcript, never answer it, expand it, rewrite it into
a spec, or drop parts of it. Each check below was calibrated on the user's own
recordings with Typeless's edits as the "good edit" reference.
"""

from __future__ import annotations

import re

_TOK = re.compile(r"[㐀-鿿]|[a-z0-9]+")
_CN_DIGIT = {"零": "0", "〇": "0", "幺": "1", "一": "1", "二": "2", "两": "2", "三": "3", "四": "4",
             "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}
_CN_RUN = re.compile("[零〇幺一二两三四五六七八九]{2,}")
_SENT = re.compile(r"[。！？!?\n]+|(?<=[a-z])\.\s")

# Spoken filler and connective characters an edit is expected to remove.
_DROPPABLE = set("嗯呃啊哦呀吧呢嘛就然后反正那个这是的了") | {"um", "uh", "er", "like", "so", "yeah", "okay", "ok"}
# Words a faithful edit has no reason to introduce. Typeless itself occasionally adds
# one weak marker ("请", "注意："), so only a strong one or two weak ones reject.
_STRONG_MARKERS = ("您", "考虑到", "**", "Note:", "以下是", "好的，")
_WEAK_MARKERS = ("请", "是否", "建议", "注意：")

# The speaker took something back ("周二，不对，周三", "Brooke 是男的，不是女的"): a good
# edit then drops the retracted part, so coverage may fall further.
CORRECTION = re.compile(r"不对|说错|我是说|我的意思是|重新说|重说|更正|纠正|搞错|弄错|记错|口误|"
                        r"不是[^，,。？?]{1,12}[，,]\s*(?:而)?是|是[^，,。？?]{1,12}[，,]\s*不是|"
                        r"\bactually\b|\bI mean\b|scratch that", re.I)

MAX_NOVELTY = 0.35   # Typeless p97 0.23; answering/expanding outputs 0.55+
MIN_COVERAGE = 0.70  # Typeless never below 0.80; rewrites 0.44-0.67
MIN_SENTENCE = 0.50  # every long transcript sentence keeps at least half its tokens
MIN_COVERAGE_CORRECTED = 0.55


def normalize_numbers(text: str) -> str:
    """一五零九 -> 1509, so spoken digits and written digits compare equal."""
    return _CN_RUN.sub(lambda m: "".join(_CN_DIGIT[c] for c in m.group()), text)


def tokens(text: str) -> list[str]:
    return _TOK.findall(normalize_numbers(text).lower())


def novelty(raw: str, out: str, allowed: set[str] = frozenset()) -> float:
    """Share of output tokens that never appear in the transcript or vocabulary."""
    seen = set(tokens(raw)) | allowed
    toks = [t for t in tokens(out) if not t.isdigit()]
    if not toks:
        return 0.0
    return sum(t not in seen for t in toks) / len(toks)


def coverage(raw: str, out: str) -> float:
    """Share of the transcript's content tokens that survive in the output."""
    kept = set(tokens(out))
    content = [t for t in tokens(raw) if t not in _DROPPABLE]
    if not content:
        return 1.0
    return sum(t in kept for t in content) / len(content)


def dropped_sentence(raw: str, out: str) -> bool:
    kept = set(tokens(out))
    for sentence in _SENT.split(raw):
        toks = [t for t in tokens(sentence) if t not in _DROPPABLE]
        if len(toks) >= 6 and sum(t in kept for t in toks) / len(toks) < MIN_SENTENCE:
            return True
    return False


def added_marker(raw: str, out: str) -> bool:
    new = lambda ms: [m for m in ms if m in out and m not in raw]
    return bool(new(_STRONG_MARKERS)) or len(new(_WEAK_MARKERS)) >= 2


def check(raw: str, out: str, allowed: set[str] = frozenset()) -> str | None:
    """Returns the reason the edit is rejected, or None if it is acceptable."""
    if not out.strip():
        return "empty"
    r, o = len(raw.strip()), len(out.strip())
    if r >= 40 and o < 0.3 * r:
        return "too short"
    if o > 2.0 * r + 40:
        return "too long"
    n_out = len(tokens(out))
    if n_out < 8:
        seen = set(tokens(raw)) | allowed
        if sum(t not in seen and not t.isdigit() for t in tokens(out)) > 1:
            return "short answer"
    elif novelty(raw, out, allowed) > MAX_NOVELTY:
        return "novel content"
    corrected = bool(CORRECTION.search(raw))
    if len(tokens(raw)) >= 20 and coverage(raw, out) < (MIN_COVERAGE_CORRECTED if corrected else MIN_COVERAGE):
        return "low coverage"
    if not corrected and dropped_sentence(raw, out):
        return "dropped sentence"
    if added_marker(raw, out):
        return "added marker"
    return None


def accept_polished(raw: str, out: str, allowed: set[str] = frozenset()) -> bool:
    return check(raw, out, allowed) is None
