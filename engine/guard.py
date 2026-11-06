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

# Who does what: "你觉得我下次应该让 GP 给我做哪些检查" once came out as "...应该让你做哪些
# 检查". An edit deletes words but keeps the speaker's 我/你/他 in order.
_PERSON = re.compile(r"[我你您他她]")
_PERSON_CLASS = {"我": "1", "你": "2", "您": "2", "他": "3", "她": "3"}

MAX_NOVELTY = 0.35   # Typeless p97 0.23; answering/expanding outputs 0.55+
MIN_COVERAGE = 0.70  # Typeless never below 0.80; rewrites 0.44-0.67
MIN_SENTENCE = 0.50  # every long transcript sentence keeps at least half its tokens
MIN_COVERAGE_CORRECTED = 0.55


_CN_NUM = "零〇一二两三四五六七八九十百千万"
_CN_VAL = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_MUL = {"十": 10, "百": 100, "千": 1000, "万": 10000}


def cn_number(s: str) -> str | None:
    """八十五 -> 85, 一千二百 -> 1200, 一二三 -> 123, 一点二 -> 1.2; None if it is not a number."""
    whole, _, frac = s.partition("点")
    if not whole or any(c not in _CN_VAL for c in frac):
        return None
    if all(c in _CN_VAL for c in whole):
        n = "".join(str(_CN_VAL[c]) for c in whole)
    else:
        total = section = digit = 0
        for i, c in enumerate(whole):
            if c in _CN_VAL:
                digit = _CN_VAL[c]
                if i == len(whole) - 1 and i and _CN_MUL.get(whole[i - 1], 0) >= 100:
                    digit *= _CN_MUL[whole[i - 1]] // 10  # 一万二 is 12000, 两百五 is 250
            elif c == "万":
                total, section, digit = (total + section + digit) * 10000, 0, 0
            else:
                section += (digit or 1) * _CN_MUL[c]
                digit = 0
        n = str(total + section + digit)
    return n + ("." + "".join(str(_CN_VAL[c]) for c in frac) if frac else "")


def numbers(text: str) -> set[str]:
    """Every number in the text, written as digits: 1509, 一五零九 and 一千五百零九 all count."""
    found = set(re.findall(r"\d+", text))
    for run in re.findall(f"[{_CN_NUM}]+", text):
        if (n := cn_number(run)) is not None:
            found.add(n)
    return found


def changed_number(raw: str, out: str) -> bool:
    """A number of three or more digits that the transcript never said."""
    said = numbers(raw)
    return any(n not in said for n in re.findall(r"\d{3,}", out))


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


def changed_person(raw: str, out: str) -> bool:
    """The output's pronouns are not the transcript's with some left out (one in 143
    real edits besides the bad one trips this; it then gets the minimal edit)."""
    people = lambda t: [_PERSON_CLASS[c] for c in _PERSON.findall(t.replace("其他", ""))]
    said = iter(people(raw))
    return not all(p in said for p in people(out))


_LIMIT = re.compile(r"只(?!是)|仅")


def dropped_limit(raw: str, out: str) -> bool:
    """A "只" or "仅" went missing ("就只回一跟二吧" edited to "就回一、二吧"), which widens
    what the user asked for. "只是" (= but) may be reworded. Dropping "不" or "没" is left
    alone: in real edits that is almost always a repeat being merged ("不知道，不知道")."""
    return len(_LIMIT.findall(out)) < len(_LIMIT.findall(raw))


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
    if changed_person(raw, out):
        return "changed person"
    if changed_number(raw, out):
        return "changed number"
    if not corrected and dropped_limit(raw, out):
        return "dropped limit"
    return None


def accept_polished(raw: str, out: str, allowed: set[str] = frozenset()) -> bool:
    return check(raw, out, allowed) is None
