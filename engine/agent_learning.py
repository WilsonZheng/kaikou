"""Learns from fixes made in a terminal, where the app cannot watch the text box.

A dictation into Orca lands in a Claude Code prompt, and what the user finally sent is
in that session's transcript. When the dictation shows up there with a few words
changed, those changes are the same kind of correction the app learns in other apps
(Learning.swift), and they go to the same place (lexicon.record_learned): fixed once, a
hint for the post-editor; twice, an automatic replacement.
"""

from __future__ import annotations

import datetime as dt
import difflib
import glob
import json
import os
import re
import unicodedata

# Apps whose text the accessibility API cannot read; elsewhere the app already learns.
TERMINALS = ("orca", "terminal", "iterm", "ghostty", "warp", "alacritty", "kitty", "wezterm")
STATE = "agent_learning.json"
SETTLE = dt.timedelta(minutes=20)  # time to fix and send a dictation
WINDOW = (dt.timedelta(minutes=-2), dt.timedelta(minutes=20))


def tokens(s: str) -> list[str]:
    """Latin words (with . _ - inside), single CJK characters, or single other characters."""
    out, word = [], ""
    for ch in s:
        if ch.isascii() and (ch.isalnum() or (ch in "._-" and word)):
            word += ch
        else:
            if word:
                out.append(word)
                word = ""
            out.append(ch)
    if word:
        out.append(word)
    return out


def _content(t: str) -> bool:
    return any(c.isalnum() for c in t)


def _cjk(t: str) -> bool:
    return any(0x3400 <= ord(c) <= 0x9FFF for c in t)


def _trim(s: str) -> str:
    return "".join(s).strip().strip("".join(c for c in s if unicodedata.category(c).startswith("P"))).strip()


def corrections(inserted: str, edited: str, max_tokens: int = 6) -> list[tuple[str, str]]:
    """(inserted, corrected) pairs for short replacements; additions, deletions and
    punctuation-only changes are ignored, and so is a rewrite. Same rules as
    TokenDiff.corrections in the app."""
    a, b = tokens(inserted), tokens(edited)
    if not a or not b or len(a) > 600 or len(b) > 800:
        return []
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    if not any(map(_content, a)) or sum(m.size for m in sm.get_matching_blocks()) < 0.6 * len(a):
        return []
    pairs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace":
            continue
        d, n = [t for t in a[i1:i2] if _content(t)], [t for t in b[j1:j2] if _content(t)]
        if not d or not n or len(d) > max_tokens or len(n) > max_tokens:
            continue
        left = right = ""
        # A one-character Chinese fix would rewrite that character everywhere; learn it
        # with its neighbours instead ("自身修" -> "自审修").
        if len(d) == 1 and _cjk(d[0]):
            left = a[i1 - 1] if i1 > 0 and _cjk(a[i1 - 1]) else ""
            right = a[i2] if i2 < len(a) and _cjk(a[i2]) else ""
            if not left and not right:
                continue
        src = left + _trim("".join(a[i1:i2])) + right
        dst = left + _trim("".join(b[j1:j2])) + right
        # Chinese heard as other Chinese keeps its syllable count ("从事" for "重置");
        # a change of length there is rewording ("直接搞" to "就快做了"), not a fix.
        if re.fullmatch(r"[\u3400-\u9fff]+", src) and re.fullmatch(r"[\u3400-\u9fff]+", dst) \
                and len(src) != len(dst):
            continue
        if src and dst and src != dst and "\n" not in dst:
            pairs.append((src, dst))
    return pairs


def sent_span(text: str, message: str) -> str | None:
    """The part of a sent message that the dictation became, if it is there at all
    (the user often types more around it)."""
    sm = difflib.SequenceMatcher(None, text, message, autojunk=False)
    blocks = [m for m in sm.get_matching_blocks() if m.size >= 2]
    if sum(m.size for m in blocks) < 0.6 * len(text):
        return None
    start = blocks[0].b - blocks[0].a          # extend to cover unmatched edges of the dictation
    end = blocks[-1].b + blocks[-1].size + (len(text) - blocks[-1].a - blocks[-1].size)
    start, end = max(0, start), min(len(message), end)
    word = lambda c: c.isascii() and c.isalnum()
    while start > 0 and word(message[start - 1]) and word(message[start]):
        start -= 1                             # never cut a word in half ("NEO", not "EO")
    while end < len(message) and word(message[end - 1]) and word(message[end]):
        end += 1
    return message[start:end]


def _message_text(entry: dict) -> str:
    content = (entry.get("message") or {}).get("content")
    if isinstance(content, list):
        content = "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    if not isinstance(content, str) or content.lstrip().startswith("<"):
        return ""  # slash commands, hook output and other machine-written turns
    return content


def sent_messages(globs: list[str], since: dt.datetime) -> list[tuple[dt.datetime, str]]:
    out = []
    for pattern in globs:
        for p in glob.glob(pattern):
            try:
                if os.path.getmtime(p) < since.timestamp():
                    continue
                with open(p, encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if '"user"' not in line:  # cheap prefilter before parsing
                            continue
                        try:
                            e = json.loads(line)
                        except ValueError:
                            continue
                        if e.get("type") != "user":
                            continue
                        text, ts = _message_text(e), e.get("timestamp")
                        if text and ts:
                            t = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
                            if t >= since:
                                out.append((t, text))
            except OSError:
                continue
    return out


def dictations(history_dir: str, since: dt.datetime, until: dt.datetime) -> list[tuple[dt.datetime, str]]:
    out = []
    day = since.astimezone().date()
    while day <= until.astimezone().date():
        p = os.path.join(history_dir, f"{day.isoformat()}.jsonl")
        day += dt.timedelta(days=1)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                    t = dt.datetime.fromisoformat(e["ts"]).astimezone()
                except (ValueError, KeyError):
                    continue
                app = (e.get("app", "") + " " + e.get("app_name", "")).lower()
                if since < t <= until and e.get("text") and any(k in app for k in TERMINALS):
                    out.append((t, e["text"]))
    return out


def learn(support: str, agent_globs: list[str] | None = None, now: dt.datetime | None = None,
          first_days: int = 7, spelling: dict[str, str] | None = None) -> list[tuple[str, str]]:
    """Corrections from terminal dictations since the last run. Each dictation is looked
    at once, after it has had time to be sent. `spelling` maps lowercase vocabulary to
    its spelling: a prompt typed as "hoffman" still teaches "Hoffman"."""
    spelling = spelling or {}
    agent_globs = agent_globs if agent_globs is not None else [os.path.expanduser("~/.claude*/projects/*/*.jsonl")]
    now = (now or dt.datetime.now().astimezone())
    state_path = os.path.join(support, STATE)
    try:
        with open(state_path, encoding="utf-8") as f:
            since = dt.datetime.fromisoformat(json.load(f)["until"])
    except (OSError, ValueError, KeyError):
        since = now - dt.timedelta(days=first_days)
    until = now - SETTLE
    pairs: list[tuple[str, str]] = []
    if until > since:
        todo = dictations(os.path.join(support, "history"), since, until)
        messages = sent_messages(agent_globs, since + WINDOW[0]) if todo else []
        for t, text in todo:
            near = [m for mt, m in messages if WINDOW[0] <= mt - t <= WINDOW[1]]
            for m in near:
                span = sent_span(text, m)
                if span is not None:
                    pairs += [(a, spelling.get(b.lower(), b)) for a, b in corrections(text, span)]
                    break
        tmp = state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"until": until.isoformat(timespec="seconds")}, f)
        os.replace(tmp, state_path)
    return pairs


def main() -> None:
    """Runs as its own process at background priority, so reading transcripts never
    holds the engine's interpreter while someone is dictating. Prints the pairs."""
    import argparse
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from lexicon import Lexicon
    ap = argparse.ArgumentParser()
    ap.add_argument("--support", required=True)
    a = ap.parse_args()
    lex = Lexicon.load(os.path.join(a.support, "lexicon.json"))
    pairs = learn(a.support, spelling={t.lower(): t for t in lex.terms})
    print(json.dumps(pairs, ensure_ascii=False))


if __name__ == "__main__":
    main()
