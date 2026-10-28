"""Names and terms the user is working with but has not said yet.

Most dictation goes into a terminal (Orca running Claude Code), where the accessibility
API reads nothing, so the window context is empty exactly where it matters. Two local
sources fill the gap:

- the Claude Code sessions written to in the last half hour: their conversation names
  the people, repos and PRs being talked about right now;
- the user's own dictations from the last few days;
- who the user has been talking to on Slack in the last two days (slack.py).

Only vocabulary-like words are taken from them. They bias the recognizer and the
post-editor toward the right spelling and never go into the text by themselves.
"""

from __future__ import annotations

import difflib
import glob
import json
import os
import re
import time
from collections import Counter

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-_./][A-Za-z0-9]+)*")
_NAME = re.compile(r"(?<![A-Za-z0-9_./-])[A-Z][a-z]{2,14}(?![A-Za-z0-9_/-])")
_COMMON: set[str] | None = None
_CALENDAR = {"January", "February", "March", "April", "May", "June", "July", "August", "September",
             "October", "November", "December", "Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"}  # capitalized in the dictionary, so not "common"


def common_words() -> set[str]:
    """Lowercase English words from the system dictionary (empty if it is missing)."""
    global _COMMON
    if _COMMON is None:
        try:
            with open("/usr/share/dict/words", encoding="utf-8", errors="ignore") as f:
                _COMMON = {w.strip() for w in f if w[:1].islower()}
        except OSError:
            _COMMON = set()
    return _COMMON


def is_common(word: str) -> bool:
    """The dictionary has no inflections, so "teams" and "deployed" are checked by stem."""
    w, common = word.lower(), common_words()
    return w in common or any(w.endswith(e) and w[:-len(e)] in common for e in ("s", "es", "ed", "d", "ing"))


def is_name_like(term: str) -> bool:
    """Worth biasing toward: names, products, repos (has a capital, digit or hyphen).
    Lowercase words like "merge" or "comment" are already in the static hotwords, and
    so are short acronyms (PR, API), which the recognizer gets right anyway."""
    return bool(re.search(r"[A-Z0-9-]", term)) and not re.fullmatch(r"[A-Z0-9/]{1,5}", term)


class TermIndex:
    """Finds vocabulary terms (up to three words, case-insensitive) in free text."""

    def __init__(self, terms: list[str]):
        self.by_lower: dict[str, str] = {}
        for t in terms:
            if is_name_like(t):
                self.by_lower.setdefault(t.lower(), t)
        self.max_words = max((len(k.split()) for k in self.by_lower), default=1)

    def find(self, text: str) -> Counter:
        words = _WORD.findall(text)
        low = [w.lower() for w in words]
        hits: Counter = Counter()
        for i in range(len(low)):
            for n in range(min(self.max_words, len(low) - i), 0, -1):
                term = self.by_lower.get(" ".join(low[i:i + n]))
                if term:
                    hits[term] += 1
                    break
        return hits


def new_names(text: str, known: set[str], min_count: int = 3) -> Counter:
    """Capitalized words that recur and are not English words: people and products the
    vocabulary does not have yet ("Alice", "Kaikou")."""
    counts = Counter(m.group() for m in _NAME.finditer(text))
    return Counter({w: c for w, c in counts.items()
                    if c >= min_count and not is_common(w) and w.lower() not in known and w not in _CALENDAR})


# ---------- sources ----------

def _message_text(entry: dict) -> tuple[str, str]:
    """(conversation text, tool output) of one transcript line."""
    if entry.get("type") not in ("user", "assistant"):
        return "", ""
    content = (entry.get("message") or {}).get("content")
    if isinstance(content, str):
        return content, ""
    said, tools = [], []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            said.append(block.get("text", ""))
        elif block.get("type") == "tool_result":
            c = block.get("content")
            if isinstance(c, list):
                c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
            if isinstance(c, str):
                tools.append(c[:4000])
    return "\n".join(said), "\n".join(tools)


def read_tail(path: str, max_bytes: int) -> list[dict]:
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        data = f.read()
    lines = data.split(b"\n")
    if size > max_bytes:
        lines = lines[1:]  # the first line is cut off
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


class ContextSource:
    """Collects the terms for one dictation. File reads are cached by mtime, so a
    dictation in the same session costs a stat() per file."""

    def __init__(self, terms: list[str], history_dir: str = "", agent_globs: list[str] | None = None,
                 agent_minutes: float = 30, history_days: int = 3, max_terms: int = 40,
                 tail_bytes: int = 400_000, slack=None):
        self.index = TermIndex(terms)
        self.known = {t.lower() for t in terms}
        self.history_dir = history_dir
        self.agent_globs = agent_globs if agent_globs is not None else [
            os.path.expanduser("~/.claude*/projects/*/*.jsonl")]
        self.agent_minutes = agent_minutes
        self.history_days = history_days
        self.max_terms = max_terms
        self.tail_bytes = tail_bytes
        self._agent_cache: dict[str, tuple[float, Counter]] = {}
        self._history_cache: dict[str, tuple[float, Counter]] = {}
        self._listed: tuple[float, list[str]] = (0.0, [])
        self.slack = slack
        self._slack_cache: tuple[int, Counter] = (0, Counter())

    @staticmethod
    def _cached(cache: dict, path: str, compute) -> Counter:
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return Counter()
        hit = cache.get(path)
        if hit and hit[0] == mtime:
            return hit[1]
        try:
            value = compute(path)
        except (OSError, ValueError):
            value = Counter()
        cache[path] = (mtime, value)
        return value

    def _agent_file(self, path: str) -> Counter:
        said, tools = [], []
        for entry in read_tail(path, self.tail_bytes):
            s, t = _message_text(entry)
            said.append(s)
            tools.append(t)
        said_text = "\n".join(said)
        hits = self.index.find(said_text) + self.index.find("\n".join(tools))
        return hits + new_names(said_text, self.known)

    def _candidates(self, now: float) -> list[str]:
        """Transcripts touched in the last day. Listing thousands of old sessions takes
        a few hundred ms, so the list is refreshed every few minutes; a session started
        since then is still found through its project directory's mtime."""
        listed_at, paths = self._listed
        if now - listed_at > 300:
            paths = []
            for pattern in self.agent_globs:
                for p in glob.glob(pattern):
                    try:
                        if now - os.path.getmtime(p) <= 86400:
                            paths.append(p)
                    except OSError:
                        continue
            self._listed = (now, paths)
            return paths
        fresh = []
        for pattern in self.agent_globs:
            for d in glob.glob(os.path.dirname(pattern)):
                try:
                    if os.path.getmtime(d) >= listed_at:
                        fresh += [p for p in glob.glob(os.path.join(d, os.path.basename(pattern))) if p not in paths]
                except OSError:
                    continue
        return paths + fresh

    def agent_terms(self, now: float | None = None) -> Counter:
        now = time.time() if now is None else now
        files = []
        for p in self._candidates(now):
            try:
                m = os.path.getmtime(p)
            except OSError:
                continue
            if now - m <= self.agent_minutes * 60:
                files.append((m, p))
        recent = [p for _, p in sorted(files, reverse=True)[:3]]  # parallel sessions: the latest few
        for p in [p for p in self._agent_cache if p not in recent]:
            del self._agent_cache[p]
        total: Counter = Counter()
        for p in recent:
            total += self._cached(self._agent_cache, p, self._agent_file)
        return total

    def _history_file(self, path: str) -> Counter:
        texts = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    texts.append(json.loads(line).get("text") or "")
                except ValueError:
                    continue
        return self.index.find("\n".join(texts))

    def history_terms(self, today: float | None = None) -> Counter:
        if not self.history_dir:
            return Counter()
        today = time.time() if today is None else today
        total: Counter = Counter()
        for d in range(self.history_days):
            day = time.strftime("%Y-%m-%d", time.localtime(today - d * 86400))
            total += self._cached(self._history_cache, os.path.join(self.history_dir, f"{day}.jsonl"),
                                  self._history_file)
        return total

    def slack_terms(self) -> Counter:
        """Vocabulary and names from recent Slack messages, plus the first and last
        names of the people in them ("Hoffman", "Buechi")."""
        if self.slack is None:
            return Counter()
        version, text, people = self.slack.snapshot()
        if version != self._slack_cache[0]:
            hits = self.index.find(text) + new_names(text, self.known)
            for full, n in people.items():
                for part in full.split():
                    if re.fullmatch(r"[A-Za-z][A-Za-z'-]{2,}", part):
                        hits[part] += n
            self._slack_cache = (version, hits)
        return self._slack_cache[1]

    def gather(self, window_context: str = "", now: float | None = None) -> list[str]:
        """Most relevant first: what is on screen, then the live agent sessions and
        Slack, then recent dictations."""
        score: Counter = Counter()
        for term, n in self.index.find(window_context).items():
            score[term] += 100 + n
        for term, n in new_names(window_context, self.known, min_count=1).items():
            score[term] += 50 + n
        for term, n in self.agent_terms(now).items():
            score[term] += 10 + min(n, 30)
        for term, n in self.slack_terms().items():
            score[term] += 8 + min(n, 20)
        for term, n in self.history_terms(now).items():
            score[term] += 5 + min(n, 10)
        return [t for t, _ in score.most_common(self.max_terms)]


# ---------- sound-alike hints ----------

_SOUND = [(r"ph", "f"), (r"ck|q|c", "k"), (r"x", "ks"), (r"[zs]+", "s"), (r"[fvbp]+", "p"),
          (r"[dt]+", "t"), (r"w", "u")]


def sound_key(word: str) -> str:
    """Rough consonant skeleton: labials, dentals and velars each fold together, the
    way a Chinese-accented "Hoffman" comes back as "Hopon" or "Hoppen"."""
    w = word.lower()
    for pat, rep in _SOUND:
        w = re.sub(pat, rep, w)
    head, rest = w[:1], re.sub(r"[aeiouy]+", "", w[1:])
    return re.sub(r"(.)\1+", r"\1", head + rest)


def sound_alike(transcript: str, candidates: list[str], allowed: set[str], limit: int = 3) -> list[tuple[str, str]]:
    """Latin words in the transcript that are not English and not in the vocabulary,
    paired with the candidate name they most resemble in spelling and sound."""
    names = [c for c in dict.fromkeys(candidates) if " " not in c and len(c) >= 3 and is_name_like(c)]
    out: list[tuple[str, str]] = []
    for word in dict.fromkeys(re.findall(r"(?<![A-Za-z0-9_./-])[A-Za-z]{3,}(?![A-Za-z0-9_/-])", transcript)):
        low = word.lower()
        if low in allowed or is_common(low):
            continue
        best, best_score = None, 0.0
        for name in names:
            if name.lower() == low:
                best = None
                break
            if name[0].lower() != low[0] and sound_key(name)[:1] != sound_key(word)[:1]:
                continue
            score = max(difflib.SequenceMatcher(None, low, name.lower()).ratio(),
                        difflib.SequenceMatcher(None, sound_key(word), sound_key(name)).ratio() - 0.05)
            if score > best_score:
                best, best_score = name, score
        if best and best_score >= 0.7:
            out.append((word, best))
            if len(out) >= limit:
                break
    return out
