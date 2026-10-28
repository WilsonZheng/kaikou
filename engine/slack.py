"""Who the user has been talking to on Slack lately, and about what.

Read through the agent-slack CLI, which signs in with the Slack desktop app's session.
One search takes 10-25 s, so a dictation never waits for it: it only reads the last
result. Refreshes are rare and happen only while nobody is talking: once at start, then
every few hours, or after half an hour if a dictation went into Slack since (new
conversations are likely then). The child runs at background priority. Message text is
kept in memory only; the terms drawn from it go the same way as the other context
(context.py).
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from collections import Counter

log = logging.getLogger("kaikou")

# The app starts the engine with a minimal PATH, so look where npm-installed CLIs live.
_CLI_GLOBS = ["~/.asdf/installs/nodejs/*/bin/agent-slack", "/opt/homebrew/bin/agent-slack",
              "/usr/local/bin/agent-slack", "~/.local/bin/agent-slack"]
# Review requests and CI notices would fill the result page and crowd out people.
_QUERIES = ["from:me", "to:me -from:@github"]
_URL = re.compile(r"https?://\S+")
_MENTION = re.compile(r"<?@([UW][A-Z0-9]{6,})>?")


def find_cli() -> str | None:
    for pattern in _CLI_GLOBS:
        hits = sorted(glob.glob(os.path.expanduser(pattern)), key=os.path.getmtime, reverse=True)
        if hits:
            return hits[0]
    return shutil.which("agent-slack")


def digest(results: list[dict], me: str = "") -> tuple[str, Counter]:
    """(message text, people) from search results. A person counts once per message
    they wrote or were mentioned in; bots and the user are left out."""
    texts: list[str] = []
    people: Counter = Counter()
    names = {uid: u["real_name"] for r in results for uid, u in (r.get("referenced_users") or {}).items()
             if u.get("real_name") and not u.get("is_bot") and not u.get("deleted") and uid != me}
    for r in results:
        for m in r.get("messages") or []:
            content = m.get("content") or ""
            author = m.get("author") or {}
            involved = set(_MENTION.findall(content))
            if not author.get("bot_id"):
                involved.add(author.get("user_id"))
            for uid in involved & names.keys():
                people[names[uid]] += 1
            content = _MENTION.sub(lambda x: " " + names.get(x.group(1), "") + " ", content)
            texts.append(_URL.sub(" ", content))
    for name in names.values():
        people[name] = max(people[name], 1)  # resolved, so somewhere in these messages
    return "\n".join(texts), people


class SlackSource:
    def __init__(self, days: int = 2, refresh_s: float = 4 * 3600, cli: str | None = None,
                 busy=lambda: False, soon_s: float = 1800):
        self.days = days
        self.refresh_s = refresh_s
        self.soon_s = soon_s
        self.cli = cli or find_cli()
        self.busy = busy
        self.me = ""
        self.version = 0
        self.text = ""
        self.people: Counter = Counter()
        self.fetched = 0.0
        self.poked = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def poke(self) -> None:
        """The user just dictated into Slack: refresh sooner than usual."""
        self.poked = True

    def due(self, now: float) -> bool:
        age = now - self.fetched
        return age >= self.refresh_s or (self.poked and age >= self.soon_s)

    def snapshot(self) -> tuple[int, str, Counter]:
        with self._lock:
            return self.version, self.text, self.people

    def start(self, delay_s: float = 30) -> None:
        if not self.cli or self._thread:
            return
        self._thread = threading.Thread(target=self._loop, args=(delay_s,), name="slack", daemon=True)
        self._thread.start()

    def _loop(self, delay_s: float) -> None:
        time.sleep(delay_s)  # let the models load first
        while True:
            if self.due(time.time()) and not self.busy():
                self.fetched, self.poked = time.time(), False  # a failure also waits a full interval
                try:
                    self.refresh()
                except Exception as e:  # offline, signed out, CLI changed: keep the last result
                    log.info("slack context skipped: %s", e)
            time.sleep(60)

    def _run(self, *args: str) -> dict:
        env = dict(os.environ)
        env["PATH"] = os.path.dirname(self.cli) + os.pathsep + env.get("PATH", "/usr/bin:/bin")
        # A company TLS proxy's root is in the macOS keychain; a shell would point
        # NODE_EXTRA_CA_CERTS at it, but the app does not start from a shell.
        env.setdefault("NODE_USE_SYSTEM_CA", "1")
        # taskpolicy -b: background QoS, so it never competes with a dictation for CPU.
        out = subprocess.run(["/usr/sbin/taskpolicy", "-b", self.cli, *args], capture_output=True,
                             text=True, timeout=120, env=env)
        if out.returncode != 0:
            raise RuntimeError((out.stderr or out.stdout).strip()[:200])
        return json.loads(out.stdout)

    def refresh(self) -> None:
        t = time.perf_counter()
        if not self.me:
            self.me = self._run("auth", "test").get("user_id", "")
        after = (dt.date.today() - dt.timedelta(days=self.days)).isoformat()
        results = [self._run("search", "messages", q, "--after", after, "--limit", "50",
                             "--max-content-chars", "600", "--resolve-users") for q in _QUERIES]
        text, people = digest(results, self.me)
        with self._lock:
            self.text, self.people, self.version = text, people, self.version + 1
        log.info("slack context: %d message(s), %d people in %.1fs",
                 sum(len(r.get("messages") or []) for r in results), len(people), time.perf_counter() - t)
