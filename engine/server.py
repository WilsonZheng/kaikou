"""Kaikou engine: local speech-to-text + post-editing over a Unix socket.

Protocol (newline-delimited JSON):
  client -> {"type":"hello"}
  client -> {"type":"begin","id":..,"app":bundle_id,"app_name":..}
  client -> {"type":"audio","id":..,"pcm16":base64 int16 LE mono 16 kHz}
  client -> {"type":"end","id":..} | {"type":"cancel","id":..}
  engine -> {"type":"status","ready":bool,"text":..}
  engine -> {"type":"result","id":..,"text":..,"raw":..,"timings":{..}}
  engine -> {"type":"error","id":..,"text":..}
Results and errors are broadcast to every client; the app filters by id, so a
reconnect in the middle of processing does not lose the text.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures as cf
import datetime as dt
import json
import logging
import os
import sys
import threading
import time
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config  # noqa: E402
from lexicon import Lexicon, learned_path, record_learned  # noqa: E402
import maintenance  # noqa: E402
from pipeline import Pipeline  # noqa: E402
from segmenter import Segmenter  # noqa: E402

log = logging.getLogger("kaikou")
LOG_DIR = os.path.expanduser("~/Library/Logs/Kaikou")


class Session:
    def __init__(self, sid: str, app: str, app_name: str, seg_cfg: dict, writer):
        self.id = sid
        self.app = app
        self.app_name = app_name
        self.writer = writer
        self.chunks: list[np.ndarray] = []
        self.n = 0
        self.segmenter = Segmenter(**seg_cfg)
        self.segments: list[tuple[int, int, cf.Future]] = []
        self.lang: str | None = None
        # Chunked editing while recording: edited text so far, the unfinished last
        # sentence carried over, and how many segments have been consumed.
        self.edited: list[str] = []
        self.continues: list[bool] = []         # edited[i] was cut mid-sentence
        self.pending = ""
        self.chunked = 0
        self.chunk_timings: dict = {}
        self.context = ""                       # text near the cursor, for terms only
        self.terms_future: cf.Future | None = None  # names likely in this dictation
        self.ended = False
        self.cancelled = False

    def audio(self) -> np.ndarray:
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0, dtype=np.float32)


class Server:
    def __init__(self, socket_path: str, support: str):
        self.socket_path = socket_path
        self.support = support
        self.lexicon_path = os.path.join(support, "lexicon.json")
        self.cfg = load_config(support)
        self.gpu = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
        # Reading recent agent sessions for names is file work: keep it off the GPU queue.
        self.ctx = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="context")
        self.pending = 0                       # GPU jobs queued or running
        self.pending_lock = threading.Lock()
        self.pipeline: Pipeline | None = None
        self.ready = False
        self.status_text = "加载模型中…"
        self.writers: set[asyncio.StreamWriter] = set()
        self.sessions: dict[str, Session] = {}
        self.loop: asyncio.AbstractEventLoop | None = None
        self.last_gpu = time.time()
        self._finishing: set[asyncio.Task] = set()
        self._maintained: dt.date | None = None

    # ---------- GPU queue ----------

    def submit(self, fn, *args) -> cf.Future:
        with self.pending_lock:
            self.pending += 1

        def run():
            try:
                return fn(*args)
            finally:
                self.last_gpu = time.time()
                with self.pending_lock:
                    self.pending -= 1
        fut = self.gpu.submit(run)
        # A job cancelled before it starts never runs its finally.
        fut.add_done_callback(lambda f: f.cancelled() and self._dec())
        return fut

    def _dec(self) -> None:
        with self.pending_lock:
            self.pending -= 1

    def gpu_busy_elsewhere(self) -> bool:
        """True when another job is waiting behind the current one."""
        with self.pending_lock:
            return self.pending > 1

    # ---------- lifecycle ----------

    async def run(self) -> None:
        self.loop = asyncio.get_running_loop()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        server = await asyncio.start_unix_server(self.handle_client, path=self.socket_path, limit=16 * 1024 * 1024)
        os.chmod(self.socket_path, 0o600)
        log.info("listening on %s", self.socket_path)
        self.submit(self._load)
        # Keep a reference: asyncio holds tasks weakly, and an unreferenced one can be
        # garbage-collected mid-flight (the watcher silently never ran).
        self._watcher = asyncio.ensure_future(self._watch_lexicon())
        async with server:
            await server.serve_forever()

    def _load(self) -> None:
        t = time.time()
        try:
            _limit_memory()
            lex = Lexicon.load(self.lexicon_path, self.cfg["hotwords_max"])
            self.pipeline = Pipeline(self.cfg, lex, status=lambda s: self._set_status(False, s),
                                     history_dir=os.path.join(self.support, "history"))
            self.pipeline.warmup()
            self._set_status(True, "就绪")
            log.info("models ready in %.1fs (asr=%s llm=%s terms=%d)", time.time() - t,
                     self.cfg["asr_model"], self.cfg["llm_model"], len(lex.terms))
        except Exception as e:  # keep the socket up so the app can show the error
            log.exception("model load failed")
            self._set_status(False, f"模型加载失败：{e}")

    async def _watch_lexicon(self) -> None:
        """Reloads lexicon.json in the background when it changes, while nobody is talking."""
        while True:
            await asyncio.sleep(3)
            p = self.pipeline
            if not p or not self.ready or self.sessions or self.pending:
                continue
            today = dt.date.today()
            if self._maintained != today and maintenance.due(self.support, today):
                self._maintained = today             # once a day, even if it fails
                try:
                    r = maintenance.run_daily(self.support, self.cfg, LOG_DIR)
                    log.info("daily maintenance: %d audio file(s) removed (%.1f MB), %d learned fix(es) dropped",
                             r["audio_removed"], r["freed_mb"], r["learned_removed"])
                except Exception:
                    log.exception("daily maintenance failed")
                continue
            # Keep the models resident: on this machine macOS pages them out after a few
            # idle minutes, and the next dictation then waited ~3 s for them.
            if time.time() - self.last_gpu > self.cfg.get("keep_warm_s", 45):
                self.submit(self._touch)
                continue
            try:
                if p.lexicon_changed(self.lexicon_path):
                    log.info("lexicon changed; reloading")
                    def reload():
                        p.reload_lexicon(Lexicon.load(self.lexicon_path, self.cfg["hotwords_max"]),
                                         should_stop=lambda: bool(self.sessions))
                    await asyncio.wrap_future(self.submit(reload))
            except Exception:
                log.exception("lexicon reload failed; keeping the previous one")
                p.lexicon.mtime = os.path.getmtime(self.lexicon_path)  # don't retry every 3 s

    def _set_status(self, ready: bool, text: str) -> None:
        self.ready, self.status_text = ready, text
        if self.loop:
            self.loop.call_soon_threadsafe(self._broadcast, {"type": "status", "ready": ready, "text": text})

    def _broadcast(self, msg: dict) -> None:
        for w in list(self.writers):
            self._send(w, msg)

    @staticmethod
    def _send(w: asyncio.StreamWriter, msg: dict) -> None:
        try:
            w.write((json.dumps(msg, ensure_ascii=False) + "\n").encode())
        except Exception:
            pass

    # ---------- client handling ----------

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.writers.add(writer)
        self._send(writer, {"type": "status", "ready": self.ready, "text": self.status_text})
        try:
            while line := await reader.readline():
                try:
                    self.dispatch(json.loads(line), writer)
                except Exception:
                    log.exception("bad message ignored")
        except (ConnectionResetError, asyncio.IncompleteReadError):
            pass
        finally:
            self.writers.discard(writer)
            # Recordings this client was still streaming can never finish.
            for sid in [sid for sid, s in self.sessions.items() if s.writer is writer]:
                self._cancel(self.sessions.pop(sid))
            writer.close()

    def dispatch(self, msg: dict, writer: asyncio.StreamWriter) -> None:
        kind, sid = msg.get("type"), msg.get("id")
        if kind == "hello":
            self._send(writer, {"type": "status", "ready": self.ready, "text": self.status_text})
        elif kind == "begin" and sid:
            self.sessions[sid] = Session(sid, msg.get("app", ""), msg.get("app_name", ""), self.cfg["segment"], writer)
            self.sessions[sid].context = str(msg.get("context", ""))[:1000]
            if self.pipeline:
                self.sessions[sid].terms_future = self.ctx.submit(self.pipeline.session_terms,
                                                                  self.sessions[sid].context)
            if self.pipeline and self.ready and time.time() - self.last_gpu > 30:
                self.submit(self._touch)
        elif kind == "audio" and sid in self.sessions:
            s = self.sessions[sid]
            pcm = np.frombuffer(base64.b64decode(msg["pcm16"]), dtype="<i2").astype(np.float32) / 32768.0
            s.chunks.append(pcm)
            s.n += len(pcm)
            cut = s.segmenter.feed(pcm) if self.cfg["incremental"] else None
            if cut is not None and self.pipeline:
                start = s.segments[-1][1] if s.segments else 0
                fut = self.submit(self.pipeline.recognize_segment, s, s.audio()[start:cut])
                s.segments.append((start, cut, fut))
                if self.cfg["chunked"]:
                    self.submit(self._advance_chunks, s)
        elif kind == "learn":
            pairs = [(str(p.get("from", "")), str(p.get("to", ""))) for p in msg.get("pairs", []) if isinstance(p, dict)]
            items = record_learned(learned_path(self.lexicon_path), pairs)
            log.info("learned %d correction(s); %d known", len(pairs), len(items))  # never the words themselves
        elif kind == "cancel" and sid in self.sessions:
            self._cancel(self.sessions.pop(sid))
        elif kind == "end" and sid in self.sessions:
            s = self.sessions.pop(sid)
            s.ended = True
            task = asyncio.ensure_future(self.finish(s))
            self._finishing.add(task)
            task.add_done_callback(self._finishing.discard)

    def _touch(self) -> None:
        try:
            self.pipeline.touch()
        except Exception:
            log.exception("warm-up failed (ignored)")

    def _cancel(self, s: Session) -> None:
        s.cancelled = True
        for _, _, fut in s.segments:
            fut.cancel()

    def _advance_chunks(self, s: Session) -> None:
        if s.cancelled:
            return
        try:
            # Stop as soon as the key is released; finalize edits whatever is left.
            self.pipeline.advance_chunks(s, should_stop=lambda: s.ended or s.cancelled)
        except Exception:
            log.exception("background edit failed; finalize will redo it")

    async def finish(self, s: Session) -> None:
        t_end = time.time()
        result = None
        try:
            if not self.pipeline or not self.ready:
                self._broadcast({"type": "error", "id": s.id, "text": self.status_text})
                return
            try:
                result = await asyncio.wrap_future(self.submit(self._finalize, s))
            except Exception as e:
                log.exception("finalize failed")
                self._broadcast({"type": "error", "id": s.id, "text": f"出错了：{e}"})
                return
            result["timings"]["after_release"] = round(time.time() - t_end, 3)
            self._broadcast({"type": "result", "id": s.id, **result})
            log.info("result %.1fs audio, %.2fs after release, %s", s.n / 16000,
                     result["timings"]["after_release"], json.dumps(result["timings"]))
        finally:
            try:
                self.save_history(s, result)  # the audio is kept even when processing failed
            except Exception:
                log.exception("history save failed")

    def _finalize(self, s: Session) -> dict:
        try:
            return self.pipeline.finalize(s)
        finally:
            _release_buffers()

    def save_history(self, s: Session, result: dict | None) -> None:
        if not self.cfg["keep_history"] or s.n == 0:
            return
        day = dt.datetime.now().strftime("%Y-%m-%d")
        hdir = os.path.join(self.support, "history")
        adir = os.path.join(hdir, "audio")
        os.makedirs(adir, exist_ok=True)
        wav_path = os.path.join(adir, f"{s.id}.wav")
        pcm = (np.clip(s.audio(), -1, 1) * 32767).astype("<i2")
        with wave.open(wav_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm.tobytes())
        rec = {"id": s.id, "ts": dt.datetime.now().isoformat(timespec="seconds"), "app": s.app,
               "app_name": s.app_name, "duration": round(s.n / 16000, 2), "audio": wav_path}
        if result:
            rec.update(raw=result["raw"], text=result["text"], timings=result["timings"])
        else:
            rec["failed"] = True
        with open(os.path.join(hdir, f"{day}.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _limit_memory() -> None:
    """Keep model weights wired (this machine swaps heavily) and cap MLX's buffer cache."""
    try:
        import mlx.core as mx
        mx.set_wired_limit(int(9.5 * 1024 ** 3))
        mx.set_cache_limit(1024 ** 3)
    except Exception:
        log.exception("could not set MLX memory limits")


def _release_buffers() -> None:
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", required=True)
    ap.add_argument("--support", required=True)
    ap.add_argument("--exit-on-stdin-eof", action="store_true",
                    help="exit when the parent app goes away (it holds our stdin open)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    os.makedirs(a.support, exist_ok=True)
    if a.exit_on_stdin_eof:
        def watch_parent() -> None:
            sys.stdin.buffer.read()
            log.info("app went away; exiting")
            os._exit(0)
        threading.Thread(target=watch_parent, daemon=True).start()
    asyncio.run(Server(a.socket, a.support).run())


if __name__ == "__main__":
    main()
