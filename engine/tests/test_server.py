"""End-to-end protocol test with a fake pipeline (no models)."""
import asyncio
import base64
import json
import os
import sys
import tempfile
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server  # noqa: E402


class FakePipeline:
    def __init__(self, cfg, lexicon, status=lambda s: None, history_dir=""):
        self.recognized = []

    def session_terms(self, window_context):
        return ["Hoffman"] if "Hoffman" in window_context else []

    def warmup(self):
        pass

    def recognize_segment(self, session, audio):
        self.recognized.append(len(audio))
        return f"seg{len(self.recognized)}", 0.01

    def advance_chunks(self, session, should_stop=None):
        session.edited.append("chunk")

    def lexicon_changed(self, path):
        return False

    def touch(self):
        pass

    def finalize(self, session):
        from pipeline import session_terms
        parts = [f.result()[0] for _, _, f in session.segments]
        return {"text": " ".join(parts + [f"tail({session.n})"]), "raw": ",".join(session.edited),
                "timings": {"terms": session_terms(session)}}


class DeadlockTest(unittest.TestCase):
    def test_background_edit_never_waits_for_queued_segments(self):
        """Regression: advance_chunks ran on the GPU thread and blocked on a segment
        future queued behind it, freezing the engine for good."""
        import concurrent.futures as cf
        import types
        from pipeline import Pipeline
        p = Pipeline.__new__(Pipeline)
        pending = cf.Future()  # never completes, like a job queued behind us
        s = types.SimpleNamespace(segments=[(0, 1, pending)], chunked=0, pending="", edited=[], continues=[],
                                  app_name="", chunk_timings={})
        done = []
        t = __import__("threading").Thread(target=lambda: (p.advance_chunks(s), done.append(True)))
        t.start(); t.join(2)
        self.assertTrue(done, "advance_chunks blocked on an unfinished segment")
        self.assertEqual(s.chunked, 0)


class ServerProtocolTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.sock = os.path.join(self.tmp, "e.sock")
        server.Pipeline = FakePipeline
        server._limit_memory = lambda: None

    async def _client(self, srv):
        for _ in range(100):
            if os.path.exists(self.sock):
                break
            await asyncio.sleep(0.02)
        reader, writer = await asyncio.open_unix_connection(self.sock)
        return reader, writer

    async def _read(self, reader, kind, timeout=5):
        end = time.time() + timeout
        while time.time() < end:
            msg = json.loads(await asyncio.wait_for(reader.readline(), timeout))
            if msg["type"] == kind:
                return msg
        raise TimeoutError(kind)

    def send(self, writer, msg):
        writer.write((json.dumps(msg) + "\n").encode())

    def test_round_trip_with_segments_and_cancel(self):
        async def run():
            srv = server.Server(self.sock, self.tmp)
            task = asyncio.ensure_future(srv.run())
            reader, writer = await self._client(srv)
            ready = await self._read(reader, "status")
            while not ready["ready"]:
                ready = await self._read(reader, "status")

            # 18 s of speech, a pause, 3 s more: one background segment + tail.
            t = np.arange(18 * 16000) / 16000
            speech = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
            audio = np.concatenate([speech, np.zeros(16000, np.float32), speech[: 3 * 16000]])
            self.send(writer, {"type": "begin", "id": "a", "app": "x", "app_name": "X", "context": "@Hoffman"})
            for i in range(0, len(audio), 1600):
                pcm = (audio[i:i + 1600] * 32767).astype("<i2").tobytes()
                self.send(writer, {"type": "audio", "id": "a", "pcm16": base64.b64encode(pcm).decode()})
            self.send(writer, {"type": "end", "id": "a"})
            res = await self._read(reader, "result")
            self.assertEqual(res["id"], "a")
            self.assertTrue(res["text"].startswith("seg1 tail("), res["text"])
            self.assertIn("after_release", res["timings"])
            self.assertEqual(res["raw"], "chunk", "background edit ran")
            self.assertEqual(res["timings"]["terms"], ["Hoffman"], "context terms reach the pipeline")

            # Cancelled sessions produce nothing.
            self.send(writer, {"type": "begin", "id": "b"})
            self.send(writer, {"type": "cancel", "id": "b"})
            self.send(writer, {"type": "end", "id": "b"})
            self.send(writer, {"type": "hello"})
            msg = await self._read(reader, "status")
            self.assertTrue(msg["ready"])

            hist = os.path.join(self.tmp, "history")
            self.assertTrue(any(f.endswith(".jsonl") for f in os.listdir(hist)))
            writer.close()
            task.cancel()

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
