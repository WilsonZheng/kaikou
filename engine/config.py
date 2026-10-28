"""Engine settings; `config.json` in the support directory overrides the defaults."""

from __future__ import annotations

import json
import os

DEFAULTS = {
    "asr_model": "mlx-community/Qwen3-ASR-1.7B-8bit",
    "llm_model": "mlx-community/Qwen3-8B-4bit",
    "polish": True,
    "hotwords_max": 300,
    "incremental": True,
    "chunked": True,
    "background_beams": 4,
    "segment": {"min_segment_s": 16.0, "max_segment_s": 28.0, "min_pause_s": 0.45},
    "keep_history": True,
    "keep_warm_s": 45,
    "audio_keep_days": 30,        # recorded audio; the text history is kept
    "learned_stale_days": 60,     # one-off corrections that never came back
    "context_terms": True,        # bias each dictation toward names in recent use
    "agent_context": True,        # ...including Claude Code sessions active lately
    "agent_context_minutes": 30,
    "slack_context": True,        # ...and who you talked to on Slack (needs agent-slack)
    "slack_context_days": 2,
    "slack_refresh_hours": 4,     # sooner (30 min) after a dictation into Slack
    "agent_learning": True,       # learn from fixes made before sending to Claude Code
}


def load_config(support: str) -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))
    path = os.path.join(support, "config.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            user = json.load(f)
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg
