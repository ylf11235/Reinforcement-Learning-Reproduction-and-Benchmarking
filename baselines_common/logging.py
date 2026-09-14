"""Small JSONL logger shared by the baseline runners."""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any


def _json_safe(value: Any):
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


class JsonlLogger:
    """Write one JSON object per line and keep non-finite values explicit nulls."""

    def __init__(self, out_dir: str, config: dict):
        self.out_dir = os.path.abspath(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "checkpoints"), exist_ok=True)
        self.train_fp = open(os.path.join(self.out_dir, "train_log.jsonl"), "a", encoding="utf-8")
        self.eval_fp = open(os.path.join(self.out_dir, "eval_log.jsonl"), "a", encoding="utf-8")
        with open(os.path.join(self.out_dir, "config.json"), "w", encoding="utf-8") as fp:
            json.dump(_json_safe(config), fp, indent=2, ensure_ascii=False)
        self.started = time.time()

    def _write(self, fp, record: dict) -> None:
        payload = dict(record)
        payload["wall_time_s"] = round(time.time() - self.started, 2)
        fp.write(json.dumps(_json_safe(payload), ensure_ascii=False, allow_nan=False) + "\n")
        fp.flush()

    def train(self, record: dict) -> None:
        self._write(self.train_fp, record)

    def evaluate(self, record: dict) -> None:
        self._write(self.eval_fp, record)

    def close(self) -> None:
        self.train_fp.close()
        self.eval_fp.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
