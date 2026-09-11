"""
logging_setup.py — one place that decides whether anything gets logged.

Two outputs:
  1. Human-readable console/file log (loguru), gated by config.isLogging.
  2. Machine-readable JSONL, one file per call, so you can replay a bad call
     event-by-event and see exactly where the latency or the silence came from.

Every log line carries the call_id, so grepping one call out of a busy server
is `grep <call_id> logs/agent.log`.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any, Optional

from loguru import logger

from . import config

_call_id_var: ContextVar[str] = ContextVar("call_id", default="-")

_configured = False


def setup_logging() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    logger.remove()

    if not config.isLogging:
        # Master switch off: keep only real errors, nothing else.
        logger.add(sys.stderr, level="ERROR", backtrace=False, diagnose=False)
        return

    fmt = (
        "<green>{time:HH:mm:ss.SSS}</green> "
        "<level>{level: <7}</level> "
        "<cyan>[{extra[call_id]}]</cyan> "
        "<magenta>{extra[tag]: <10}</magenta> "
        "{message}"
    )
    logger.configure(extra={"call_id": "-", "tag": "-"})
    logger.add(
        sys.stderr,
        level=config.LOG_LEVEL,
        format=fmt,
        backtrace=True,
        diagnose=False,
        enqueue=True,
    )

    os.makedirs(config.LOG_DIR, exist_ok=True)
    logger.add(
        os.path.join(config.LOG_DIR, "agent.log"),
        level=config.LOG_LEVEL,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <7} | "
            "[{extra[call_id]}] {extra[tag]: <10} | {message}"
        ),
        rotation="50 MB",
        retention="14 days",
        enqueue=True,
    )


def set_call_id(call_id: str) -> None:
    _call_id_var.set(call_id or "-")


def get_call_id() -> str:
    return _call_id_var.get()


def new_call_id() -> str:
    return uuid.uuid4().hex[:12]


def log(tag: str, message: str, level: str = "DEBUG") -> None:
    """Human-readable line. No-op when isLogging is False."""
    if not config.isLogging:
        return
    logger.bind(call_id=_call_id_var.get(), tag=tag).log(level, message)


def info(tag: str, message: str) -> None:
    log(tag, message, "INFO")


def warn(tag: str, message: str) -> None:
    log(tag, message, "WARNING")


def error(tag: str, message: str) -> None:
    # Errors always surface, even with logging off — you want to know.
    logger.bind(call_id=_call_id_var.get(), tag=tag).error(message)


class CallLog:
    """
    Per-call JSONL event stream: logs/calls/<call_id>.jsonl

    Each line is {"t": <seconds since call start>, "event": "...", ...fields}.
    This is the file to send us when a call sounds wrong.
    """

    def __init__(self, call_id: str) -> None:
        self.call_id = call_id
        self.t0 = time.monotonic()
        self._fh = None
        self._marks: dict[str, float] = {}
        if config.isLogging and config.LOG_JSONL:
            d = os.path.join(config.LOG_DIR, "calls")
            os.makedirs(d, exist_ok=True)
            try:
                self._fh = open(
                    os.path.join(d, f"{call_id}.jsonl"), "a", encoding="utf-8"
                )
            except OSError as exc:  # never let logging kill a call
                error("log", f"cannot open call log: {exc}")
                self._fh = None

    # ── events ──────────────────────────────────────────────────────────────
    def event(self, event: str, **fields: Any) -> None:
        if not config.isLogging:
            return
        rec = {
            "t": round(time.monotonic() - self.t0, 4),
            "call_id": self.call_id,
            "event": event,
        }
        rec.update(fields)
        if self._fh is not None:
            try:
                self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self._fh.flush()
            except OSError:
                pass

    # ── latency marks ───────────────────────────────────────────────────────
    def mark(self, name: str) -> None:
        """Record a timestamp you can measure against later."""
        self._marks[name] = time.monotonic()

    def since(self, name: str) -> Optional[float]:
        t = self._marks.get(name)
        return None if t is None else round((time.monotonic() - t) * 1000, 1)

    def latency(self, name: str, from_mark: str, **fields: Any) -> None:
        ms = self.since(from_mark)
        if ms is None:
            return
        self.event("latency", metric=name, ms=ms, **fields)
        log("latency", f"{name} = {ms} ms" + (f" {fields}" if fields else ""), "INFO")

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
