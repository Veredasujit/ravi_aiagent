# app/flow_metadata.py
"""
flow_metadata.py — in-memory cache of phone numbers Teler sends in /flow.

When Teler POSTs to /flow, it includes `from_number`, `to_number`, and
`direction`. We stash that keyed by call_id so CallSession can pick it up
when the WebSocket `start` frame lacks phone info (which is common).

Thread-safe; bounded size; not persistent across restarts (fine for our
use case since flow metadata is short-lived).
"""

from __future__ import annotations

from threading import Lock
from typing import Optional

_lock = Lock()
_store: dict[str, dict] = {}


def store(call_id: str, meta: dict) -> None:
    with _lock:
        _store[str(call_id)] = meta


def get(call_id: str) -> Optional[dict]:
    with _lock:
        return _store.get(str(call_id))


def pop(call_id: str) -> Optional[dict]:
    with _lock:
        return _store.pop(str(call_id), None)


def cleanup(max_entries: int = 5000) -> None:
    """Keep the cache bounded. Drop oldest ~20% if exceeded."""
    with _lock:
        if len(_store) <= max_entries:
            return
        to_drop = int(max_entries * 0.2)
        for k in list(_store.keys())[:to_drop]:
            _store.pop(k, None)