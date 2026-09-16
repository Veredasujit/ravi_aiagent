"""
message_queue.py — async producer/consumer queue for WhatsApp messages.

Multiple bookings → multiple messages → processed by N background workers
with retry + backoff. Prevents blocking the media-stream handler and
protects the WhatsApp API from bursts.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

from .logging_setup import error, info, warn


class MessageKind(str, Enum):
    BOOKING_CONFIRMED = "booking_confirmed"
    FOLLOWUP_NO_BOOKING = "followup_no_booking"
    CALL_ENDED_GENERIC = "call_ended_generic"


@dataclass
class OutboundMessage:
    kind: MessageKind
    phone_number: str
    text: str
    call_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    attempts: int = 0
    enqueued_at: float = field(default_factory=time.time)
    next_attempt_at: float = field(default_factory=time.time)
    last_error: Optional[str] = None


class MessageQueue:
    """
    Bounded async queue with multiple workers, retry, and backoff.

    Producer (webhook / call session) → enqueue()
    Consumer (worker task)            → send_function(message)

    `send_function` must be an async callable returning a dict with at least
    {"message_id": "..."} on success, or raising on failure.
    """

    def __init__(
        self,
        *,
        send_function: Callable[[OutboundMessage], Awaitable[dict]],
        max_size: int = 10000,
        worker_count: int = 3,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        on_sent: Optional[Callable[[OutboundMessage, dict], Awaitable[None]]] = None,
        on_failed: Optional[Callable[[OutboundMessage, str], Awaitable[None]]] = None,
    ) -> None:
        self._queue: asyncio.Queue[OutboundMessage] = asyncio.Queue(maxsize=max_size)
        self._send = send_function
        self._worker_count = worker_count
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._on_sent = on_sent
        self._on_failed = on_failed
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._stats = {
            "enqueued": 0,
            "sent": 0,
            "failed": 0,
            "retried": 0,
            "dropped": 0,
        }

    # ---------------- lifecycle ----------------
    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        loop = asyncio.get_running_loop()
        for i in range(self._worker_count):
            self._workers.append(loop.create_task(self._worker(i), name=f"wa-worker-{i}"))
        info("queue", f"started {self._worker_count} workers")

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        # Drain remaining (with a cap so shutdown isn't stuck forever)
        try:
            await asyncio.wait_for(self._queue.join(), timeout=30)
        except asyncio.TimeoutError:
            warn("queue", "shutdown drain timed out; cancelling workers")
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        info("queue", f"stopped. stats={self._stats}")

    # ---------------- producer ----------------
    async def enqueue(self, msg: OutboundMessage) -> bool:
        try:
            self._queue.put_nowait(msg)
            self._stats["enqueued"] += 1
            info(
                "queue",
                f"enqueued {msg.kind.value} for {msg.phone_number} "
                f"(call={msg.call_id}, id={msg.message_id}, size={self._queue.qsize()})",
            )
            return True
        except asyncio.QueueFull:
            self._stats["dropped"] += 1
            error("queue", f"queue full — dropping {msg.kind.value} for {msg.phone_number}")
            return False

    # ---------------- consumer ----------------
    async def _worker(self, idx: int) -> None:
        while self._running:
            try:
                msg = await self._queue.get()
            except asyncio.CancelledError:
                break
            try:
                await self._handle(msg)
            except Exception as exc:
                error("queue", f"worker#{idx} unhandled: {exc}")
            finally:
                self._queue.task_done()

    async def _handle(self, msg: OutboundMessage) -> None:
        # honour backoff
        delay = msg.next_attempt_at - time.time()
        if delay > 0:
            await asyncio.sleep(delay)

        msg.attempts += 1
        try:
            result = await self._send(msg)
        except Exception as exc:
            msg.last_error = str(exc)
            if msg.attempts < self._max_retries:
                msg.next_attempt_at = time.time() + self._retry_backoff ** msg.attempts
                self._stats["retried"] += 1
                warn(
                    "queue",
                    f"retry {msg.attempts}/{self._max_retries} "
                    f"for {msg.phone_number}: {exc}",
                )
                await self._queue.put(msg)
            else:
                self._stats["failed"] += 1
                error(
                    "queue",
                    f"permanently failed for {msg.phone_number} "
                    f"after {msg.attempts} attempts: {exc}",
                )
                if self._on_failed:
                    await self._on_failed(msg, str(exc))
            return

        self._stats["sent"] += 1
        info("queue", f"sent {msg.kind.value} → {msg.phone_number} (id={msg.message_id})")
        if self._on_sent:
            await self._on_sent(msg, result or {})

    # ---------------- introspection ----------------
    def stats(self) -> dict[str, int]:
        s = dict(self._stats)
        s["pending"] = self._queue.qsize()
        return s