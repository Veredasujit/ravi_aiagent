"""
messaging_service.py — the brain that decides *what* message to enqueue
when a call ends, and *how* the queue dispatches it.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from .booking_state import BookingRecord, BookingStatus, booking_registry
from .logging_setup import error, info, warn
from .message_queue import MessageKind, MessageQueue, OutboundMessage
from .whatsapp import build_message_for, send_outbound_message


class MessagingService:
    """
    Single place that owns the MessageQueue and exposes:
      - enqueue_for_completed_call(call_id) : decide + enqueue
      - start()/stop()                      : worker lifecycle
      - stats()                             : observability
    """

    def __init__(self) -> None:
        self.queue = MessageQueue(
            send_function=send_outbound_message,
            max_size=10000,
            worker_count=3,
            max_retries=3,
            retry_backoff=2.0,
            on_sent=self._on_sent,
            on_failed=self._on_failed,
        )
        self._started = False

    # ---------------- lifecycle ----------------
    async def start(self) -> None:
        if self._started:
            return
        await self.queue.start()
        self._started = True
        info("messaging", "service started")

    async def stop(self) -> None:
        if not self._started:
            return
        await self.queue.stop()
        self._started = False
        info("messaging", "service stopped")

    # ---------------- decision logic ----------------
    def _decide_kind(self, rec: BookingRecord) -> MessageKind:
        if rec.status == BookingStatus.CONFIRMED:
            return MessageKind.BOOKING_CONFIRMED
        if rec.status in (BookingStatus.NOT_BOOKED, BookingStatus.CANCELLED):
            return MessageKind.FOLLOWUP_NO_BOOKING
        # NOT_STARTED / COLLECTING → treat as no booking
        return MessageKind.FOLLOWUP_NO_BOOKING

    async def enqueue_for_completed_call(
        self, call_id: str, *, phone_number: Optional[str] = None
    ) -> bool:
        rec = booking_registry.get(call_id)
        if not rec:
            warn("messaging", f"no booking record for call_id={call_id}; skipping")
            return False
        if not rec.phone_number and phone_number:
            rec.phone_number = phone_number
        if not rec.phone_number:
            warn("messaging", f"no phone_number for call_id={call_id}; skipping")
            return False
        if rec.whatsapp_sent:
            info("messaging", f"already sent for call_id={call_id}; skipping")
            return False

        kind = self._decide_kind(rec)
        text = build_message_for(rec, kind)
        msg = OutboundMessage(
            kind=kind,
            phone_number=rec.phone_number,
            text=text,
            call_id=call_id,
            metadata={"booking_status": rec.status.value},
        )
        ok = await self.queue.enqueue(msg)
        if ok:
            info(
                "messaging",
                f"queued {kind.value} for call_id={call_id} phone={rec.phone_number}",
            )
        return ok

    # ---------------- queue callbacks ----------------
    async def _on_sent(self, msg: OutboundMessage, result: dict) -> None:
        msg_id = (
            result.get("message_id")
            or result.get("id")
            or result.get("data", {}).get("message_id")
            if isinstance(result, dict)
            else None
        )
        booking_registry.mark_whatsapp_sent(msg.call_id, msg_id)
        if msg.kind == MessageKind.BOOKING_CONFIRMED:
            booking_registry.mark_confirmed  # noop reference (kept explicit)
        info("messaging", f"delivered {msg.kind.value} call_id={msg.call_id}")

    async def _on_failed(self, msg: OutboundMessage, error_text: str) -> None:
        booking_registry.mark_whatsapp_failed(msg.call_id, error_text)
        error("messaging", f"gave up on {msg.kind.value} call_id={msg.call_id}: {error_text}")

    # ---------------- introspection ----------------
    def stats(self) -> dict:
        return self.queue.stats()


# module-level singleton
messaging_service = MessagingService()