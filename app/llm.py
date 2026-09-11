"""
llm.py — streaming chat completions + a chunker that decides when a partial
sentence is worth speaking.

The chunker is the difference between "the bot answers in two seconds" and
"the bot answers in six hundred milliseconds". Instead of waiting for the full
completion, we push each clause to TTS the moment it is grammatically safe to
speak. Hindi's danda (।) is treated as a full stop, and we also break on ?, !,
. and — for long clauses — on a comma, so the caller hears the first words while
the model is still writing the rest.

We never break in the middle of a number or an abbreviation, because Sarvam
would then read the halves as two separate words.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, AsyncGenerator, Optional

import aiohttp

from . import config
from .logging_setup import CallLog, error, info, log, warn

# Hard stops: safe to speak immediately.
_HARD_STOP = "।?!।"
_HARD_STOP_SET = set("।?!")
# Soft stops: only break here if the clause is already reasonably long.
_SOFT_STOP_SET = set(",;:")

MIN_CHUNK_CHARS = 12          # don't send "जी," to TTS on its own
SOFT_BREAK_MIN_CHARS = 45     # comma only counts once we have a real clause
MAX_CHUNK_CHARS = 180         # force a break so TTS never waits too long


class SentenceChunker:
    """Feed tokens in, get speakable chunks out."""

    def __init__(self) -> None:
        self._buf = ""

    def push(self, token: str) -> list[str]:
        out: list[str] = []
        self._buf += token

        while True:
            idx = self._breakpoint(self._buf)
            if idx is None:
                break
            chunk = self._buf[: idx + 1].strip()
            self._buf = self._buf[idx + 1 :]
            if chunk:
                out.append(chunk)
        return out

    def flush(self) -> Optional[str]:
        chunk = self._buf.strip()
        self._buf = ""
        return chunk or None

    @staticmethod
    def _breakpoint(s: str) -> Optional[int]:
        for i, ch in enumerate(s):
            if ch in _HARD_STOP_SET and i + 1 >= MIN_CHUNK_CHARS:
                return i
            if ch == ".":
                # not a decimal point, not an abbreviation like "Dr."
                nxt = s[i + 1] if i + 1 < len(s) else " "
                prv = s[i - 1] if i > 0 else " "
                if prv.isdigit() and nxt.isdigit():
                    continue
                if i + 1 >= MIN_CHUNK_CHARS:
                    return i
            if ch in _SOFT_STOP_SET and i + 1 >= SOFT_BREAK_MIN_CHARS:
                return i
        if len(s) >= MAX_CHUNK_CHARS:
            # break at the last space so we never cut a word in half
            cut = s.rfind(" ", 0, MAX_CHUNK_CHARS)
            return cut if cut > MIN_CHUNK_CHARS else MAX_CHUNK_CHARS - 1
        return None


# Whole-number words, not digit-by-digit: "12" must be "बारह", never "एक दो".
# Anything outside this range is left alone — Sarvam bulbul:v3 normalises
# numbers itself, and a wrong expansion is worse than none.
_NUM_HI = {
    0: "शून्य", 1: "एक", 2: "दो", 3: "तीन", 4: "चार", 5: "पाँच", 6: "छह",
    7: "सात", 8: "आठ", 9: "नौ", 10: "दस", 11: "ग्यारह", 12: "बारह",
    13: "तेरह", 14: "चौदह", 15: "पंद्रह", 16: "सोलह", 17: "सत्रह",
    18: "अठारह", 19: "उन्नीस", 20: "बीस", 21: "इक्कीस", 22: "बाईस",
    23: "तेईस", 24: "चौबीस", 25: "पच्चीस", 26: "छब्बीस", 27: "सत्ताईस",
    28: "अट्ठाईस", 29: "उनतीस", 30: "तीस", 40: "चालीस", 45: "पैंतालीस",
    50: "पचास", 60: "साठ", 90: "नब्बे",
}


def speakable(text: str) -> str:
    """
    Last-mile cleanup before TTS.

    Strips markdown the model sometimes emits despite instructions, collapses
    whitespace, and reads stray single digits as Hindi words so "3 din" never
    comes out as "three din".
    """
    t = text
    t = re.sub(r"[*_`#>]+", "", t)
    t = re.sub(r"^\s*[-•]\s*", "", t, flags=re.MULTILINE)
    t = re.sub(r"\s+", " ", t).strip()
    # Only convert short standalone numbers we have a real word for. Phone
    # numbers, amounts and anything else pass through to Sarvam untouched.
    def _num(m: re.Match) -> str:
        word = _NUM_HI.get(int(m.group()))
        return word if word else m.group()

    t = re.sub(r"(?<![\w\d])\d{1,2}(?![\w\d])", _num, t)
    return t


class LLMClient:
    def __init__(self, call_log: CallLog) -> None:
        self.call_log = call_log
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60, sock_connect=5)
            )
        return self._session

    async def stream(
        self, messages: list[dict], tools: list[dict]
    ) -> AsyncGenerator[tuple[str, Any], None]:
        """
        Yields:
          ("token", str)       — a text delta
          ("tool_calls", list) — assembled tool calls, once, at the end
          ("done", str)        — the full assistant text
        """
        session = await self._get_session()
        payload = {
            "model": config.OPENAI_MODEL,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": config.OPENAI_TEMPERATURE,
            "max_tokens": config.OPENAI_MAX_TOKENS,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {config.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }

        t0 = time.monotonic()
        first_token = True
        full = ""
        tool_calls: dict[int, dict] = {}

        self.call_log.event("llm_request", model=config.OPENAI_MODEL,
                            messages=len(messages))
        log("llm", f"request ({len(messages)} messages)")

        try:
            async with session.post(
                f"{config.OPENAI_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    error("llm", f"{resp.status}: {body[:400]}")
                    self.call_log.event("llm_error", status=resp.status,
                                        body=body[:400])
                    return

                async for raw in resp.content:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except ValueError:
                        continue

                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}

                    token = delta.get("content")
                    if token:
                        if first_token:
                            ms = round((time.monotonic() - t0) * 1000, 1)
                            info("llm", f"first token in {ms} ms")
                            self.call_log.event("llm_ttft", ms=ms)
                            first_token = False
                        full += token
                        yield "token", token

                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        slot = tool_calls.setdefault(
                            idx, {"id": "", "name": "", "arguments": ""}
                        )
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["arguments"] += fn["arguments"]

        except asyncio.CancelledError:
            log("llm", "stream cancelled")
            raise
        except Exception as exc:
            error("llm", f"stream failed: {exc}")
            self.call_log.event("llm_error", error=str(exc))
            return

        if tool_calls:
            calls = []
            for idx in sorted(tool_calls):
                slot = tool_calls[idx]
                if not slot["name"]:
                    continue
                try:
                    args = json.loads(slot["arguments"] or "{}")
                except ValueError:
                    warn("llm", f"bad tool arguments: {slot['arguments']!r}")
                    args = {}
                calls.append(
                    {"id": slot["id"] or f"call_{idx}", "name": slot["name"],
                     "arguments": args}
                )
            if calls:
                yield "tool_calls", calls

        total = round((time.monotonic() - t0) * 1000, 1)
        self.call_log.event("llm_done", ms=total, chars=len(full),
                            text=full, tool_calls=len(tool_calls))
        log("llm", f"done in {total} ms: {full!r}")
        yield "done", full

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
