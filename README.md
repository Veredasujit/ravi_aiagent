# Ravi — Capital Hospital, Yamunanagar voice agent

Inbound + outbound AI phone agent on a Teler (FreJun) number.

```
Teler WS ──► 8 kHz PCM ──┬──► Silero VAD   (is a human talking?)
                         └──► Deepgram     (what did they say?)

turn end ──► OpenAI (streaming) ──► clause chunker ──► Sarvam bulbul:v3
                                                            │
             Teler WS ◄── paced 20 ms playout ◄─────────────┘
```

No Pipecat. Direct WebSocket clients to every vendor, so there is nothing
between the phone line and the model that you cannot see in the logs.

---

## Knowledge base

Every protocol fact this build relies on — Teler message shapes, the Sarvam
config payload, audio format, endianness, the pacing rule — is written up with
its provenance in [`docs/KNOWLEDGE_BASE.md`](docs/KNOWLEDGE_BASE.md), marked
SOURCE / DOCS / MEASURED / UNVERIFIED. Read that before changing anything in
the audio path.

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then fill it in
```

`torch` is needed even in ONNX mode — Silero's Python wrapper uses the tensor
API. On a CPU-only VPS install the CPU wheel to save ~2 GB:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 2. Verify the vendors before you burn a phone call

```bash
python -m tools.selftest     # or: ./run.sh test
```

This opens each vendor connection independently and writes
`selftest_sarvam.wav`. **Listen to that file.** If it sounds clean at 8 kHz,
any voice problem you hear on the phone is transport or playout, not TTS —
and the per-call JSONL log will tell you which.

## 3. Run

```bash
./run.sh                          # uvicorn on :8000
ngrok http 8000                   # in another terminal
# put the ngrok host (no scheme, no slash) into PUBLIC_HOST, restart
```

Point your Teler number's flow URL at `https://$PUBLIC_HOST/flow`.

Outbound:

```bash
./run.sh call +91XXXXXXXXXX
```

Inbound and outbound run the identical flow, session, and prompt. The only
difference is who dialled.

---

## Endpoints

| Method | Path              | Purpose                                   |
|--------|-------------------|-------------------------------------------|
| POST   | `/flow`           | Teler asks what the call should do        |
| POST   | `/webhook`        | call lifecycle events (HMAC-verified)     |
| WS     | `/media-stream`   | bidirectional audio                       |
| POST   | `/call/outbound`  | place a call                              |
| GET    | `/health`         | config sanity + whether Silero loaded     |

---

## The two decisions that make it feel real

### Barge-in — when to stop talking

Silero alone gets fooled by fans, a TV, a scooter horn. So barge-in is two
gated stages:

1. Silero fires → we **arm** (sub-100 ms reaction).
2. Deepgram must return actual words within 1.2 s → we **cut**.

If no words follow, we disarm and keep talking. It was noise.

While the bot is speaking the bar rises, because the carrier echoes our own
audio back down the line and we must not interrupt ourselves:

| | idle | bot speaking |
|---|---|---|
| probability threshold | 0.55 | 0.72 |
| continuous speech needed | 130 ms | 260 ms |

The first `BARGE_IN_GRACE_MS` (350 ms) of our own utterance is ignored
outright — that window is where echo lives.

Underneath sits an **adaptive noise floor**: a frame only counts as speech if
it is `VAD_ENERGY_MARGIN_DB` (8 dB) above the running background level. Steady
hum raises the floor and stops triggering itself.

### Turn end — when to start talking

Whichever fires first wins: Deepgram `speech_final`, Deepgram `UtteranceEnd`,
or our own VAD silence timer. Each one alone has a failure mode; together they
don't.

---

## Why the audio is clean

**Zero resampling in the happy path.** Teler streams L16 @ 8 kHz. We configure
Deepgram for `linear16/8000` and Sarvam for `output_audio_codec=linear16`,
`speech_sample_rate=8000`. Audio goes Sarvam → phone line untouched: no MP3
decode, no rate conversion, no extra buffering. This single choice is worth
more than any downstream tuning.

**Audio is relayed in buffered chunks, not paced.** Teler runs its own jitter
buffer and synchronisation on the far side — FreJun's reference bridge does no
timing at all, it collects several TTS chunks and relays them as one message.
Pacing on top of that buffer starves it: delivering at exactly real time leaves
zero slack, and any jitter on the path makes the audio cut in and out. We buffer
to 250 ms (or 120 ms of wall time, whichever comes first), flush on
end-of-clause, and drop from ~50 messages/second to ~4. All sends go through a
lock so a threshold flush can't reorder against an end-of-clause flush.

What we keep is a *model* of playback rather than a driver of it: we track how
much audio has been handed over and estimate when it finishes. That tells us
when the farewell has actually been heard (so hangup doesn't clip it) and how
far into an utterance we are (for the barge-in grace window). Barge-in stays
instant — we drop our buffer and send Teler `clear`.

**The RIFF header is stripped** from Sarvam's first `linear16` chunk. Left in,
it is an audible click at the start of every reply.

**The LLM streams into TTS clause by clause.** The chunker breaks on `।`, `?`,
`!`, and on commas once a clause is long enough, so the caller hears the first
words while the model is still writing the rest.

---

## Logging

`IS_LOGGING=true` is the master switch (`isLogging` in `config.py`). Two outputs:

- `logs/agent.log` — human-readable, every line tagged with the call id.
  `grep <call_id> logs/agent.log` isolates one call from a busy server.
- `logs/calls/<call_id>.jsonl` — one JSON event per line with a relative
  timestamp. This is the file to look at when a call sounds wrong.

Latency marks recorded automatically:

| metric | meaning |
|---|---|
| `llm_ttft` | OpenAI first token |
| `tts_ttfb` | Sarvam first audio chunk |
| `user_speech_to_first_audio` | caller stopped → first byte queued to the line |

Set `LOG_AUDIO_FRAMES=true` to log every 32 ms VAD frame with its probability,
dBFS, and the current noise floor. Very noisy — use it only when tuning VAD.

---

## Troubleshooting, by symptom

| Symptom | Look at | Likely fix |
|---|---|---|
| No voice at all | `tts_connected`, `tts_ttfb` in the JSONL | Sarvam key or codec; run the self-test |
| Click at the start of replies | — | You are on a build without `strip_wav_header` |
| Robotic / tinny | `SARVAM_SAMPLE_RATE` | Must be 8000 with `linear16`, or you are resampling |
| Bot talks over the caller | `barge_ignored`, `barge_disarmed` | Lower `VAD_THRESHOLD_WHILE_SPEAKING` or set `BARGE_IN_REQUIRE_ASR=false` |
| Choppy / "buffering" outbound audio | `playout_drained` message count | Raise `PLAYOUT_FLUSH_BYTES`; check `sample_rate` is in the flow |
| Gaps *between* sentences | `playout_start` / `playout_drained` pairs mid-turn | Raise `PLAYOUT_PRIME_BYTES` (600 ms default) |
| Static instead of speech | `endian_detected` | Force `TELER_ENDIAN=big` or `little` |
| Bot cuts itself off on noise | `barge_in` with `source=vad` | Raise `VAD_THRESHOLD_WHILE_SPEAKING`, raise `VAD_ENERGY_MARGIN_DB` |
| Long pause before replies | `llm_ttft` vs `tts_ttfb` | Whichever is large is the culprit |
| Bot interrupts too eagerly | `barge_in` at low `speaking_ms` | Raise `BARGE_IN_GRACE_MS` |
| Turn never ends | no `turn_start` after `transcript` | Lower `DEEPGRAM_ENDPOINTING_MS` |
| Hinglish transcribed as Spanish | `transcript` lines | Set `DEEPGRAM_LANGUAGE=hi` |
| Booking not saved | `tool_result` `persisted:false` | `CLINIC_001_API_KEY` unset or `CLINIC_API_DRY_RUN=true` |

---

## Notes on your existing `tool_calling.py`

Three things there will not work against Teler:

1. Auth is `X-Api-Key`, not `Authorization: Bearer`.
2. `/calls/terminate` is not in the Teler SDK. This build ends calls by
   draining playout, then closing the media WebSocket — which Teler treats as
   hangup. That is why `end_call` is a tool the model calls *after* speaking
   the farewell, and `wait_drained()` guarantees the farewell finishes.
3. `create_data` returned success without persisting anything. Here it POSTs
   to the clinic API and reports `persisted: true/false` honestly, so a silent
   failure cannot look like a booking.

## Deployment

Run **one worker**. Each call holds three WebSockets plus an asyncio playout
clock; extra workers only fragment that state. Scale with more containers
behind a load balancer, not `--workers N`.

```bash
uvicorn app.server:app --host 0.0.0.0 --port 8000 --workers 1
```

Rough capacity on your existing VPS (2 vCPU): Silero ONNX costs ~0.3 ms per
32 ms frame, so VAD is not the limit — network and vendor concurrency quotas
are. Check Sarvam's streaming concurrency limit before promising volume.

---

## Verified against FreJun's own references

This build was cross-checked line by line against two authoritative sources:

- `frejun-tech/teler-py` — the official Python SDK (`streams.py`, `flows.py`,
  `clients.py`)
- `teler-sarvam-node-bridge` — FreJun's own Teler + Sarvam reference bridge

| | FreJun reference | this build |
|---|---|---|
| Teler base URL | `api.frejun.ai/api/v1` | same |
| Teler auth header | `X-Api-Key` | same |
| Hangup | `call_ws.close(code=1000)` | same |
| Inbound audio | `msg["data"]["audio_b64"]` | same (with fallbacks) |
| Outbound audio | `{type, audio_b64, chunk_id}` | same |
| Barge-in | `{"type":"clear"}` to Teler | same |
| Flow `sample_rate` | `"8k"` | same |
| Flow `chunk_size` | `500` | same |
| Outbound relay | buffer several chunks, no pacing | buffer 250 ms, no pacing |
| Sarvam codec | `linear16` @ 8000 | same |

Two deliberate differences:

1. **Barge-in cancels Sarvam by closing the socket** rather than sending
   `flush`. Closing is Sarvam's documented cancel; `flush` only forces out what
   is already buffered, so the interrupted sentence would keep playing. The
   cost is ~600 ms to reconnect, which we hide by reconnecting in the
   background.

2. **STT and LLM are Deepgram and OpenAI**, not Sarvam. The reference bridge
   uses `saaras:v3-realtime` and `sarvam-105b-conversations` end to end. If
   round-trip latency becomes the bottleneck, moving the LLM to
   `sarvam-105b-conversations` removes one international hop per turn — it sits
   in the same datacenter as the TTS. Sarvam's realtime STT
   (`wss://api.sarvam.ai/speech-to-text-realtime/ws`, `encoding=linear16`,
   `sample_rate=8000`, `endpointing=vad`) is a drop-in for the same reason.
   Both are worth measuring before switching: Deepgram `nova-3/multi` is
   currently transcribing Hinglish accurately here.
