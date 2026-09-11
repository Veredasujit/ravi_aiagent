# Knowledge base — Teler + Sarvam

Everything this build relies on, with where each fact came from and how sure we
are of it. Provenance matters here: several rounds of debugging were lost to
marketing pages and stale SDK docstrings that turned out to disagree with the
running service.

Confidence key:
- **SOURCE** — read directly from SDK/reference source code. Trust it.
- **DOCS** — from official API reference pages. Usually right.
- **MEASURED** — observed on a live call or in the self-test. Trust it most.
- **UNVERIFIED** — could not confirm; flagged so nobody assumes it.

---

## 1. Teler (FreJun)

### Endpoints and auth

| Fact | Value | Source |
|---|---|---|
| Base URL | `https://api.frejun.ai/api/v1` | SOURCE `teler/constants.py` |
| Auth header | `X-Api-Key: <key>` | SOURCE `teler/clients.py` |
| Create call | `POST /voice/calls/initiate` | SOURCE `teler/resources/calls.py` |
| Call params | `from_number`, `to_number`, `flow_url`, `status_callback_url`, `record` | SOURCE |
| Webhook signature header | `X-Teler-Signature` | DOCS (platform dashboard) |

`Authorization: Bearer` is **wrong** for Teler and was the reason hangup never
worked in the original `tool_calling.py`. There is no documented
`/calls/terminate` endpoint in the SDK.

### Call flow

`CallFlow.stream()` in the Python SDK returns only three fields:

```python
{"action": "stream", "ws_url": ..., "chunk_size": 400, "record": True}
```
SOURCE `teler/flows.py` — default `chunk_size` is 400.

FreJun's Sarvam bridge sends a fourth field, `sample_rate`, as the **string**
`"8k"`, and uses `chunk_size: 500`:

```javascript
res.json({action:'stream', ws_url:…, sample_rate: config.telerSampleRate, chunk_size: 500, record: false});
```
SOURCE `teler-sarvam-node-bridge/src/api/endpoints/calls.ts`, and its README
documents `TELER_SAMPLE_RATE=8k`.

We send `sample_rate` because the Sarvam-specific bridge is the closer
reference. **UNVERIFIED:** whether Teler requires it, ignores it, or whether
omitting it changes the negotiated rate.

### Media stream protocol

Inbound (Teler → us):
```json
{"type":"audio","data":{"audio_b64":"<base64>"}}
{"type":"start","call_id":"cs_..."}
{"type":"dtmf","data":{"digit":"1"}}
{"type":"stop"}
```
SOURCE — both `streamHandlers.ts` and the PyPI ElevenLabs sample read
`msg["data"]["audio_b64"]` and `control?.call_id`.

Outbound (us → Teler):
```json
{"type":"audio","audio_b64":"<base64>","chunk_id":1}
{"type":"clear"}
```
SOURCE — note the asymmetry: inbound nests audio under `data`, outbound does
not. `chunk_id` increments from 1.

Hangup: **close the WebSocket with code 1000.** SOURCE `teler/streams.py`,
`StreamOp.STOP` → `await call_ws.close(code=1000, ...)`. There is no hangup
message and no REST call.

### Audio format

- **L16, 8000 Hz, mono** — DOCS (frejun.ai landing page: "raw audio frames
  (L16/8000Hz)")
- **Little-endian** — MEASURED. RFC 2586 defines L16 as big-endian, but the
  live stream measured little-endian (roughness 0.349 little vs 0.87 big). The
  detector in `audio.py` re-measures every call; set `TELER_ENDIAN=little` to
  skip the check.

### Outbound pacing — the important one

**Do not pace.** Teler runs its own jitter buffer and synchronisation.

FreJun's reference relays without any timing: it accumulates
`SARVAM_MESSAGE_BUFFER_SIZE` (default 5) TTS chunks, concatenates them, sends
one message, and flushes the remainder on Sarvam's `final` event. No clock, no
frame timer. SOURCE `streamHandlers.ts`.

MEASURED: a 20 ms real-time pacer emitting 50 messages/second produced audio
the caller described as *"कटती हुई … बहुत buffering"* — cutting out with heavy
buffering. Pacing at exactly real time leaves zero slack, so any jitter starves
the far-side buffer. This build buffers to 250 ms (or 120 ms wall time) and
sends ~4 messages/second.

---

## 2. Sarvam TTS

`wss://api.sarvam.ai/text-to-speech/ws?model=bulbul:v3`
Header: `api-subscription-key`

Messages: `config` → `text` → `flush`. Optional `ping` keepalive.
Responses: `{"type":"audio","data":{"audio":"<base64>"}}`, `{"type":"event",...}`,
`{"type":"error","data":{"message":...,"code":422}}`.

### Config payload

FreJun's bridge sends (SOURCE `ttsClient.ts`):
```json
{"type":"config","data":{
  "model":"bulbul:v3","language_code":"...","speaker":"shubh",
  "speech_sample_rate":"8000","output_audio_codec":"linear16",
  "pace":1.0,"temperature":0.6,"min_buffer_size":30,"max_chunk_length":150}}
```

The official Python SDK sends all 14 fields including explicit nulls
(`model:null`, `temperature:null`, `dict_id:null`) because `_send_model()` calls
`.dict()` without `exclude_none`. SOURCE `sarvamai/text_to_speech_streaming/`.

**MEASURED:** an 8-field payload omitting the optional keys was rejected with
`422 "Input parameters has to be a valid dictionary"` on every codec. The
14-field SDK-shaped payload works. Which specific key mattered is
**UNVERIFIED** — the fix was applied wholesale.

### Model differences (DOCS)

| | bulbul:v2 | bulbul:v3 |
|---|---|---|
| pitch / loudness | supported | **not supported** |
| pace range | 0.3–3.0 | 0.5–2.0 |
| temperature | no | yes |
| default sample rate | 22050 | 24000 |
| preprocessing | opt-in | always on |

Because v3 preprocessing is always on, it normalises numbers itself — which is
why `speakable()` only converts numbers it has a correct whole word for and
leaves everything else alone.

### Codec

`linear16 @ 8000` — MEASURED working, confirmed by probe (8800 B). This is the
single most valuable choice in the stack: it matches Teler exactly, so audio
goes Sarvam → phone line with no resample and no MP3 decode. TTFB 249 ms.

The SDK docstring claims `output_audio_codec` "currently supports MP3 only".
That is **stale** — linear16 works.

**Barge-in:** there is no cancel message. Closing the socket is the documented
way to stop generation. FreJun's bridge sends `flush` instead, which only
forces out what is already buffered — the interrupted sentence keeps playing.
We close and reconnect in the background (~600 ms, hidden).

---

## 3. Sarvam STT (not currently used — Deepgram is)

`wss://api.sarvam.ai/speech-to-text-realtime/ws` — DOCS

Query: `language_code` (required), `model` (`saaras:v3-realtime` default),
`encoding` (`linear16` default), `sample_rate` (`8000` or `16000`; anything
else closes with code 4000), `endpointing` (`vad` default),
`silence_duration_ms` (500), `min_speech_duration_ms` (250), `threshold` (0.3),
`stream_type` (`fast` / `balanced` / `simulated`), `mode` (`transcribe` /
`codemix` / `translit` / …).

`mode=codemix` is the interesting one for Hinglish — it returns mixed
native+English output, which is exactly what callers speak here.

Drop-in for Deepgram: `encoding=linear16&sample_rate=8000` matches our stream
byte for byte. Worth measuring before switching — Deepgram `nova-3/multi` is
MEASURED transcribing Hinglish accurately (`नमस्ते, यह एक test है.`).

---

## 4. Sarvam LLM (not currently used — OpenAI is)

`POST https://api.sarvam.ai/v1/chat/completions` — DOCS
Auth: `api-subscription-key` header or `Authorization: Bearer`.

Models: `sarvam-105b` (128K, reasoning/agentic) and
`sarvam-105b-conversations` (32K, built for real-time voice).

OpenAI-compatible: `messages`, `stream`, `tools`, `tool_choice`,
`response_format`, `temperature`, `max_tokens`. Also `reasoning_effort`
(`low`/`medium`/`high`) and `wiki_grounding`.

The migration case: `sarvam-105b-conversations` is colocated with Sarvam TTS,
removing one international round trip per turn. MEASURED baseline to beat —
OpenAI `gpt-4o-mini` warm TTFT 779–1300 ms from Bihar. Tool-calling is
supported, so `create_data` / `end_call` should port unchanged, but that is
**UNVERIFIED** against their implementation.

---

## 5. Latency budget (MEASURED)

| Stage | Observed | Notes |
|---|---|---|
| Deepgram connect | 1.7–4.4 s | prewarm, parallel with TTS |
| Sarvam socket + probe | 0.6–1.0 s | prewarm |
| LLM TTFT (cold) | 1.2–1.7 s | fresh TLS |
| LLM TTFT (warm) | 0.78–1.3 s | session reused |
| Sarvam TTS TTFB | 0.24–0.27 s | consistently good |
| **user speech → first audio** | **2.0–2.7 s** | the number that matters |

Under 1.5 s feels conversational; above 2 s callers start talking over the bot.
The LLM is the dominant term, which is the argument for
`sarvam-105b-conversations`.

---

## 6. Things that cost time, so they don't again

1. **Marketing pages are not specs.** "L16/8000Hz" was right about the format
   and wrong-by-omission about endianness and pacing.
2. **SDK docstrings go stale.** "MP3 only" was false.
3. **Read the reference implementation, not the docs.** Every genuine fix in
   this project — the config shape, the pacing, `sample_rate` in the flow —
   came from reading FreJun's and Sarvam's actual source, not their docs.
4. **A 422 that repeats identically across four different codecs is not about
   the codec.** Vary one thing at a time and check whether the error changes.
5. **Test a speech detector with speech.** Silero correctly rejects sine waves;
   testing it with one proves nothing.
