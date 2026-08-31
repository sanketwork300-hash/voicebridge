# WebSocket protocol

Executable form: `voicebridge/protocols/websocket/messages.py`. Keep both in
sync; the extension's `common/protocol.js` mirrors the constants.

## Connection

```
POST /v1/sessions          →  { "session_id": "...", "stream_url": "..." }
WS   /v1/sessions/{id}/stream
```

Sessions are created over HTTP and streamed over WebSocket. Splitting them means
configuration errors surface as an HTTP status with a readable body, rather than
an opaque socket close.

### Authentication

`Authorization: Bearer <token>`, or `?token=<token>` on the WebSocket URL. The
query fallback exists because the browser WebSocket API cannot set headers; it
is more exposed (proxy logs, history), which is why `docs/security.md`
recommends short-lived tokens for remote mode.

Close codes:

| Code | Meaning |
| --- | --- |
| 4401 | Invalid or missing token |
| 4403 | Origin not in the allow-list |
| 4404 | Unknown session |
| 4429 | Rate limited |

## Framing

| Frame | Contains |
| --- | --- |
| **Binary** | Raw PCM audio. No envelope. |
| **Text** | A JSON control message with a `type` field. |

Audio is sent as bare binary because wrapping it in JSON+base64 inflates it by a
third and adds encode/decode cost to the hottest path in the system.

Server→client audio *is* base64-in-JSON, deliberately: it is far lower volume,
and keeping it in the ordered event channel guarantees it interleaves correctly
with the subtitle events it belongs with.

### Audio format

PCM `s16le`, 16 kHz, mono, little-endian. Declare anything else in
`SESSION_CONFIG` and the server converts on entry.

Limits: **256 KiB** per audio frame, **64 KiB** per control frame, and frames
must be sample-aligned (even byte length). A misaligned frame is rejected rather
than padded, because padding injects a click into every frame.

---

## Client → server

### `SESSION_CONFIG`
```json
{ "type": "SESSION_CONFIG", "input": { "sample_rate": 48000, "channels": 2 } }
```
Declares the format of subsequent binary frames.

### `AUDIO_START` / `AUDIO_STOP`
```json
{ "type": "AUDIO_START" }
```
`AUDIO_STOP` flushes the pipeline and closes the socket. The session object
survives until deleted over HTTP.

### `PAUSE` / `RESUME`
Pausing discards incoming audio rather than buffering it — "pause" means stop
listening, not accumulate a backlog to replay.

### `PING`
```json
{ "type": "PING", "echo": 1730000000 }
```
Answered with `PONG` carrying the same `echo`.

---

## Server → client

Every event carries this envelope:

```json
{
  "event_type": "TRANSLATION_FINAL",
  "session_id": "uuid",
  "sequence_id": 42,
  "timestamp": 1730000000.123
}
```

`sequence_id` is monotonic per session, so a client can order and de-duplicate
without relying on arrival order.

### `SESSION_STARTED`
Carries the resolved `config` and a `providers` description including each
model's licence.

### `ASR_PARTIAL` — **provisional, may be retracted**
```json
{ "event_type": "ASR_PARTIAL", "text": "I think the main", "start": 0.0, "end": 1.2, "language": "en" }
```
Display it as visually distinct from committed text. **Never synthesise it.**

### `ASR_STABLE` — committed source text
```json
{ "event_type": "ASR_STABLE", "text": "I think the main reason is that we waited.",
  "start": 0.0, "end": 2.4, "speaker": null, "language": "en" }
```

### `TRANSLATION_FINAL` — committed translation
```json
{
  "event_type": "TRANSLATION_FINAL",
  "source_text": "五条悟の領域展開は本当に印象的でした。",
  "translated_text": "Satoru Gojo's Domain Expansion was truly striking.",
  "source_language": "ja", "target_language": "en",
  "start": 12.4, "end": 15.1, "speaker": null,
  "translation_sequence": 7, "committed": true
}
```
`translation_sequence` is the ordering key shared with `TTS_AUDIO`.

### `TRANSLATION_PARTIAL` — revisable translation
Reserved for revisable subtitles in low-latency subtitle-only mode. Clients must
tolerate it replacing a previously shown partial. It is never a TTS input.

### `TTS_AUDIO` — synthesised speech, in order
```json
{
  "event_type": "TTS_AUDIO",
  "audio": "<base64 PCM s16le>",
  "encoding": "pcm_s16le", "sample_rate": 22050, "channels": 1,
  "duration": 2.8, "translation_sequence": 7,
  "source_start": 12.4, "source_end": 15.1, "text": "..."
}
```
Segments arrive in strictly increasing `translation_sequence`. A gap means a
segment was skipped after a synthesis failure; play what arrives.

`source_start`/`source_end` are positions on the *source* sample clock, which is
what lets a client align the dub to the original media.

### `LANGUAGE_DETECTED`
```json
{ "event_type": "LANGUAGE_DETECTED", "language": "ja", "confidence": 0.98, "stable": true }
```
Only emitted when detection has **stabilised**. VoiceBridge does not flip
language on a single noisy window.

### `SPEAKER_CHANGED`
Only when diarization is enabled and confident.

### `WARNING` — degraded, still running
```json
{ "event_type": "WARNING", "code": "tts_backlog",
  "message": "Dubbed audio is 6.2s behind…", "repeats": 4 }
```
Throttled to one per `code` per 10 seconds; `repeats` reports suppressed counts,
so a sustained condition is visible without flooding the channel.

Codes: `audio_backlog`, `translation_backlog`, `tts_backlog`,
`translation_unavailable`, `tts_error`, `asr_rollback`.

### `ERROR`
```json
{ "event_type": "ERROR", "code": "protocol_error", "message": "...", "recoverable": true }
```
`recoverable: true` leaves the socket usable. `false` means the session is over.

### `SESSION_ENDED`
Carries a final `metrics` summary.

---

## Reconnection

The socket is expected to drop; the session is not destroyed with it. Clients
should reconnect with exponential backoff (the extension caps at 15 s) and may
resume streaming to the same `session_id`.

The reference client creates a *fresh* session on reconnect so the transcript
starts clean rather than resuming a stale buffer. Both are valid; the server
supports either.

Sessions outlive their sockets until deleted or reaped
(`max_session_seconds`, default 4 h).

---

## Client checklist

1. Treat `ASR_PARTIAL` and `TRANSLATION_PARTIAL` as provisional; style them differently.
2. Order by `sequence_id`, never arrival.
3. Play `TTS_AUDIO` by `translation_sequence`; schedule each after the previous.
4. Surface `WARNING`; do not silently discard it.
5. Reconnect with backoff; do not hot-loop.
6. Delete the session over HTTP when finished.
