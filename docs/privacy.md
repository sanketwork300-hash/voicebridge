# Privacy

## Defaults

```yaml
save_audio: false
save_transcripts: false
save_translation: false
```

Nothing is written to disk unless you turn it on, per session. There is no
implicit history, no analytics and no telemetry.

## What exists, and for how long

| Data | Where | Lifetime |
| --- | --- | --- |
| Audio frames | In memory, in a bounded queue | Milliseconds to seconds; overwritten or dropped |
| Transcript | In memory, on the session object | Until the session ends |
| Context window | In memory, capped at 4 segments / 600 chars | Continuously trimmed |
| Metrics | In memory | Timings and counts only, never content |
| Session record | In memory | Until deleted or reaped (default 4 h) |

Deleting a session (`DELETE /v1/sessions/{id}`) drops all of it. There is no
database, and no filesystem persistence path unless recording is enabled.

## Local mode

With `LOCAL` providers, after models are downloaded **no network request is made
by the translation pipeline**. Audio never leaves the machine.

Components that may make network calls, all optional and all at setup time:

| Component | When |
| --- | --- |
| Hugging Face model download | First use of a model, then cached |
| Piper voice download | Manual; never automatic |
| A remote gateway URL | Only if you configure one |

Set `HF_HUB_OFFLINE=1` after the first run to guarantee no outbound calls.

## Recording, if you enable it

Enabling any `save_*` flag requires that you:

1. obtain consent from everyone recorded — in many jurisdictions recording a
   conversation without consent is unlawful;
2. show a visible recording indicator in your UI;
3. configure retention and deletion.

VoiceBridge provides per-session deletion. Retention policy is yours.

## Browser extension

Stores only your settings (`chrome.storage.local`): languages, mode, gateway
URL, subtitle preferences. **No audio and no transcripts are stored.** Nothing
is sent anywhere except the gateway you configure — by default `127.0.0.1`.

Captured tab audio exists only in the offscreen document's audio graph and is
streamed to your gateway. It is not written to disk and not sent to any third
party.

## GDPR-style considerations

If you deploy VoiceBridge as a service for others, you are the controller.
Audio and transcripts are personal data, and speech may be biometric data
depending on jurisdiction and processing. In-memory-only, no-persistence
operation is the default and materially reduces obligations — enabling recording
changes that.
