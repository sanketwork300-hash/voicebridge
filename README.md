# VoiceBridge

**Self-hostable, real-time speech-to-speech translation.**

VoiceBridge captures audio from a source you already have access to — a browser
tab, a microphone, system audio, a file — recognises it with streaming ASR,
translates it, and returns translated subtitles and optionally synthesised
speech, with latency measured rather than claimed.

It is a general-purpose translation **engine** with pluggable input and output
adapters, not a plugin for any particular website.

```
Browser tab ─┐                                                  ┌─ Subtitles
Microphone  ─┤                                                  ├─ Dubbed speech
System audio ┼─► Audio engine ─► Streaming ASR ─► Stabiliser ─►  ┼─ Virtual audio
File        ─┤        ▲          Segmenter ─► Translation ─►    ├─ WebSocket/API
Network     ─┘        │          Context ─► TTS ─► Scheduler ─► └─ File (SRT/WAV)
                      └───────────── bounded queues, metrics ────────┘
```

---

## Status

Alpha. What works today, verified by the test suite:

- ✅ Chrome MV3 extension captures tab audio and streams 16 kHz PCM over WebSocket
- ✅ Streaming ASR with a distinct partial / committed transcript contract
- ✅ Language-aware segmentation (Japanese and Korean wait for sentence-final grammar)
- ✅ Pluggable translation: OPUS-MT, IndicTrans2, NLLB, mock
- ✅ TTS with strict playback ordering and backlog accounting
- ✅ Session glossary, honorific policy, dual subtitles
- ✅ Mock mode — the full UI with no GPU, no models, no network
- ✅ Docker (mock / cpu / gpu profiles), 126 passing tests
- ⚠️ Latency targets are **not yet benchmarked on real audio** — see [Known limitations](#known-limitations)
- ⚠️ Desktop app and virtual-audio output are designed but not implemented

---

## Quick start (60 seconds, no GPU)

Mock mode runs the entire pipeline and UI with no models and no downloads.

```bash
git clone https://github.com/voicebridge/voicebridge
cd voicebridge
pip install -e .
voicebridge-gateway
```

Open <http://127.0.0.1:8000> and press **Start translation**. The built-in demo
page captures your microphone and shows the live subtitle flow.

With Docker instead:

```bash
docker compose -f deploy/compose/docker-compose.yml --profile mock up
```

Mock providers replay a fixed script and produce audible placeholder audio. They
prove the plumbing; they say nothing about recognition or translation quality.

---

## Browser extension

1. `chrome://extensions` → enable **Developer mode** → **Load unpacked**
2. Select `apps/browser-extension/`
3. Start the gateway (`voicebridge-gateway`)
4. Open the tab you want to translate, click the VoiceBridge icon, choose
   languages, press **Start translation**

Chrome grants tab capture only when the extension is invoked on the tab itself,
so click the icon **on the tab you want** rather than pinning and clicking
elsewhere. Restricted pages (`chrome://`, the Web Store) cannot be captured.

Capturing a tab normally detaches its audio from your speakers; VoiceBridge
reconnects it, so you keep hearing the original unless you ask it to duck or
mute.

---

## Going live (real models)

Edit `config/voicebridge.yaml` or set environment variables:

```yaml
providers:
  asr:
    provider: whisperlivekit
    model: large-v3-turbo      # use 'small' or 'base' on CPU
  translation:
    provider: opus_mt          # Apache-2.0, ja→en and ko→en
  tts:
    provider: piper
    voices:
      en: /models/piper/en_US-amy-medium.onnx
```

```bash
pip install 'voicebridge[asr-whisperlivekit,translation-local,tts-piper]'
voicebridge-gateway
```

GPU:

```bash
docker compose -f deploy/compose/docker-compose.yml --profile gpu up
```

> **Python version:** VoiceBridge core needs 3.11+ and its test suite passes on
> 3.11–3.14. WhisperLiveKit publishes `requires_python = ">=3.11,<3.14"`, so on
> Python 3.14+ the `asr-whisperlivekit` extra is **silently skipped** by its
> environment marker — `pip install` succeeds but streaming ASR is absent, and
> the provider then fails with an explicit `ProviderUnavailable`. Use Python
> 3.11–3.13 for real ASR.

---

## Supported inputs and outputs

| Input | Status |
| --- | --- |
| Browser tab (Chrome MV3 `tabCapture`) | ✅ implemented |
| Microphone | ✅ implemented (`sounddevice`) |
| System audio | ✅ via a monitor/loopback device — `voicebridge devices` flags them |
| Audio file (WAV/PCM native, others via FFmpeg) | ✅ implemented |
| WebRTC / network stream | ⛔ adapter interface exists, not implemented |

| Output | Status |
| --- | --- |
| Translated subtitles (overlay, SRT, VTT) | ✅ implemented |
| Dubbed speech (browser playback) | ✅ implemented |
| WAV file | ✅ implemented |
| WebSocket / API | ✅ implemented |
| Virtual audio device, OBS | ⛔ not implemented |

---

## Languages

Tier 1 (the benchmarked launch set): **ja→en, ko→en, en→hi**
Tier 2: en→ja, en→ko, en→mr, en→ta, en→te, en→bn

Nothing about these pairs is hard-coded — they are product priorities, not
capabilities. `GET /v1/languages` lists them; `GET /v1/providers` is the
authoritative capability list for whatever providers you have configured.

| Pair | Default provider | Model licence |
| --- | --- | --- |
| ja→en, ko→en | `opus_mt` (Marian) | **Apache-2.0** |
| en→hi and other Indic | `indictrans2` | **MIT** |
| anything (200 languages) | `nllb` | **CC-BY-NC-4.0 — non-commercial** |

NLLB is not the default despite its coverage, because its weights forbid
commercial use. It refuses to load until you explicitly acknowledge that. See
[docs/license-matrix.md](docs/license-matrix.md).

### Why Japanese and Korean get special handling

Both languages put tense, negation, politeness and question/statement mood at
the *end* of a sentence. Translating a clause before its ending does not merely
produce rougher output — it can assert the opposite of what the speaker said
("he will go" vs "he won't go"). VoiceBridge's segmenter therefore waits for a
sentence-final signal on `ja`/`ko` before translating, and low-latency mode
never disables that. This is the single most important quality mechanism in the
system; `docs/architecture.md` explains it in detail.

---

## Configuration

Precedence: environment variables → `config/voicebridge.yaml` → defaults.

| Variable | Purpose |
| --- | --- |
| `VOICEBRIDGE_HOST` / `_PORT` | Bind address |
| `VOICEBRIDGE_API_TOKEN` | Shared bearer token (**required** for any non-localhost bind) |
| `VOICEBRIDGE_WS_ORIGINS` | Comma-separated WebSocket origin allow-list |
| `VOICEBRIDGE_ASR_PROVIDER` etc. | Override provider choice |
| `VOICEBRIDGE_DEVICE` | `cpu`, `cuda`, `mps` |
| `VOICEBRIDGE_ALLOW_NC_MODELS` | Acknowledge non-commercial model licences |

### Latency tuning

Three profiles trade latency against completeness:

| Profile | Behaviour |
| --- | --- |
| `low_latency` | Shorter waits, more subtitle revisions |
| `balanced` | Recommended, and the default for ja/ko |
| `accurate` | Waits longer for complete sentences |

Segmentation thresholds are per language and fully configurable
(`voicebridge/core/segmentation/profiles.py`). **The shipped values are reasoned
defaults, not measured optima** — the benchmark harness exists to tune them.

---

## CLI

```bash
voicebridge serve                         # run the gateway
voicebridge translate talk.wav --source ja --target en --subtitles out.srt --dual
voicebridge devices                       # audio devices; flags system-audio monitors
voicebridge providers                     # providers, availability and model licences
```

---

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/sessions` | Create a session |
| `GET /v1/sessions/{id}` | Status, queue depths, latency |
| `DELETE /v1/sessions/{id}` | Stop and destroy |
| `GET /v1/languages` | Language and pair catalogue |
| `GET /v1/providers` | Providers, capabilities, model licences |
| `GET /health`, `/ready`, `/metrics` | Liveness, readiness, Prometheus metrics |
| `WS /v1/sessions/{id}/stream` | Bidirectional audio and events |

Full message specification: [docs/protocol.md](docs/protocol.md).

---

## Development

```bash
pip install -e '.[dev]'
pytest                       # 126 tests, ~8s, no models needed
ruff check voicebridge tests
```

The whole suite runs in mock mode, so CI needs no GPU and no downloads.

```
voicebridge/
├── core/          pipeline machinery (audio, transcript, segmentation,
│                  context, scheduling, session, metrics, streaming)
├── providers/     asr/ translation/ tts/ behind three ABCs
├── adapters/      input/ and output/
├── protocols/     WebSocket messages and handler
└── apps/          gateway (FastAPI) and CLI
apps/browser-extension/    Chrome MV3 client
```

Nothing under `core/` imports a concrete provider or an app — the dependency
direction is always inward.

---

## Benchmarking

```bash
python benchmarks/run_benchmark.py --help
```

Measures RTF, first-subtitle latency, committed-subtitle latency, first-audio
latency and end-to-end latency, with categories for clean speech, background
music, sound effects, overlapping speakers and rapid dialogue.

**No benchmark numbers are published in this repository yet.** The harness is
implemented; it has not been run against a real evaluation set on reference
hardware. Any figure you see quoted for VoiceBridge that is not reproducible
with this harness is not ours.

---

## Security and privacy

Defaults: `save_audio: false`, `save_transcripts: false`, `save_translation: false`.
Nothing is persisted unless you turn it on. Raw audio is never logged, and
transcripts are not logged by default.

A localhost deployment runs open. **Any network-reachable deployment must set
`VOICEBRIDGE_API_TOKEN` and `VOICEBRIDGE_WS_ORIGINS`** — the gateway warns at
startup if you bind publicly without them. Details in
[docs/security.md](docs/security.md) and [docs/privacy.md](docs/privacy.md).

VoiceBridge translates audio your browser or OS already provides to it. It does
not bypass DRM, authentication, paywalls or access controls, and it will not be
extended to.

---

## Licensing

VoiceBridge is **Apache-2.0**. Third-party components carry their own terms, and
some models are more restrictive than the code that loads them —
`GET /v1/providers` and `voicebridge providers` report each model's licence at
runtime. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) and
[docs/license-matrix.md](docs/license-matrix.md).

No model weights and no copyrighted media are distributed with this repository.

---

## Known limitations

Stated plainly rather than discovered later:

1. **No published latency numbers.** The sub-second P50 target is a design goal.
   It has not been demonstrated on reference hardware, so it is not claimed.
2. **Segmentation thresholds are unmeasured.** The ja/ko sentence-final logic is
   linguistically motivated and unit-tested, but its constants have not been
   tuned against real audio.
3. **Honorific preservation is a heuristic**, applied only when exactly one
   honorific appears in a segment. With two it deliberately does nothing.
4. **Mock providers are not models.** They demonstrate the system; they measure
   nothing.
5. **Diarization is off by default** and speaker labels should be shown only
   when the backend reports adequate confidence.
6. **No WebRTC input, no virtual-audio output, no desktop app** yet.
7. **Code-switching** (English inside Japanese or Korean speech) is handled only
   as well as the underlying ASR and NMT models handle it; there is no dedicated
   logic, deliberately, until tests justify it.
8. Translation context is passed only to providers that declare
   `supports_context`; Marian models do not, so ja/ko→en currently translates
   each segment independently.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The two rules that matter most: verify
claims against source before implementing against them, and never add an
unbounded queue.
