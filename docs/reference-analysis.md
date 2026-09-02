# Reference analysis

Every repository below was **cloned and read** before the corresponding part of
VoiceBridge was written. Where a reference does not implement something we need,
that is stated explicitly rather than assumed.

Inspection date: 2026-08-31. Where a commit is given it is the tip of the
default branch on that date. Licences were read from the repository itself, not
from a README badge.

---

## 1. WhisperLiveKit — primary streaming-ASR reference

* Repository: <https://github.com/QuentinFuxa/WhisperLiveKit>
* Version inspected: 0.2.26, commit `b781ce9334c8085131b2b7a146a61d4e22ba5af1`
* Licence: **Apache-2.0** (`LICENSE`)
* Status: **runtime dependency** (optional extra `asr-whisperlivekit`), adapted
  behind our own interface. No source copied.

### Why it is the ASR reference

It already implements the hard part: deciding *when* streaming ASR output is
committed. `whisperlivekit/local_agreement/` and `whisperlivekit/simul_whisper/`
hold LocalAgreement and SimulStreaming policies respectively, with VAD
(`silero_vad_iterator.py`), a voice-activity controller, optional diarization
and per-session buffering. Reimplementing that on top of fixed-size chunks
produces duplicated and truncated words at every boundary.

### API actually used (read from source, not the README)

| Symbol | File | Use in VoiceBridge |
| --- | --- | --- |
| `TranscriptionEngine(config=…)` | `core.py` | Process-wide model holder. **It is a singleton** — `__new__` returns the existing instance — so we share one across sessions. |
| `AudioProcessor(transcription_engine=…, language=…, mode=…, target_language=…, pcm_input=…)` | `audio_processor.py` | Per-session processor. |
| `await processor.create_tasks()` | `audio_processor.py:999` | Returns an async generator of `FrontData`. |
| `await processor.process_audio(bytes)` | `audio_processor.py:1105` | Feeds audio; an empty message signals end of stream. |
| `await processor.cleanup()` | `audio_processor.py:1067` | Teardown. |
| `FrontData.lines` | `timed_objects.py` | Cumulative `Segment`s. Committed *tokens*, but see below: the last line keeps growing. New text per line index → our `ASR_STABLE`. |
| `FrontData.buffer_transcription` | `timed_objects.py` | Unstable hypothesis tail → our `ASR_PARTIAL`. |
| `WhisperLiveKitConfig.from_kwargs` | `config.py:262` | Field names for `TranscriptionEngine(**kwargs)`. **Unknown keys are dropped with a log warning, not rejected.** The size field is `model_size`, not `model`; the language field is `lan`. |

`results_formatter` (`audio_processor.py:928`) is where the two are assembled.

### Corrections found by running it (2026-09-02)

The first version of this section was written from the source alone and got
two things wrong that only showed up when real audio was streamed through:

* **`lines` grouping is not commitment.** `tokens_alignment.py::get_lines`
  appends committed tokens to `current_line_tokens` and re-emits that as the
  *last* element of `lines` on every snapshot — same `start`, growing `end`
  and `text`. The line is moved to `validated_segments` only when a `Silence`
  token arrives, and `audio_processor.py` sets `MIN_DURATION_REAL_SILENCE = 5`
  seconds. Treating each snapshot's lines as final therefore re-emitted the
  whole growing sentence (duplicated text downstream); waiting for the line to
  close held every sentence back until a five-second pause. The adapter now
  forwards, per line index, only the text a line has grown by. The tokens
  themselves are immutable once committed (`local_agreement/online_asr.py`,
  `committed_in_buffer`); a revision is logged and counted, never re-emitted.
* **`model=` was silently ignored.** `from_kwargs` dropped it, so every
  deployment loaded the default `base` model whatever the YAML said.
* Silence gaps appear in `lines` as segments with `speaker == -2` and empty
  text; `-1` means "no diarization". `Segment.speaker` is annotated `str` but
  holds these ints.
* `backend_policy` selects the streaming policy: `simulstreaming` (default;
  faster-whisper encoder + a PyTorch decoder, which also downloads OpenAI's
  `small.pt` to `~/.cache/whisper`) or `localagreement` (faster-whisper /
  CTranslate2 end to end). On a 16-core CPU with no GPU, `small` +
  `simulstreaming` fell 24–60 s behind real time; `base` + `localagreement`
  produced the first committed text ~13 s after the audio and the last
  translation ~22 s after the end of a 9.6 s clip. Those are observations
  from one run, not benchmarks.

### What we reuse conceptually, not literally

* The **partial-vs-committed split**. We adopt the contract; we do not
  reimplement the commitment policy. Layering our own policy on top of a backend
  that already has one double-buffers and makes latency impossible to attribute.
* The **PCM-vs-FFmpeg split**. `ffmpeg_manager.py` prints install instructions
  and points at `--pcm-input` as the alternative. We always pass
  `pcm_input=True`: the browser extension already produces 16 kHz mono PCM, and
  requiring FFmpeg for the primary flow would break a clean-machine install.
* Cumulative snapshots. `lines` is the whole transcript each time, so our
  stabiliser de-duplicates on segment end time.

### What we do not reuse

* Its web UI and its WebSocket message shape (`{"lines": [...], "buffer_transcription": ...}`).
  VoiceBridge needs a typed, sequenced envelope carrying translation and TTS
  events too — see `docs/protocol.md`.
* Its built-in translation (`translation.py`, via the `nllw`/NLLB path). We need
  a provider abstraction across OPUS-MT, IndicTrans2 and NLLB with licence
  gating, so translation lives in our pipeline instead.

### Constraint discovered

`pyproject.toml` declares `requires-python = ">=3.11, <3.14"`. **WhisperLiveKit
cannot be installed on Python 3.14+.** VoiceBridge core supports 3.11+ and its
mock mode runs anywhere; the ASR extra is marked `python_version < '3.14'`.

---

## 2. Kami Subs — browser tab-capture reference

* Repository: <https://github.com/MohammdKopa/kami-subs>
* Licence: **MIT** (`LICENSE`)
* Status: architecture and API-sequence reference. No source copied.

The clearest small implementation of the MV3 capture path. Read
`extension/background.js` and `extension/offscreen.js`.

Confirmed sequence, which VoiceBridge follows:

1. Service worker calls `chrome.tabCapture.getMediaStreamId({targetTabId})`.
2. It creates an offscreen document with `reasons: ['USER_MEDIA']`.
3. The offscreen document calls `navigator.mediaDevices.getUserMedia` with
   `{audio: {mandatory: {chromeMediaSource: 'tab', chromeMediaSourceId: streamId}}}`.
4. It reconnects a **passthrough gain node** to `destination`, because capturing
   a tab detaches its audio from the speakers — without this the user hears
   silence. This is a non-obvious detail and the reason to read the source.
5. It resamples to 16 kHz, converts to `Int16`, and sends over a WebSocket.

### Where we deliberately differ

* Kami Subs uses `createScriptProcessor(4096, 2, 1)`, noting it is deprecated
  "but works reliably". We use an **AudioWorklet** instead: `ScriptProcessorNode`
  runs on the main thread and drops audio on busy pages, and video pages are
  always busy.
* It resamples in JS with a hand-written linear interpolator. We construct the
  `AudioContext` at `{sampleRate: 16000}` and let the browser resample, which is
  both better quality and cheaper.
* Its native-messaging host that spawns the Python backend is out of scope; we
  expect the gateway to be started by the user or by Docker.

---

## 3. WhisperLive (Collabora) — Chrome extension

* Repository: <https://github.com/collabora/WhisperLive>, `Audio-Transcription-Chrome/`
* Licence: **MIT**
* Status: corroborating reference for the manifest permission set.

Its `manifest.json` requests `storage`, `activeTab`, `tabCapture`, `scripting` —
which confirmed the minimum permission set. Note it is an older-style extension
that injects a preprocessor into the page rather than using an offscreen
document; we follow the offscreen approach instead, which is the current
supported MV3 route.

---

## 4. AuraLang — offscreen + worklet reference

* Repository: <https://github.com/CristinaFores/auralang>
* Licence: **MIT**
* Status: architecture reference. No source copied.

Read `manifest.json`, `public/capture-worklet.js` and `src/offscreen/`. Its
worklet is minimal — downmix to mono and `postMessage` the Float32 frame — which
confirmed the AudioWorklet route is viable inside an offscreen document. Its
manifest also demonstrates the `offscreen` + `sidePanel` permission combination
and a `wasm-unsafe-eval` CSP for local WASM inference.

VoiceBridge's worklet differs: it converts to Int16 **inside** the worklet and
emits fixed 100 ms frames, so the main thread never touches per-sample data and
frame size is predictable for the server's buffering.

AuraLang runs Whisper locally in the browser via WASM. VoiceBridge deliberately
does not: a server-side streaming backend is what makes larger models, GPU
acceleration and Japanese/Korean quality possible.

---

## 5. STT_extension — closest end-to-end shape

* Repository: <https://github.com/xu-0306/STT_extension>
* Licence: **NONE FOUND.** The repository contains no `LICENSE` file.
* Status: **architecture observation only. No code may be copied or adapted.**

This matters: absent an explicit licence, the work is under exclusive copyright
and grants no rights. Its structure (Chrome extension → WebSocket → FastAPI →
WhisperLiveKit → translation → subtitle overlay) is the closest existing analogue
to VoiceBridge's MVP and confirmed that the overall shape is sound, but nothing
from it has been reused.

---

## 6. PolyTalk — self-hosted architecture reference

* Repository: <https://github.com/PolyTalkIO/polytalk>
* Licence: **AGPL-3.0** (`LICENSE`)
* Status: **architecture reference only. No code copied or adapted.**

AGPL-3.0 is incompatible with VoiceBridge's Apache-2.0 licensing: linking or
copying AGPL code would force the whole work under AGPL, including for anyone
who runs it as a network service. We read its layout (`app/`, `stt/`, `tts/`,
`config/`, `docker-compose.yml`, `docker-compose.gpu.yml`, `tools/benchmarks/`)
and adopted three *ideas*, all independently implemented:

* mock mode as a first-class deployment profile, not a test double;
* CPU and GPU compose profiles;
* configurable buffering thresholds as deployment-time settings.

---

## 7. Seamless Communication — speech translation research reference

* Repository: <https://github.com/facebookresearch/seamless_communication>
* Licence: **three-tier**, verified by reading the repository:
  * code and the w2v-BERT 2.0 speech encoder — **MIT** (`MIT_LICENSE`)
  * the SeamlessM4T / SeamlessStreaming **models — CC-BY-NC 4.0** (`LICENSE`)
  * SeamlessExpressive — a separate **Seamless Licensing Agreement**
    (`SEAMLESS_LICENSE`) limited to "Noncommercial Research Uses"
* Status: research reference. **Not a dependency, not a default.**

SeamlessStreaming genuinely supports simultaneous S2ST/S2TT, which is the
closest published system to VoiceBridge's end goal. But its models are
non-commercial, so making it mandatory would impose that restriction on every
deployment. The provider interface leaves room for a Seamless backend; adding
one would be an opt-in extra gated like the NLLB provider.

---

## 8. IndicTrans2 — Indic translation

* Repository: <https://github.com/AI4Bharat/IndicTrans2>
* Licence: code **MIT** (`LICENSE`); the README's artifact table lists
  **model checkpoints as MIT** too — verified on the
  `ai4bharat/indictrans2-en-indic-dist-200M` model card. Training data carries
  separate terms (CC0/CC-BY-4.0 depending on the corpus).
* Status: **implemented provider** (`providers/translation/indictrans2.py`).

API read from `huggingface_interface/example.py`, not inferred:

* checkpoints load with `trust_remote_code=True` (the architecture ships with
  the model);
* text **must** pass through `IndicProcessor` from `IndicTransToolkit` —
  `preprocess_batch(batch, src_lang=…, tgt_lang=…)` before tokenising and
  `postprocess_batch(decoded, lang=…)` after. It performs script normalisation
  and entity restoration, so skipping it materially degrades output;
* language codes are FLORES-style (`eng_Latn`, `hin_Deva`), not ISO-639-1;
* checkpoints are directional: `en-indic`, `indic-en`, `indic-indic`.

**IndicTrans2 does not support Japanese or Korean.** Its capability object says
so and `supports()` returns `False`, so the pipeline never routes those pairs to
it.

---

## 9. Realtime-Speech-to-Speech-Translation — S2S concept reference

* Repository: <https://github.com/kensonhui/Realtime-Speech-to-Speech-Translation>
* Status: concept reference for the audio→ASR→translation→TTS→audio loop and
  virtual-audio-device usage.

Useful as a demonstration of the complete loop. Not used as a production
architecture: it is built around per-utterance processing rather than a
streaming commitment policy, which is the specific thing VoiceBridge needs from
WhisperLiveKit.

---

## Capabilities no reference provided

Stated explicitly, per the anti-hallucination requirement. These were designed
and implemented from first principles:

| Capability | Where | Why no reference applies |
| --- | --- | --- |
| Language-aware segmentation profiles (`ja`/`ko` sentence-final bias) | `core/segmentation/` | No reference repository conditions its translation trigger on source-language grammar. This is the core quality mechanism for Tier-1 pairs. |
| Glossary placeholder protection | `core/context/glossary.py` | None of the references implement terminology preservation. |
| Honorific policy | `core/context/honorifics.py` | Not present in any reference; implemented as a documented post-translation heuristic with a deliberate no-op on ambiguity. |
| TTS reorder scheduler with gap recovery | `core/scheduling/` | References play audio as it arrives; none handle out-of-order synthesis or a failed segment blocking the stream. |
| Bounded queues with per-stage overflow policies | `core/streaming/queue.py` | No reference declares backpressure policy per stage. |
| Language-detection stabilisation | `core/session/language.py` | No reference guards against per-chunk language flapping. |
| Typed, sequenced event protocol | `protocols/websocket/` | Reference protocols are ad-hoc and ASR-only. |
