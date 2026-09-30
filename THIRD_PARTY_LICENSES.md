# Third-party licences

VoiceBridge is licensed under Apache-2.0 (see `LICENSE`).

**No third-party source code is copied or adapted into this repository.** The
projects below are either runtime dependencies used through their public APIs,
or references that were read to verify APIs and architecture. Detailed analysis:
`docs/reference-analysis.md`. Full matrix: `docs/license-matrix.md`.

Verified 2026-08-31; updated 2026-09-30 for the v0.2 providers (licences read
from the installed package metadata and the Hugging Face model cards).

---

## Runtime dependencies

### WhisperLiveKit — Apache-2.0
<https://github.com/QuentinFuxa/WhisperLiveKit> · version 0.2.26, commit `b781ce9`

Optional dependency providing streaming ASR, used through its public API
(`TranscriptionEngine`, `AudioProcessor`). Copyright © Quentin Fuxa.
Licensed under the Apache License, Version 2.0.

### FastAPI (MIT) · Starlette (BSD-3-Clause) · Uvicorn (BSD-3-Clause)
### websockets (BSD-3-Clause) · NumPy (BSD-3-Clause) · PyYAML (MIT)

### PyTorch (BSD-3-Clause) · Transformers (Apache-2.0) · SentencePiece (Apache-2.0)
Optional, for local translation models.

### IndicTransToolkit — MIT
<https://github.com/VarunGumma/IndicTransToolkit> · required preprocessing for
IndicTrans2 checkpoints.

### piper-tts — **GPL-3.0-or-later** (versions ≥ 1.3)
The original <https://github.com/rhasspy/piper> (MIT, last release 1.2.0) is
archived. Current releases come from <https://github.com/OHF-Voice/piper1-gpl>;
the installed `piper-tts` 1.7.0 declares `License: GPL-3.0-or-later` in its
package metadata. It is an *optional* dependency that VoiceBridge imports at
runtime and does not redistribute; anyone distributing a build that bundles it
must comply with the GPL. **Voice models carry their own separate licences and
are not distributed here** — `en_US-lessac` (Blizzard 2013 research licence),
`en_US-ryan` and every `hi_IN` voice checked (CC-BY-NC-SA-4.0) are
non-commercial.

### faster-whisper — MIT · CTranslate2 — MIT · onnxruntime — MIT
Offline ASR for files and the bundled Silero VAD ONNX model (MIT).

### qwen-tts, qwen-asr — Apache-2.0
Run in a separate worker virtualenv (`.venv-workers/qwen`), used through their
public APIs (`Qwen3TTSModel`, `Qwen3ASRModel`).

### seamless_communication — MIT (code) · fairseq2 — MIT · SimulEval — CC-BY-SA-4.0
Run in a separate worker virtualenv (`.venv-workers/seamless`) for the optional
SeamlessStreaming engine. The Seamless **model weights are CC-BY-NC-4.0**.

### anthropic (Python SDK) — MIT
Optional, for the contextual translator's `anthropic` backend.

### sacrebleu — Apache-2.0 · psutil — BSD-3-Clause · SciPy — BSD-3-Clause · soundfile — BSD-3-Clause

### sounddevice (MIT) / PortAudio (MIT)
Optional, for microphone and system-audio capture.

### FFmpeg — LGPL-2.1-or-later / GPL depending on build
Invoked as an external subprocess, never linked. Optional; not needed for the
browser flow.

---

## Model weights (downloaded at runtime, never redistributed)

| Model | Licence |
| --- | --- |
| `Helsinki-NLP/opus-mt-ja-en`, `opus-mt-ko-en` | Apache-2.0 |
| `ai4bharat/indictrans2-*` | MIT |
| `facebook/nllb-200-*` | **CC-BY-NC-4.0 — non-commercial only** |
| Whisper large-v3-turbo (`openai/whisper-large-v3-turbo`, CT2 conversions) | MIT |
| `Qwen/Qwen3-ASR-1.7B` | Apache-2.0 |
| Silero VAD v6 (bundled in faster-whisper) | MIT |
| `MIT/ast-finetuned-audioset-10-10-0.4593` | BSD-3-Clause |
| `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B-Instruct-2507` (contextual translation) | Apache-2.0 |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, `Qwen/Qwen3-TTS-Tokenizer-12Hz` | Apache-2.0 |
| `facebook/seamless-streaming`, `facebook/seamless-m4t-v2-large` | **CC-BY-NC-4.0 — non-commercial only** |
| Piper voices | Per-voice; the voices checked for en/hi are non-commercial |

## Evaluation data (downloaded by scripts, never committed)

| Data | Licence |
| --- | --- |
| Google FLEURS (`google/fleurs`) | CC-BY-4.0 |
| Non-speech clips from Wikimedia Commons (`scripts/fetch_nonspeech.py`) | CC0 / public domain / CC-BY-3.0 / CC-BY-SA-4.0 per file; recorded in `ATTRIBUTION.json` |

IndicTrans2 citation:

> Gala et al., *IndicTrans2: Towards High-Quality and Accessible Machine
> Translation Models for all 22 Scheduled Indian Languages*, AI4Bharat.

---

## Architecture references (read, not copied)

| Project | Licence |
| --- | --- |
| Kami Subs | MIT |
| WhisperLive (Collabora) | MIT |
| AuraLang | MIT |
| Seamless Communication | MIT code / CC-BY-NC-4.0 models |
| IndicTrans2 | MIT |
| Realtime-Speech-to-Speech-Translation | see repository |
| PolyTalk CE | **AGPL-3.0** — incompatible; ideas only, no code |
| STT_extension | **No licence file** — no rights granted; no code used |

---

## Attribution

If you redistribute VoiceBridge, retain this file and `LICENSE`, and comply with
the terms of any model weights you bundle or serve — particularly non-commercial
restrictions on NLLB and Seamless.
