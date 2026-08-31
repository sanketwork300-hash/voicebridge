# Third-party licences

VoiceBridge is licensed under Apache-2.0 (see `LICENSE`).

**No third-party source code is copied or adapted into this repository.** The
projects below are either runtime dependencies used through their public APIs,
or references that were read to verify APIs and architecture. Detailed analysis:
`docs/reference-analysis.md`. Full matrix: `docs/license-matrix.md`.

Verified 2026-08-31.

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

### piper-tts — MIT
<https://github.com/rhasspy/piper> · TTS engine. **Voice models carry their own
separate licences and are not distributed here.**

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
| Whisper checkpoints (via WhisperLiveKit) | MIT |
| Piper voices | Per-voice; verify individually |

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
