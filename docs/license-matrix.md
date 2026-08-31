# Licence matrix

VoiceBridge is **Apache-2.0**. Third-party components carry their own terms, and
**model weights are frequently more restrictive than the code that loads them.**
That gap is the single most common licensing mistake in ML projects, so it is
tracked explicitly here.

Verified 2026-08-31 by reading each repository's licence file and each model
card. Licences change: re-verify before redistributing.

> "Open source" does not mean "compatible with our licence", and a permissively
> licensed inference library does not make its model weights permissive.

---

## Reference repositories

| Project | Licence | Code copied? | Adapted? | Studied? | Notes |
| --- | --- | :-: | :-: | :-: | --- |
| [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) | **Apache-2.0** | ❌ | ❌ | ✅ | Optional runtime dependency, used through its public API. Apache-2.0 is compatible; attribution retained in `THIRD_PARTY_LICENSES.md`. |
| [Kami Subs](https://github.com/MohammdKopa/kami-subs) | **MIT** | ❌ | ❌ | ✅ | MV3 tab-capture *sequence* reference. Our implementation differs (AudioWorklet, browser-side resampling). |
| [WhisperLive](https://github.com/collabora/WhisperLive) | **MIT** | ❌ | ❌ | ✅ | Manifest permission set corroboration. |
| [AuraLang](https://github.com/CristinaFores/auralang) | **MIT** | ❌ | ❌ | ✅ | Offscreen + worklet viability. |
| [STT_extension](https://github.com/xu-0306/STT_extension) | **NONE** | ❌ | ❌ | ✅ | ⚠️ **No licence file.** Exclusive copyright, no rights granted. Architecture observation only — nothing may be copied. |
| [PolyTalk CE](https://github.com/PolyTalkIO/polytalk) | **AGPL-3.0** | ❌ | ❌ | ✅ | ⚠️ Incompatible with Apache-2.0. Copying or linking would force the whole work under AGPL, including for network use. Ideas only (mock mode, compose profiles), independently implemented. |
| [Seamless Communication](https://github.com/facebookresearch/seamless_communication) | code **MIT**; models **CC-BY-NC-4.0**; Expressive under a separate agreement | ❌ | ❌ | ✅ | Research reference. Not a dependency. |
| [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) | code **MIT**, checkpoints **MIT** | ❌ | ❌ | ✅ | API learned from `huggingface_interface/example.py`; provider written independently. |
| [Realtime-S2S-Translation](https://github.com/kensonhui/Realtime-Speech-to-Speech-Translation) | see repo | ❌ | ❌ | ✅ | Concept reference only. |

**No third-party source code has been copied or adapted into VoiceBridge.**
Every reference was used to verify APIs and architecture.

---

## Runtime dependencies

| Package | Licence | Required? |
| --- | --- | --- |
| FastAPI, Starlette | MIT / BSD-3 | ✅ core |
| Uvicorn | BSD-3 | ✅ core |
| websockets | BSD-3 | ✅ core |
| NumPy | BSD-3 | ✅ core |
| PyYAML | MIT | ✅ core |
| WhisperLiveKit | Apache-2.0 | optional (`asr-whisperlivekit`) |
| PyTorch | BSD-3 | optional |
| Transformers | Apache-2.0 | optional |
| SentencePiece | Apache-2.0 | optional |
| IndicTransToolkit | MIT | optional (`translation-indic`) |
| piper-tts | MIT | optional (`tts-piper`) |
| sounddevice / PortAudio | MIT | optional (`audio-device`) |
| FFmpeg | LGPL-2.1+ / GPL depending on build | optional, invoked as an external process |

FFmpeg is executed as a **subprocess**, never linked, so its terms do not
propagate. It is not required for the browser flow, which sends raw PCM.

---

## Model weights — read this before deploying

| Model | Licence | Commercial | Default? |
| --- | --- | :-: | :-: |
| `Helsinki-NLP/opus-mt-ja-en` | **Apache-2.0** | ✅ | ✅ ja→en |
| `Helsinki-NLP/opus-mt-ko-en` | **Apache-2.0** | ✅ | ✅ ko→en |
| `ai4bharat/indictrans2-*` | **MIT** | ✅ | ✅ Indic |
| `facebook/nllb-200-*` | **CC-BY-NC-4.0** | ❌ | ❌ gated |
| SeamlessM4T / SeamlessStreaming | **CC-BY-NC-4.0** | ❌ | ❌ not integrated |
| SeamlessExpressive | Seamless Licensing Agreement, noncommercial research | ❌ | ❌ not integrated |
| Whisper (OpenAI) | MIT | ✅ | via WhisperLiveKit |
| Piper **engine** | MIT | ✅ | ✅ |
| Piper **voices** | ⚠️ **per voice** — varies (CC0, CC-BY, dataset-specific) | ⚠️ check each | ❌ none shipped |

### Why NLLB is not the default

NLLB-200 covers all Tier-1 and Tier-2 pairs in a single model, which is
technically attractive. Its weights are CC-BY-NC-4.0, so making it the default
would silently impose a non-commercial restriction on every deployment.

The provider therefore refuses to load until the operator acknowledges it:

```yaml
translation:
  provider: nllb
  acknowledge_non_commercial: true      # or VOICEBRIDGE_ALLOW_NC_MODELS=1
```

For commercial use choose `opus_mt` (Apache-2.0) or `indictrans2` (MIT).

### Piper voices

The engine is MIT; **individual voice models are not**. They are trained on
different corpora with different terms. VoiceBridge ships no voices and
downloads none automatically — you point at a file and are responsible for its
licence. Check the voice's card at
<https://huggingface.co/rhasspy/piper-voices> before commercial use.

---

## What this repository does *not* distribute

* No model weights of any kind.
* No copyrighted anime, drama, music or other entertainment audio. Test fixtures
  are synthetically generated at test time.
* No third-party source code.

---

## Runtime licence visibility

Licensing is queryable rather than buried in docs:

```bash
voicebridge providers
curl localhost:8000/v1/providers
```

Both report each provider's `model_license` and `commercial_use`.

---

## Adding a dependency or model

1. Read the actual `LICENSE` file — not a README badge, not the package index.
2. For models, read the **model card**; weights and code often differ.
3. Confirm Apache-2.0 compatibility. **AGPL and GPL are not compatible.** No
   licence at all means no rights — do not use it.
4. Add a row here and to `THIRD_PARTY_LICENSES.md`.
5. Set `model_license` and `commercial_use` on the provider's capabilities.
6. If it is non-commercial, gate it behind explicit acknowledgement as `nllb` is.
