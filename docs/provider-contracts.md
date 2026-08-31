# Provider contracts

Three ABCs in `voicebridge/providers/base.py`. The pipeline imports only these;
no module under `core/` imports a concrete provider.

## Capability declaration

Providers differ in what they can do, and the pipeline must adapt rather than
assume. Each publishes a capabilities object, queried by the pipeline and
exposed at `GET /v1/providers`.

**Capabilities must not over-claim.** The pipeline routes on them: if
IndicTrans2 says it supports `ja→en`, Japanese audio will be sent to a model
that has never seen Japanese, and the failure will look like a quality problem
rather than a routing bug.

Licence fields are part of the contract, not documentation:

```python
TranslationCapabilities(
    model_license="CC-BY-NC-4.0",
    commercial_use=False,
)
```

`voicebridge providers` prints this, so an operator can see a non-commercial
restriction without reading any docs.

---

## `ASREngine`

```python
async def start_session(session_id, language=None, **options) -> None
async def push_audio(session_id, chunk: AudioChunk) -> None
def     get_events(session_id) -> AsyncIterator[Event]
async def stop_session(session_id) -> None
async def warmup() -> None
```

Emits `ASR_PARTIAL`, `ASR_STABLE`, `ASR_FINAL`, `LANGUAGE_DETECTED`,
`SPEAKER_CHANGED`, `ERROR`.

### Rules

1. **The backend owns the commitment policy.** Do not add a second one in the
   adapter; the stabiliser only de-duplicates.
2. **`ASR_STABLE` is a promise.** Do not emit text you may retract. If a backend
   does retract, the stabiliser reports `rolled_back` and refuses to re-emit.
3. **Snapshots may be cumulative.** De-duplicate on segment end time.
4. `emits_partial_and_stable=False` tells the pipeline the backend has no
   partial/stable distinction, so everything is final on arrival and the
   dubbing path cannot lag the subtitle path.
5. Never block the event loop — model calls go through `asyncio.to_thread`.

### Implementations

| Name | Notes |
| --- | --- |
| `mock` | Scripted, deterministic; exercises partial→stable transitions. |
| `whisperlivekit` | Real streaming ASR. Always `pcm_input=True` to avoid an FFmpeg dependency. Requires Python <3.14. |

---

## `TranslationEngine`

```python
async def translate(source_text, source_language, target_language,
                    context=None, metadata=None) -> TranslationResult
def supports(source_language, target_language) -> bool
```

### Rules

1. `supports()` must be honest. Prefer returning `False` over failing at
   generation time.
2. `context` is passed **only** if `capabilities.supports_context` is true.
3. Glossary handling is the pipeline's job, not the provider's: text arrives
   with sentinels already substituted and the provider must pass them through
   unchanged. This is why glossaries work with every provider, including remote
   ones.
4. Raise `ProviderError` for a recoverable failure; the pipeline degrades to
   source-language subtitles rather than killing the session.
5. Declare the **model** licence, not the code licence.

### Implementations

| Name | Pairs | Model licence | Commercial |
| --- | --- | --- | --- |
| `mock` | any | n/a | ✅ |
| `opus_mt` | ja→en, ko→en verified; others resolved at load | Apache-2.0 | ✅ |
| `indictrans2` | en ↔ 22 scheduled Indian languages | MIT | ✅ |
| `nllb` | 200 languages | CC-BY-NC-4.0 | ❌ gated |

`opus_mt` publishes one bilingual checkpoint per direction and **not every
direction exists**. The provider resolves the checkpoint at load time and raises
a clear error naming the missing model rather than inferring support from the
naming scheme.

`nllb` refuses to load unless the operator sets
`acknowledge_non_commercial: true` or `VOICEBRIDGE_ALLOW_NC_MODELS=1`. That
makes the restriction a decision someone made, not one they inherited.

---

## `TTSEngine`

```python
async def synthesize(text, language, speaker=None, speed=1.0,
                     sequence_id=0) -> SynthesisedAudio
```

### Rules

1. **Return the sequence id you were given.** The scheduler's ordering depends
   on it entirely.
2. Report the **actual** sample rate of the voice, do not assume one. Piper
   voices vary (16 kHz, 22.05 kHz).
3. `duration` must match the audio; the backlog calculation uses it.
4. Empty text returns a valid zero/minimum-length result, never a crash.
5. Raise on failure — the pipeline calls `scheduler.mark_failed()` so later
   segments are not stuck behind the gap.

### Implementations

| Name | Notes |
| --- | --- |
| `mock` | Real audible PCM, plausible duration. Not speech. |
| `piper` | MIT engine. **Voices carry their own licences** and are neither shipped nor auto-downloaded. |

---

## Adding a provider

```python
from voicebridge.providers.registry import translation_registry

class MyEngine(TranslationEngine):
    name = "mine"
    @property
    def capabilities(self): ...
    async def translate(self, ...): ...

translation_registry.register("mine", MyEngine)
```

Register the **factory**, never an instance: importing the registry must not
import torch, or mock mode stops working on machines without ML dependencies.
Add the module to `load_builtin_providers()`; each import is individually
guarded, so one broken provider cannot take down the rest.

Missing dependencies must raise `ProviderUnavailable` at `create()`/first use
with an actionable message naming the install command — not at import time.
