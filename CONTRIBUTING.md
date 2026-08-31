# Contributing to VoiceBridge

## Two rules that matter more than the rest

### 1. Verify before you implement

If you are integrating with something — a library, a model, an API — **read its
source or its model card first.** Not its README, not a blog post, not your
recollection.

This project has already been bitten by the difference:

* WhisperLiveKit pins `requires-python = ">=3.11, <3.14"`. Nothing in its README
  says so.
* NLLB-200's weights are CC-BY-NC-4.0 even though the surrounding code is
  permissive.
* IndicTrans2 needs `IndicProcessor.preprocess_batch` / `postprocess_batch`
  around every call; skipping it silently degrades output rather than failing.

When you add an integration, record what you verified in
`docs/reference-analysis.md`: the file you read, the API you used, the licence
you found. If a reference does **not** provide something we need, say so
explicitly and design the missing piece — do not imply it exists.

Never invent an API from a naming pattern. `Helsinki-NLP/opus-mt-{src}-{tgt}` is
a naming scheme, not a guarantee that a checkpoint exists; that is why the
provider resolves it at load time and fails with a clear message.

### 2. Never add an unbounded queue

`asyncio.Queue()` with no `maxsize` is a bug in this codebase. Use
`BoundedQueue` and choose an overflow policy deliberately:

| Policy | Use for |
| --- | --- |
| `DROP_OLDEST` | Live audio — freshest is most useful |
| `DROP_NEWEST_DISPOSABLE` | Partial results — expendable; committed ones are not |
| `BLOCK` | Committed work that must not be lost or reordered |

An unbounded queue does not fail loudly. It drifts further behind while
continuing to look fine, which is worse.

---

## Setup

```bash
git clone https://github.com/voicebridge/voicebridge
cd voicebridge
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest                          # 126 tests, ~8 s, no models required
ruff check voicebridge tests
```

The whole suite runs in mock mode. If a change makes CI need a GPU or a model
download, that change is wrong.

## Architecture rules

* **Dependency direction is inward.** Nothing in `core/` imports from
  `providers/`, `adapters/` or `apps/`. The pipeline talks to ABCs only.
* **Register factories, not instances.** Importing the provider registry must
  never import torch, or mock mode breaks on machines without ML dependencies.
* **Missing dependencies raise at use, not import.** `ProviderUnavailable` with
  an actionable message naming the install command.
* **Never block the event loop.** Model inference goes through
  `asyncio.to_thread`, guarded by a per-model lock.
* **Partial results never reach TTS.** Audio cannot be retracted. This is
  asserted in `tests/e2e/`.

## Adding a provider

See `docs/provider-contracts.md`. In short: implement the ABC, declare honest
capabilities (including `model_license` and `commercial_use`), register a
factory, add the module to `load_builtin_providers()`, add tests, and add a row
to `docs/license-matrix.md` and `THIRD_PARTY_LICENSES.md`.

If the model is non-commercial, gate it behind explicit acknowledgement the way
`nllb` is. Do not make a restricted model a default.

## Adding a language

1. Add a `LanguageSegmentationProfile` if the language needs different flush
   behaviour — and **explain the linguistic reason in the docstring.** The `ja`
   and `ko` profiles exist because those languages put tense, negation and mood
   sentence-finally; that reasoning is the useful part, not the numbers.
2. Add it to `apps/gateway/languages.py`.
3. Verify a provider actually supports the pair. Do not add a pair to the
   catalogue that no configured provider can serve.
4. Add tests covering the segmentation behaviour.

## Tests

Every change needs tests. Prefer tests that state a property rather than pin an
implementation detail:

```python
def test_low_latency_never_disables_sentence_final_bias():
    """Speed must not be bought by translating incomplete Japanese."""
```

Areas that must stay covered: audio normalisation, transcript stability,
segmentation, translation buffering, context bounding, TTS ordering,
backpressure, queue limits, provider failure, language switching, subtitle
timing, reconnection.

## Documentation

Explain *why*, not *what*. The code shows what it does; comments and docs should
capture the reasoning that is not recoverable from reading it — the constraint
discovered, the alternative rejected, the failure mode being prevented.

Do not document a feature that does not exist. If something is designed but
unimplemented, put it in the README's **Known limitations**.

## Performance claims

**Do not publish a latency number that is not reproducible with
`benchmarks/run_benchmark.py` on stated hardware.** No exceptions, including in
the README, release notes and issue comments. A figure without its environment
is not a measurement.

## Out of scope

VoiceBridge translates audio the browser or OS legitimately provides. Pull
requests that bypass DRM, authentication, paywalls or access controls will be
declined.

Do not commit model weights, or copyrighted anime, drama or music audio. Test
fixtures are generated at test time for exactly this reason.

## Commits and PRs

Explain the reasoning, not just the diff. If you changed a default, say what you
measured or what constraint forced it. If you fixed a bug, describe how it
manifested — future readers need the symptom, not only the patch.
