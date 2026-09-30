# Architecture

## The problem this shape solves

Real-time speech translation has one structural difficulty that dominates every
design decision: **you must decide when to act on text that is not final yet.**

A streaming recogniser continuously revises its hypothesis. Act too early and
you translate a fragment whose meaning has not been established; act too late
and the output is no longer real-time. Every stage below exists to place that
decision explicitly rather than let it happen by accident.

The second structural difficulty is that the answer is **language-dependent**.
In English, a clause usually means roughly what it will mean once finished. In
Japanese and Korean it frequently does not, because tense, negation, politeness
and question/statement mood sit at the end. This is why VoiceBridge does not
have one global latency knob.

---

## Stage graph

```
input adapter
     │  AudioChunk (PCM s16le / 16 kHz / mono, on a session sample clock)
     ▼
[audio_queue]            bounded · DROP_OLDEST (live) or BLOCK (file)
     ▼
ASR provider             owns the commitment policy
     │  ASR_PARTIAL (retractable)   ASR_STABLE (committed)
     ▼
TranscriptStabilizer     de-duplicates cumulative snapshots
     ▼
TranslationSegmenter     language-aware; decides what a "unit" is
     ▼
[translation_queue]      bounded · DROP_NEWEST_DISPOSABLE
     ▼
TranslationEngine        + glossary protection, context window, honorifics
     │  TRANSLATION_FINAL  ──────────────────────────────► subtitles
     ▼
[tts_queue]              bounded · BLOCK (committed speech is never dropped)
     ▼
TTSEngine → TTSScheduler reorder buffer, strict sequence, backlog accounting
     ▼
TTS_AUDIO ──────────────────────────────────────────────► dubbed audio
```

---

## Invariant 1 — partial text never becomes speech

Subtitles and dubbing run at different commitment levels, deliberately:

| Path | Consumes | Why |
| --- | --- | --- |
| Subtitles | partial + committed | A retracted subtitle is a flicker the viewer forgives. |
| Speech | committed only | Audio cannot be un-played. A spoken sentence that turns out to be wrong is unrecoverable. |

So the dub always trails the subtitles by one commitment stage, and
`subtitle_latency` and `speech_latency` are reported separately rather than
averaged into one misleading number. The e2e suite asserts this directly:
everything spoken is a subset of what was committed.

## Invariant 2 — no unbounded queues

A pipeline that buffers without limit does not fail. It drifts further behind
while continuing to produce plausible output, which is worse, because nobody can
tell it is happening. Every queue is bounded and declares what it sacrifices:

| Queue | Policy | Rationale |
| --- | --- | --- |
| audio (live) | `DROP_OLDEST` | Freshest audio is the useful audio; a live source cannot be slowed. |
| audio (file) | `BLOCK` | A file *can* be slowed. Dropping would silently produce an incomplete transcript. |
| translation | `DROP_NEWEST_DISPOSABLE` | Partial hypotheses are expendable; committed segments never are. |
| TTS | `BLOCK` | Ordering and completeness are mandatory. |

Overload is surfaced as a throttled `WARNING` event, not hidden.

## Invariant 3 — every stage degrades rather than fails

| Failure | Behaviour |
| --- | --- |
| TTS provider fails | Warning; segment skipped in the scheduler so ordering recovers; subtitles continue. |
| Translation provider fails | Warning; translation disabled for the session; source-language transcript continues. |
| ASR fails | Recoverable error event, or session-fatal error with a clear message. |
| Socket drops | Session survives; client reconnects with exponential backoff. |

---

## Transcript stabilisation

Streaming backends report a *snapshot*: committed text plus a retractable tail.
WhisperLiveKit exposes exactly this as `FrontData.lines` and
`FrontData.buffer_transcription`.

`TranscriptStabilizer` converts successive snapshots into a monotonic stream of
newly-stable spans, so each piece of source text is translated exactly once. It
deliberately does **not** implement its own commitment policy: the backend
already has one, and layering a second produces double-buffering and latency
nobody can attribute.

It also handles script-aware joining. Japanese and Chinese are written without
inter-word spaces, so joining two segments with `" "` corrupts the text and
changes tokenisation downstream. Korean *does* use spaces, so Hangul is
excluded from that rule — a distinction worth spelling out because getting it
backwards silently degrades a Tier-1 language pair.

---

## Language-aware segmentation

The most important quality mechanism in the system.

Translating every stable fragment yields "I", "think", "the" as separate units:
wasted compute and bad output. The correct unit is a meaningful language unit,
and what counts as one is language-dependent.

`LanguageSegmentationProfile` (`core/segmentation/profiles.py`) declares, per
language: pause threshold, minimum length, maximum length, maximum duration,
whether punctuation flushes, and whether the language needs **sentence-final
bias**.

Flush triggers, in priority order:

1. `max_chars` / `max_seconds` — hard safety valves; always fire, so a speaker
   who never pauses still produces output.
2. Sentence-terminating punctuation.
3. A language-specific sentence-final ending (`ja`: ます/ました/です…, `ko`: 습니다/어요/죠…).
4. A pause at least `pause_threshold_ms` long.
5. A soft break, only when the buffer is already substantial.

Below `minimum_stable_chars`, rules 2–5 cannot fire. Under `sentence_final_bias`
nothing weaker than rule 1–3 may fire at all: a clause without its ending is
precisely what must not be translated.

**`low_latency` never disables sentence-final bias.** It scales the waiting
parameters, not the correctness rule. Buying latency by translating incomplete
Japanese is not a trade-off, it is a bug — and it is unit-tested as such.

> The thresholds shipped today are reasoned defaults, **not measured optima.**
> They are configuration precisely because they need benchmarking.

---

## Context, glossary and honorifics

**Context** (`ContextWindow`) is bounded on both segment count and character
count. Japanese and Korean omit subjects freely, so prior sentences carry real
information — but an unbounded window is a latency bug, since every extra token
is encoder work on the hot path. Previous translations steer terminology
consistency only; they are never authoritative output.

**Glossary** uses placeholder protection: each term is replaced with an opaque
sentinel before translation and restored after. This works with any provider,
including remote ones, because it needs no model support. If a model mangles a
sentinel, the restorer strips it rather than showing `VBX3X` to a viewer.
Longest terms match first, so 五条悟 wins over 五条.

**Honorifics** are a documented post-translation heuristic, not a model
capability. With exactly one honorific in the source it rewrites "Mr. Tanaka"
to "Tanaka-san". With two it does nothing, because attributing titles to names
without word alignment would be guessing, and a wrong honorific attached to the
wrong character is worse than an unconverted title. A glossary entry always
beats the heuristic.

---

## TTS scheduling

Synthesis runs concurrently and finishes out of order, so a reorder buffer keyed
by sequence id is mandatory — segment 5 must never be heard before segment 4.

Two failure modes get explicit handling:

* **A failed segment** would otherwise deadlock every later segment behind a
  sequence number that never arrives. `mark_failed` closes the gap; a
  `gap_timeout` closes it as a backstop when a provider neither returns nor
  raises.
* **Backlog.** If TTS is slower than real time the dub drifts steadily behind
  the video while still sounding fine. The scheduler measures backlog
  continuously and reports overload so the pipeline can degrade deliberately
  rather than drift.

---

## Threading and concurrency

The pipeline is a set of `asyncio` tasks in one process. Model inference is
CPU/GPU-bound and blocking, so every provider runs it via `asyncio.to_thread`,
and each loaded model is guarded by a lock because HF models are not safe to
call re-entrantly from several threads.

Providers are process-wide, not per-session: a 600 MB model loaded per session
is both slow and a quick route to OOM.

## Why one process

Splitting ASR, translation and TTS into separate services adds a network hop to
every stage of a latency-critical path. For the deployment sizes this project
targets, that cost buys nothing. `docs/deployment.md` describes when the split
becomes worthwhile and how to do it — independent scaling, not tidiness, is the
trigger.

---

## v0.2 additions

### Speech gate (`core/pipeline/speech_gate.py`)

VAD (Silero) answers *is there voice?*; the AudioSet classifier (AST) answers
*what kind of sound?* and is multi-label. The combination rule, in order:

1. VAD says no voice → reject.
2. Dialogue score ≥ `dialogue_threshold` → transcribe, **even if music scores
   higher** (dialogue over a soundtrack is the normal case in drama/anime).
3. Some non-dialogue category ≥ `non_speech_threshold` → reject as that category.
4. Otherwise `UNKNOWN` → transcribe when `transcribe_unknown` (the ASR
   validator gets a second look).

Realtime: decisions are made per `window_seconds` window, and rejected windows
are forwarded to the streaming ASR as **silence of equal length**. Dropping
them would shift WhisperLiveKit's sample-derived timestamps and starve its own
VAD of the silence it needs to close an utterance. File mode: Silero regions
over the whole file, classified in 2 s windows, merged.

### ASR validation (`core/asr_validation/engine.py`)

Measured on this project's non-speech clips: Whisper large-v3-turbo produced
text on 8/8 clips (laughter, crying, applause, singing, music, synthetic
explosion, silence), typically "ご視聴ありがとうございました", with
`no_speech_prob = 0.00`. Decoder signals alone therefore cannot catch these;
the validator combines decoder signals, text-shape checks (density, token
repetition, character and phrase loops) and the speech gate's verdict. A
stock phrase is accepted only with positive dialogue evidence from the gate.

### Two orchestrations, one set of stages

`TranslationPipeline` (realtime) and `FilePipeline` (files) share the gate,
the validators, the segmenter, `translate_step.translate_segment` and
`tts_step.synthesize_fitted`. Providers do not know which mode they run in; the
only mode-specific provider method is `ASREngine.transcribe_array` (offline
decoding with confidence signals), which the file pipeline prefers over the
streaming session API.

### Contextual translation

`GlobalTranslationContext` keeps recent (source, translation) pairs within a
segment count and an estimated token budget. Pairs that do not fit are folded
into a short "Earlier:" summary instead of being dropped. The contextual
provider receives the glossary *as prompt constraints* — the output is never
string-replaced — while segment NMT providers keep the original sentinel
masking. The honorific policy becomes a prompt instruction; for `preserve`, the
old unambiguous-title heuristic still runs afterwards as a safety net because
small local models often ignore the instruction.

### TTS timing

`synthesize_fitted` speeds up output that overruns its source line by more
than `duration_tolerance`, capped at `max_rate`: by regenerating at a higher
rate when the engine controls rate natively (file mode), else by
pitch-preserving time-stretch (FFmpeg `atempo`). The `TimelineRenderer` places
each clip at its source start; if the previous clip is still playing, the new
one waits (no overlapping voices) and the delay is recorded as drift.

### Engines

`TranslationEngineMode` = `cascade` | `seamless_streaming` | `seamless_m4t_v2`
| `benchmark`. S2ST engines produce the same `SpeechTranslationResult` as the
cascade, so rendering is shared. An engine chosen by the user is never replaced
by another on failure.

### Model workers

Qwen3-TTS/ASR and SeamlessStreaming run in separate virtualenvs behind a
JSON-lines protocol (`voicebridge/workers`). The worker moves the real stdout to
a private descriptor before importing model code, because libraries print to
stdout. One request at a time per worker; idle workers can be stopped
(`idle_unload_seconds`), which is the only reliable way to return a large
model's memory.
