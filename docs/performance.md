# Performance and benchmarking

## The honest position

**VoiceBridge publishes no latency numbers yet.**

The design targets P50 < 1.0 s and P95 < 2.0 s end-to-end for supported hardware
and language pairs. That is a *goal*, not a measurement. It has not been
demonstrated on reference hardware with a real evaluation set, so it is not
claimed anywhere in this repository, and it should not be quoted as if it were.

`benchmarks/run_benchmark.py` exists so the numbers can be produced and
reproduced. Until it has been run on a stated machine with a stated corpus,
there is nothing to report.

## What to measure

| Metric | Meaning |
| --- | --- |
| `first_subtitle_latency` | Audio in → first provisional text. What "is it working?" feels like. |
| `committed_subtitle_latency` | Audio in → committed translation. What the viewer reads. |
| `first_audio_latency` | Audio in → first dubbed speech. Necessarily higher; speech waits for commitment. |
| `end_to_end_latency` | Per segment, full pipeline. |
| `rtf` | Processing time ÷ audio duration. **> 1.0 means the system cannot keep up and will drift.** |

Subtitle and speech latency are reported **separately, never averaged**. They
are different products of the same pipeline at different commitment levels, and
one number hides that.

### Audio conditions

A system tuned only on clean microphone speech fails on the content people
actually watch. The harness separates:

`clean_speech` · `background_music` · `sound_effects` · `overlapping_speech` ·
`rapid_dialogue`

Anime and livestream audio is dominated by the middle three.

### Language pairs

Minimum release matrix: **ja→en, ko→en, en→hi, en→ja, en→ko**.

For Japanese, report **CER** alongside or instead of WER: Japanese has no
inter-word spaces, so word-level error rates depend on an arbitrary tokenisation
and are not comparable across systems.

## Running it

```bash
# Plumbing smoke test — mock providers, synthetic audio.
python benchmarks/run_benchmark.py --synthetic --all-pairs

# A real measurement.
python benchmarks/run_benchmark.py \
  --audio corpus/ja_clean_01.wav --source ja --target en \
  --category clean_speech --profile balanced --realtime \
  --asr whisperlivekit --translation opus_mt --tts piper \
  --output benchmarks/results/ja-en-clean.json
```

Use `--realtime` for anything you intend to quote. Without it the harness pushes
audio as fast as the pipeline accepts, which measures throughput, not the
latency a viewer experiences.

Every report embeds its environment (CPU, GPU, torch version, timestamp). **A
latency figure without its hardware is meaningless** — do not publish one.

## Reading RTF

| RTF | Meaning |
| --- | --- |
| < 0.5 | Comfortable headroom; a larger model is worth trying |
| 0.5–0.9 | Working, little headroom; bursts will cause backlog |
| ≈ 1.0 | Marginal; drift under any load spike |
| > 1.0 | Cannot keep up. Latency grows without bound until queues drop. |

Backlog is visible at runtime too: watch `tts_backlog_seconds` in `/metrics` and
the throttled `WARNING` events.

## Tuning, in the order worth trying

1. **Smaller ASR model.** Usually dominant. `large-v3-turbo` → `small` → `base`.
2. **Latency profile.** `low_latency` shortens waits — but never disables
   sentence-final bias on ja/ko, by design.
3. **Segmentation thresholds.** `core/segmentation/profiles.py`. Shipped values
   are reasoned defaults, not measured optima; this is exactly what the harness
   is for.
4. **Fewer beams.** `num_beams: 1` is materially faster than 5, with a quality
   cost worth measuring rather than assuming.
5. **Subtitle-only mode.** Removes the TTS stage entirely.
6. **GPU.** The largest single step for real models.

## Hardware profiles

Guidance, **not measurements** — do not treat as a promise:

| Profile | Target | Suggested starting point |
| --- | --- | --- |
| LOW | CPU only, ≤ 8 GB RAM | ASR `base`/`small`, OPUS-MT, Piper, subtitles-only |
| BALANCED | Mid-range GPU (~8 GB VRAM) | ASR `small`/`medium`, OPUS-MT or IndicTrans2 distilled, Piper |
| QUALITY | ≥ 16 GB VRAM | ASR `large-v3-turbo`, IndicTrans2 1B, higher-quality TTS |

Measure before claiming a model runs on a given GPU.

---

# Human evaluation

Automated metrics do not capture what makes entertainment subtitles good.
BLEU and COMET are insensitive to exactly the failures viewers notice most:
a character's name rendered three different ways, a lost honorific, a subtitle
that is correct but arrives after the scene has moved on.

Score each clip 1–5:

| Dimension | 1 | 5 |
| --- | --- | --- |
| Meaning preservation | Meaning lost or inverted | Fully preserved |
| Fluency | Unreadable | Natural target language |
| Proper-name handling | Names wrong or inconsistent | Correct and consistent |
| Context correctness | Omitted subjects wrongly inferred | Correct throughout |
| Subtitle timing | Badly out of sync | Well aligned |
| Dub naturalness | Robotic, overlapping | Natural, well paced |

For anime and drama, additionally score:

* **Honorific preservation** — kept where the preset asks
* **Terminology consistency** — same term rendered the same way throughout
* **Pronoun inference** — omitted subjects resolved correctly from context
* **Character-name accuracy** — names match the glossary

### Protocol

1. At least 3 evaluators per clip; report inter-rater agreement.
2. At least 10 clips per language pair, spread across the audio categories.
3. Blind to configuration — evaluators must not know which profile produced
   which output.
4. Include a human-translated reference for calibration.
5. Publish the corpus description and evaluator instructions with the scores.

**Never use clips you cannot legally redistribute.** Do not commit copyrighted
anime, drama or music audio to this repository.
