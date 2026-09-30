# Evaluation: measured results

Everything on this page was **measured on one machine**, with the commands
shown, and is reproducible with the benchmark tooling. Nothing here is a
target or an extrapolation. Where a number is missing, it was not measured.

## Machine and software

| | |
| --- | --- |
| CPU | AMD Ryzen 7 250 (16 threads, AVX-512 incl. BF16) |
| RAM | 14 GB (≈ 7–8 GB free with a desktop session running) |
| GPU | none (all runs CPU) |
| OS | Ubuntu 26.04.1, Linux 7.0 |
| Python | 3.12.14 (gateway), 3.12 (Qwen worker), 3.10 (Seamless worker) |
| PyTorch | 2.14.0+cpu (gateway), CPU wheels in workers |
| transformers | 5.16.1 (gateway), 4.57.6 (Qwen worker) |
| faster-whisper / CTranslate2 | 1.2.1 / 4.8.2 |
| WhisperLiveKit | 0.2.26 |
| FFmpeg | 7.0.2 static |
| Code | commit `d2dc26c` + the v0.2 working tree (reports record `commit_dirty: true`) |

Configuration: [`config/examples/cascade-cpu.yaml`](../config/examples/cascade-cpu.yaml).

## Data

- **FLEURS dev** (Google, CC-BY-4.0): the first 20 sentence ids shared by the
  source and target language. FLEURS is n-way parallel, so the English (or Hindi)
  reading of the same sentence is a human reference translation. Read speech,
  not conversational dialogue. Built with `python -m voicebridge.benchmark.datasets`.
- **Non-speech clips**: laughter (CC-BY-3.0), newborn crying (CC-BY-SA-4.0),
  applause (CC0), female vocalise (CC0), choir music (public domain), all from
  Wikimedia Commons via `scripts/fetch_nonspeech.py`, plus synthetic music, a
  synthetic explosion-like burst, and 5 s of digital silence.
- **Episodes** (`scripts/make_test_media.py`): 4–5 FLEURS utterances separated by
  silences, with a 6 s synthetic music bed and a 2 s noise burst inserted. Used
  as audio (WAV/MP3) and as 1280×720 H.264 video (MP4, and MKV for Korean).

## 1. ASR and the hallucination problem

`python -m voicebridge.benchmark --dataset benchmark/datasets/ja_mixed.json --components asr --asr whisperlivekit [--gate] --ablate-validation`

20 Japanese FLEURS utterances + the 8 non-speech clips. Language forced to `ja`.
"Hallucination rate" is the share of non-speech clips for which text survived.

| Configuration | CER (speech) | Hallucination rate (8 non-speech clips) | RTF | Peak RSS |
| --- | --- | --- | --- | --- |
| Whisper large-v3-turbo alone (no gate, no validator) | 2.44 % | **100 %** (8/8) | 0.70 | 1.5 GB |
| + ASR validator | 2.44 % | 0 % (0/8) † | 0.71 | 1.5 GB |
| + speech gate (Silero + AST), validator off | 3.00 % | 0 % (0/8) | 0.78 | 2.3 GB |
| + speech gate + validator (production path) | 3.00 % | 0 % (0/8) | 0.78 | 2.2 GB |

What Whisper produced on non-speech without either layer (verbatim):
laughter → 「ありがとうございました」, crying → 「あ、あ、あ、あ、あ、あ、あ」,
applause → 「ありがとうございました」, vocalise / choir / synthetic music /
synthetic burst / **silence** → 「ご視聴ありがとうございました」 ("thanks for
watching"), each with `no_speech_prob = 0.00` and `avg_logprob` ≈ −0.3. So
Whisper's own confidence signals did not flag any of them.

† **Caveat:** 「ありがとうございました」 was added to the validator's stock-phrase
list *after* seeing it in this run, so the validator-only 0 % is partly fit to
this set. The speech gate result is not: Silero sent none of the 8 clips to
ASR, and the gate's thresholds were not tuned on these clips.

Cost of the gate: CER rose from 2.44 % to 3.00 % (region edges trim a little
speech), and processing slowed (AST classification on CPU).

### Korean: a held-out check of the validator

Same experiment with 20 Korean FLEURS utterances + the same 8 clips, run
*before* any Korean-specific tuning (`benchmark/results/asr_ko_nogate`):

| Configuration | CER (speech) | Hallucination rate |
| --- | --- | --- |
| Whisper alone | 8.33 % | **100 %** (8/8) |
| + ASR validator (as it was) | 8.33 % | **62.5 %** (5/8 got through) |

Got through: 「감사합니다.」 (laughter), 「아멘」 (vocalise), 「한글자막 by 한효정」
(choir: a subtitle credit), 「다음 영상에서 만나요.」 (burst and silence). So the
validator alone **does not generalise**: it is a phrase list plus shape checks,
and new languages bring new stock phrases. Those phrases and a subtitle-credit
pattern were then added; the re-run is below. The speech gate is the
layer that does the real work, and the validator is a backstop.

Re-run on the 8 Korean-forced clips after the additions
(`asr_ko_nonspeech_v2`, `asr_ko_nonspeech_gate`):

| Configuration | Hallucination rate |
| --- | --- |
| Whisper alone | 100 % (8/8) |
| + validator (phrases added after seeing them; fitted to this set) | 0 % (0/8) |
| + speech gate, validator off (nothing tuned on these clips) | 0 % (0/8) |

### Whisper large-v3-turbo vs Qwen3-ASR-1.7B (Japanese, 20 FLEURS utterances)

`--components asr --asr whisperlivekit,qwen3_asr` (`benchmark/results/asr_ja_whisper_vs_qwen`)

| Model | CER | P50 latency | P95 latency | RTF | Load | Peak RSS |
| --- | --- | --- | --- | --- | --- | --- |
| Whisper large-v3-turbo (faster-whisper, int8) | 2.44 % | 5.79 s | 6.32 s | 0.44 | 15.1 s | 1.5 GB |
| Qwen3-ASR-1.7B (transformers, bf16) | 4.49 % | 8.59 s | 15.35 s | 0.59 | 20.3 s | 6.1 GB |

On this set and hardware Whisper had the lower CER, lower latency and a quarter
of the memory. Individual sentences went both ways (Qwen3-ASR got 国々 and 暴動
right where Whisper wrote クリグリ and ボード). Twenty read-speech sentences on a
CPU are not enough to generalise; noisy or overlapping dialogue was not tested.
Whisper stays the default.

## 2. End-to-end file jobs (cascade, real models)

All submitted through the HTTP API of a running gateway
(`POST /api/v1/files/translate`), exactly as the web UI does.

| Input | Direction | Outputs | Processing | RTF |
| --- | --- | --- | --- | --- |
| `ja_episode.mp3` (80.6 s) | ja → en | `.en.wav`, `.en.mp3`, `.en.srt`, `.en.vtt` | 305.7 s | 3.80 |
| `ja_episode.mp4` (80.6 s, 1280×720@25) | ja → en | `.en.mp4` (dub + original as 2nd track + subtitles), `.en.subtitled.mp4`, `.en.srt`, `.en.vtt`, `.en.wav` | 361.7 s | 4.49 |
| `ko_episode.mkv` (70.8 s), source `auto` | ko → en | `.en.mkv` (mix mode), `.en.srt`, `.en.vtt`, `.en.wav` | 321.5 s | 4.54 |
| `en_episode.wav` (50.4 s) | en → hi | `.hi.m4a`, `.hi.srt`, `.hi.vtt` | 158.0 s | 3.14 |

Stage times for the first job: speech detection 28.8 s, ASR 29.6 s, translation
(Qwen3-1.7B, bf16) 47.7 s, **TTS (Qwen3-TTS, bf16) 198.5 s**. TTS dominates on
CPU: Qwen3-TTS ran at RTF ≈ 5–6 (generating 1 s of speech took 5–6 s). The
English→Hindi job used Piper for Hindi (Qwen3-TTS has no Hindi) at TTS RTF 0.10, so its time went to translation (104.9 s, including loading the LLM on first use).

Checks on the outputs:

- **Timing.** The dubbed track's voiced spans started at 2.6, 19.3, 46.4, 54.9 and
  71.1 s, against source segments at 2.65, 18.87, 46.1, 54.9 and 71.1 s. Clips
  sit at their source times (drift 0.0 s), not concatenated.
- **Video.** H.264 stream copied (1280×720, 25 fps unchanged); duration
  80.576 s vs 80.56 s source; English dub default + original second track;
  `mov_text` English subtitles. MKV input → MKV output.
- **Music / noise.** The inserted 6 s music bed and 2 s noise burst produced no
  subtitles or speech in any job.
- **Intelligibility.** The dub, transcribed back by Whisper, reproduced the
  subtitle text. Whisper also hallucinated "The Seam." five times on the dub's
  trailing silence, which is the failure mode §1 measures.
- **Language auto-detection.** Korean was detected from `auto` and used from
  the first segment.

Observed translation errors (Qwen3-1.7B): 「企画後」 (an ASR error for 帰国後)
→ "after the planning phase"; 200만 년 → "200,000 years"; 가나 (Ghana) →
"Gaana"; "Javanese coconut sugar" → Hindi "जापानी कोको सुअर" ("Japanese cocoa
pig"). §3 measures translation quality systematically.

### Bugs found by these runs (fixed)

1. **Memory thrash.** With every model resident (Qwen3-TTS fp32 = 6.5 GB RSS),
   the 14 GB machine swapped and translation stalled. Fixes: bf16 workers on
   BF16-capable CPUs (Qwen3-TTS 4.8 GB), lazy loading (`runtime.warmup: [asr]`),
   `idle_unload_seconds`, and `runtime.low_memory`.
2. **Prompt-induced ASR loop.** Passing the previous chunk's transcript as
   Whisper's prompt produced "The UN is a production of the UN, and the UN is a
   production of the UN…" for an unrelated utterance, which the same model
   transcribes correctly without the prompt. The prompt is now off by default,
   and the validator rejects phrase loops.
3. `seamless_m4t_v2` was not a valid engine value (found by an integration test).

## 3. Translation quality: OPUS-MT vs contextual LLM

`--components translation --translation opus_mt,contextual:Qwen/Qwen3-1.7B,contextual:Qwen/Qwen3-4B-Instruct-2507`

Input is the FLEURS *reference* transcript (so MT is isolated from ASR errors);
the reference translation is the FLEURS English reading of the same sentence.
20 sentences per direction, single reference, sacrebleu defaults. COMET was not
computed (`unbabel-comet` needs transformers < 5, which conflicts with the
gateway environment).

| Direction | Model | BLEU | chrF | P50 latency | P95 latency | Peak RSS |
| --- | --- | --- | --- | --- | --- | --- |
| ja → en | OPUS-MT | 17.8 | 49.9 | 0.59 s | 1.42 s | 0.8 GB |
| ja → en | contextual, Qwen3-1.7B | 21.0 | 50.4 | 7.90 s | 13.60 s | 4.5 GB |
| ja → en | contextual, Qwen3-4B-Instruct | 25.8 | 56.3 | 17.39 s | 32.04 s | 9.4 GB |
| ko → en | OPUS-MT | 19.7 | 50.4 | 0.62 s | 1.83 s | 0.8 GB |
| ko → en | contextual, Qwen3-1.7B | 21.6 | 51.4 | 9.31 s | 17.81 s | 4.6 GB |
| ko → en | contextual, Qwen3-4B-Instruct | 24.0 | 56.6 | 19.67 s | 40.44 s | 10.2 GB |
| en → hi | OPUS-MT | 14.6 | 37.4 | 0.58 s | 3.32 s | 0.7 GB |
| en → hi | contextual, Qwen3-1.7B | **2.9** | **21.5** | 47.35 s | 62.57 s | 4.5 GB |
| en → hi | contextual, Qwen3-4B-Instruct | 17.2 | 42.7 | 85.32 s | 119.4 s | 9.9 GB |

Reading these honestly:

- The 1.7B model is **barely ahead of OPUS-MT on chrF** (+0.5 and +1.0) while
  being 13–15× slower on this CPU. With 20 sentences that gap is within noise.
  It is not evidence that the LLM path is better at this size.
- The 4B model is clearly ahead (+6 chrF, +4 to +8 BLEU) and 30× slower than
  OPUS-MT here; on CPU it is a file-mode option, not a live one.
- **For English → Hindi the 1.7B model is much worse than OPUS-MT** (chrF 21.5
  vs 37.4) and 80× slower. The en→hi file job in §2, which used it, shows the
  same thing qualitatively. The example configuration therefore routes Hindi to
  OPUS-MT (`translation_by_target`); the 4B model is better than both but took
  85 s per sentence on this CPU. IndicTrans2 was not benchmarked (not installed).
- FLEURS sentences are independent encyclopedic statements, so this set cannot
  show what context buys. §4 tests that separately.

## 4. Does conversational context help? (ablation)

`--dataset tests/fixtures/translation/{ja,ko}_dialogue.json --components translation --ablate-context`

Short consecutive dialogue lines written for this project (omitted subjects,
pronouns, honorifics, slang, idioms, negation). **Small sets (15 ja, 10 ko
lines) with the developers' own single reference translations**, so treat the
numbers as indicative. "No context" translates every line in isolation with the
same model and prompt.

| Set | Model | chrF with context | chrF without | BLEU with | BLEU without |
| --- | --- | --- | --- | --- | --- |
| ja (15) | OPUS-MT (context-free by design) | 56.9 | 56.9 | 38.5 | 38.5 |
| ja (15) | contextual, Qwen3-1.7B | 45.2 | 41.0 | 20.0 | 21.2 |
| ja (15) | contextual, Qwen3-4B-Instruct | 55.6 | 55.7 | 33.5 | 33.5 |
| ko (10) | OPUS-MT | 46.9 | 46.9 | 22.7 | 22.7 |
| ko (10) | contextual, Qwen3-1.7B | 56.2 | 45.7 | 43.7 | 29.6 |
| ko (10) | contextual, Qwen3-4B-Instruct | 60.1 | 56.5 | 42.4 | 40.3 |

(The "Peak RSS" column of the no-context rows in the raw report is inflated:
the earlier models were still loaded in the same process.)

What this shows and does not show:

- Context moved the scores in the expected direction for the 1.7B model on both
  sets and for the 4B model on Korean; for the 4B model on Japanese the score
  did not change, although individual lines did:

  | Source (after 「彼には言わないで。」「どうして？」) | With context | Without |
  | --- | --- | --- |
  | まだ怒ってるから。 (ref: Because he's still angry.) | 4B: "He's still angry." | 4B: "You're still mad at me." |
  | うん、でも信じてくれなかった。 (ref: Yeah, but he didn't believe me.) | 4B: "Yeah, but he didn't believe me." | 4B: "Yeah, but you didn't believe me." |
  | もう行っちゃったよ。 (ref: He's already gone.) | 4B: "He's already left." | 4B: "They've already left." |

- **On the Japanese set OPUS-MT scored higher than both LLMs.** It also guessed
  "he" correctly in several of these lines with no context at all. On the
  Korean set both LLMs were well ahead of OPUS-MT.
- The 1.7B model makes errors no context fixes ("Still angry.", "I've already
  left.") and **invented a name**: 五条先生 → "Watanabe-sensei" (4B: "Gokou-sensei";
  OPUS-MT: "Dr. Five"). A glossary entry is the reliable fix for names.
- Neither local model wrote the prototype example's "Don't tell him **about
  it**"; all produced "Don't tell him."

**Conclusion supported by the data:** the contextual provider works and context
measurably helps it, but at CPU-sized model scale it is *not* uniformly better
than OPUS-MT: ahead on Korean and on FLEURS with the 4B model, behind on the
Japanese dialogue set, and much worse for Hindi at 1.7B. A larger model or an
API backend may change this; that has not been measured here.

## 5. TTS: Piper vs Qwen3-TTS

`--components tts --tts piper,qwen3 --tts-items 5 --tts-reference benchmark/data/media/ref_voice_ja.wav`

Five English FLEURS reference sentences. "Round-trip" = the synthesised audio
transcribed back by Whisper large-v3-turbo and compared with the input text;
it measures intelligibility, **not** naturalness or voice similarity, which
were not measured (no MOS or speaker-similarity model was run).

| Model | RTF | P50 per sentence | Round-trip WER | Round-trip CER | Peak RSS |
| --- | --- | --- | --- | --- | --- |
| Piper `en_US-lessac-medium` | 0.08 | 0.32 s | 3.5 % | 0.9 % | 2.1 GB † |
| Qwen3-TTS-12Hz-1.7B-Base (bf16, voice cloned from an 8 s Japanese clip) | 5.44 | 44.98 s | 2.9 % | 0.2 % | 6.5 GB † |

† process tree including the Whisper model used for the round trip.

Both are intelligible. On this CPU Qwen3-TTS is about 68× slower than Piper and
far from real time; whether it *sounds* better has to be judged by listening
(the file-job outputs in `storage/outputs/` are there for that) or measured on
a GPU with a proper naturalness metric. Qwen3-TTS load time was 14.6 s.

## 6. Live (real-time) cascade

`scripts/stream_ws_client.py` does what the browser page does: creates a session,
opens the WebSocket, and sends 100 ms PCM frames **at wall-clock pace**, then
`AUDIO_STOP`. Input: `ja_episode.wav` (80.6 s, five utterances, a music bed and
a noise burst). Gateway config: [`config/examples/live-cpu.yaml`](../config/examples/live-cpu.yaml)
(Whisper **small**, Silero gate, contextual Qwen3-1.7B or OPUS-MT, Piper TTS).
These are observations from single runs on this CPU, not benchmarks.

| Translation | Time to first text | Time to first audio | Segment latency P50 | P95 | Segments |
| --- | --- | --- | --- | --- | --- |
| contextual, Qwen3-1.7B | 18.7 s | 20.0 s | 19.0 s | 21.5 s | 7 |
| `fast` preset (OPUS-MT), run 1 | 17.0 s | 17.4 s | 9.3 s | 15.7 s | 7 |
| `fast` preset (OPUS-MT), run 2 | 12.3 s | 12.5 s | 8.7 s | 12.2 s | 7 |

"Segment latency" = arrival time of a translated segment minus the source time
at which that segment's speech *ended*. Time to first text is large partly
because the first utterance is 11 s long and Japanese segments wait for a
sentence-final form.

**The design target (P50 < 1 s, P95 < 2 s) is not met on this CPU by a wide
margin** and nothing here suggests it would be without a GPU. All five
utterances were translated and spoken, nothing was produced for the music or
the noise burst, and the socket closed cleanly after the last segment.

What the live runs forced, in order:

1. **Whisper large-v3-turbo cannot stream on this CPU**: each 2.5 s decode took
   5.5 s and ASR was 47 s behind within a minute. The live CPU config uses
   `small` (lag 1–7 s), at a visible cost in accuracy: e.g. はぐれて → 羽ぐれて,
   歌手 → 菓子 ("famous candy"), 聖歌 → 成果. large-v3-turbo remains the
   default for files and for GPU.
2. **The AudioSet classifier is slower than real time on CPU** (0.7–0.85 s per
   0.6 s window), so live sessions on CPU gate with Silero VAD only
   (`realtime_event_classifier: auto`); files still use the classifier.
3. **Qwen3-TTS is not usable live on CPU** (RTF ≈ 5), and holding it together
   with the other models would need ≈ 12 GB; the live config uses Piper.
4. Bugs found and fixed: the socket was dropped at `AUDIO_STOP` before the
   last segments were translated (the pipeline now drains ASR → translation →
   TTS before closing); ASR validation ran on WhisperLiveKit's token-sized
   deltas and rejected real speech as "too dense/too short" (it now runs on
   the segmented unit); a segment spanning a pause could be vetoed by one
   silent gate window (now decided over the whole span).

**Not tested:** a physical microphone or browser tab (the client streams a
file through the same WebSocket protocol the page uses), overlapping speakers,
and real anime/drama audio.

## 7. Meta SeamlessStreaming on this machine

SeamlessStreaming runs through Meta's own streaming agent
(`SeamlessStreamingS2STJointVADAgent`) in the `seamless` worker. It **works**,
and this machine is **too small for it**:

| Run | Input | Result | Time | Worker memory |
| --- | --- | --- | --- | --- |
| Provider, direct | 8.5 s Japanese FLEURS utterance | "Argentina is known for having the best professional teams and professional athletes in the world." (ref: "…one of the best polo teams and players in the world"); speech output transcribed back identically | 153 s (RTF 18), model load 28.5 s | 10.5 GB RSS |
| File job via the gateway | `ja_short.mp3` (20.1 s: speech + 6 s music) | `ja_short.en.mp3`, `.en.srt`, `.en.vtt`; "In many cases, by enrolling in a gap year course abroad, / it is easier to go to university in the country."; no text for the music | 752 s (RTF 37) | ≈ 11 GB, heavy swapping |
| File job via the gateway (fresh worker, memory guard on) | `ja_short.mp4` (20.1 s, 1280×720) | `ja_short.en.mp4` (H.264 copied, translated AAC track, `mov_text` subtitles, 20.16 s), `.en.srt`, `.en.vtt`, `.en.wav`; same translation | 742 s (RTF 37) | released after the job |

The agent pipeline loads fp32 on CPU (its own rule), so the three checkpoints
(3.6 + 4.3 + 0.17 GB on disk) became 10–12.5 GB resident on a 14 GB machine
with a desktop session. The run was swap-bound and used about one core; these
timings say nothing about SeamlessStreaming on adequate hardware, and **no
latency comparison against the cascade is drawn from them**.

**Incident.** On a second consecutive file the worker reached 12.5 GB and the
kernel OOM killer killed it; systemd then stopped the whole terminal scope
(twice). Fixes that came out of it:

- `ModelWorker` memory guard: while a call is in flight it watches
  MemAvailable + SwapFree and stops the worker below a reserve (1.5 GB,
  `VOICEBRIDGE_WORKER_MEMORY_RESERVE_MB`), so the job fails with
  `SEAMLESS_MODEL_ERROR: … system memory nearly exhausted …` instead of an OOM kill.
- `restart_per_file: true` for the SeamlessStreaming provider (fresh worker per
  file; memory had grown from one file to the next).

**Not verified here:** real-time SeamlessStreaming (microphone → speech at
live pace). At RTF 18–37 it cannot keep up on this machine; the code path
(`S2STRealtimePipeline` → `translate_stream`) is covered by unit tests with a
stand-in engine only. It needs a machine with ≥ 16 GB free RAM or a GPU.

Timing of SeamlessStreaming output is the **emission offset** (first text for
the utterance that starts at 1.0 s was emitted after 4.8 s of input), not
source alignment; subtitles from this engine are labelled accordingly.

## 8. Cascade vs Meta Seamless on identical audio

One clip, `ja_short.wav` (20.1 s: one 11 s Japanese utterance, then a 6 s music
bed). Each engine ran in its own process through the same job executor and
output layer. **One sentence is an anecdote, not a benchmark**; it shows that
the comparison machinery works end to end and what each engine cost *on this
machine*.

Reference: "In many cases, enrolling on a gap-year course abroad can actually
improve your chances of moving into higher education back in your home country."

| Engine | Output text | chrF | BLEU | Wall time | RTF | Peak RSS | Output length |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VoiceBridge cascade (Whisper large-v3-turbo → Qwen3-1.7B → Qwen3-TTS) | "In many cases, entering an overseas gap year program can make it easier to pursue university after the planning phase." | 25.7 | 10.0 | 75.1 s | 3.74 | 6.5 GB | 20.1 s (ratio 1.00) |
| Meta SeamlessStreaming (streaming agent, fp32) | "In many cases, by enrolling in a gap year course abroad, it is easier to go to university in the country." | 41.0 | 13.3 | 752 s | 37.4 | ≈ 11 GB (swapping) | 20.2 s |
| Meta SeamlessM4T v2 large (offline, bf16, regions from the shared speech gate) | "In many cases, it's easier to actually go to college after you've graduated by enrolling in a gap year course abroad." | 40.5 | 12.5 | 33.8 s | 1.68 | 11.0 GB transient while loading, 3.8 GB mean | 20.1 s (ratio 1.00) |

Observations, without declaring a winner:

- On this sentence both Seamless models scored higher than the cascade. The
  cascade's error chain is visible: Whisper heard 帰国後 ("after returning
  home") as 企画後 ("after planning"), and the translation faithfully carried
  the error. A direct model has no such intermediate text to get wrong, and no
  intermediate text to inspect, constrain with a glossary, or validate either.
- None of the three produced any speech or text for the music bed.
- Time and memory here are dominated by this machine's limits (no GPU, 14 GB):
  Qwen3-TTS at RTF ≈ 5 for the cascade, swap-bound fp32 for SeamlessStreaming.
  They are not a statement about the engines on suitable hardware.
- Cascade segment timing is source-aligned; SeamlessM4T v2 segments carry
  their speech-gate region's source times; SeamlessStreaming reports emission
  offsets.
- Both Seamless models are CC-BY-NC-4.0.

Reproduce (one engine per process on a small machine):

```bash
python -m voicebridge.benchmark --dataset benchmark/datasets/ja_short.json \
    --config config/examples/cascade-cpu.yaml --components engines --engines cascade
python -m voicebridge.benchmark --dataset benchmark/datasets/ja_short.json \
    --config config/examples/seamless-cpu.yaml --components engines --engines seamless_m4t_v2
# or both in one report, plus live latency, where memory allows:
python -m voicebridge.benchmark.compare --dataset benchmark/datasets/ja_short.json --realtime
```

## What has not been measured

- Anything on a GPU; real-time latency on hardware that can meet the target.
- Real anime/drama/film audio, overlapping speakers, noisy dialogue: the
  speech data is FLEURS read speech plus synthetic/CC non-speech clips.
- Speech naturalness and voice similarity (no MOS / speaker-similarity model).
- COMET; any translation model larger than Qwen3-4B; API backends (OpenAI-
  compatible, Anthropic) and IndicTrans2.
- Live SeamlessStreaming; a physical microphone or the browser extension
  against the new pipeline stages.
- Files longer than 81 s (the streaming/chunked design is unit-tested, not
  load-tested on hour-long media).
