# Troubleshooting

## Extension

**"Chrome would not grant tab capture"**
Chrome grants `tabCapture` only when the extension is invoked *on the tab being
captured*. Click the VoiceBridge icon on that tab and press Start; do not start
it from a different tab.

**"This page cannot be captured"**
`chrome://` pages, the Web Store and other restricted pages are off-limits to
extensions. Use a normal http(s) page.

**Subtitles appear but the tab has gone silent**
Capturing a tab detaches its audio from the speakers. VoiceBridge reconnects it
through a passthrough gain node — if you hear nothing, check that **Original
audio** is not set to *Mute original*.

**No overlay, but the popup says it is connected**
The content script cannot be injected into restricted pages. Capture and
translation still work; only the on-page overlay is missing. Watch the popup
status instead.

**"Cannot reach the VoiceBridge gateway"**
Check it is running (`curl localhost:8000/health`) and that the URL in Options
matches. For a remote gateway you must grant the optional host permission and
use `https://` (browsers block `ws://` from `https://` pages).

**Extension reconnects in a loop**
Usually a rejected origin. Add your extension id to `VOICEBRIDGE_WS_ORIGINS`
(`chrome-extension://<id>`, from `chrome://extensions`), or check the token.

---

## Gateway

**`ProviderUnavailable: whisperlivekit is not installed`**
```bash
pip install 'voicebridge[asr-whisperlivekit]'
```
On Python 3.14+ this command **succeeds but installs nothing**: WhisperLiveKit
publishes `requires_python = ">=3.11,<3.14"`, so the extra's environment marker
excludes it. That is why the error persists after an apparently successful
install. Check with `python -c "import whisperlivekit"`; use Python 3.11–3.13
for real ASR. Core and mock mode work on 3.14.

Also check *which* interpreter runs the gateway. `voicebridge-gateway` on
`PATH` may belong to a different Python than the venv you installed into;
`head -1 $(which voicebridge-gateway)` shows its interpreter. Running
`python -m voicebridge.apps.gateway.main` from the venv removes the doubt.
A CPU-only install that avoids the multi-gigabyte CUDA wheels:

```bash
uv pip install --python .venv/bin/python \
  --index https://download.pytorch.org/whl/cpu \
  -e '.[asr-whisperlivekit,translation-local,tts-piper]'
```

**Live providers configured, but the first session shows nothing for a minute
and then the extension reconnects**
Model download and load used to happen inside the first session's WebSocket.
The gateway now warms every configured provider at startup and logs
`provider warmup finished in N s`; `/ready` answers only after that. Give
OPUS-MT the pairs you use (`preload_pairs: [[ja, en]]`) so its ~300 MB
checkpoints are fetched then too. If a Whisper checkpoint download is
interrupted, WhisperLiveKit reports `SHA256 checksum does not match` on the
next start: delete `~/.cache/whisper/<model>.pt` and restart.

**Subtitles lag further and further behind (CPU)**
WhisperLiveKit's default `simulstreaming` policy runs a PyTorch decoder as
well as the faster-whisper encoder; on CPU it fell 24–60 s behind with
`small`. Set `backend_policy: localagreement` and `model: base` for a
CPU-only box, or use a GPU. `/v1/sessions/{id}` will not show this lag — the
audio queue stays empty because WhisperLiveKit buffers internally — so watch
the gateway log's `lag=` lines.

**The demo page shows English sentences I never said**
That is mock mode (the page says so in a banner): the mock ASR ignores audio
and replays a fixed script. Nothing is recognised or translated until the
providers in `config/voicebridge.yaml` are switched to real ones. Check
`curl localhost:8000/health` — `"mock_mode": true` means exactly this.

**`NLLB-200 weights are licensed CC-BY-NC-4.0`**
Working as intended. Either acknowledge it
(`acknowledge_non_commercial: true`) or use `opus_mt` / `indictrans2`, which are
Apache-2.0 and MIT respectively.

**`could not load OPUS-MT checkpoint … for xx->yy`**
OPUS-MT does not publish a checkpoint for every direction. Pick another provider
for that pair — the error names the direction it tried.

**`IndicTrans2 does not support ja->en`**
Correct: IndicTrans2 covers English and the 22 scheduled Indian languages only.
Use `opus_mt` for Japanese and Korean.

**`FFmpeg was not found on PATH`**
Only needed for compressed or containerised input. The extension sends raw PCM
and does not need it. Install it, or feed WAV/PCM.

**`/ready` returns 503**
A configured provider cannot be built. The response body names it. Common
causes: a missing extra, a missing Piper voice file, or an unacknowledged
non-commercial licence.

---

## Quality

**Translations are fragmentary — half sentences**
The segmenter is flushing too early. Use `--profile accurate`, or raise
`minimum_stable_chars` / `pause_threshold_ms` for that language.

**Japanese or Korean output says the opposite of the speech**
The classic symptom of translating before the sentence ending, where tense and
negation live. Check the source language is actually set to `ja`/`ko` — the
sentence-final profile only applies when the language is known. With
`source_language: auto`, the profile is generic until detection stabilises.

**Character or artist names keep changing**
Add glossary terms. This is what they are for:
```json
{"glossary": {"terms": [{"source": "五条悟", "target": "Satoru Gojo"}]}}
```

**Honorifics are dropped**
Set `honorifics: preserve_honorifics`. Note it deliberately does nothing when a
segment contains two *different* honorifics — attributing them without word
alignment would be guessing. A glossary entry always wins.

**Subtitles flicker and rewrite constantly**
That is the partial hypothesis being revised, and it is expected in
`low_latency`. Switch to `balanced`, or style partials less prominently.

---

## Latency

**Dubbed audio drifts further behind**
TTS is slower than real time. `/metrics` shows `tts_backlog_seconds` rising and
you will see `tts_backlog` warnings. Use a faster voice, shrink segments, or
switch to subtitle-only mode. The system reports this rather than drifting
silently — that is the design.

**`audio_backlog` warnings on a live source**
ASR cannot keep up; audio is being dropped to stay current. Use a smaller model
or a GPU. Check RTF with the benchmark harness — above 1.0 the model is simply
too big for the machine.

**First subtitle takes 30+ seconds**
Almost always first-run model download, not latency. Watch the logs; subsequent
sessions use the cache. Call `warmup()` (or hit the gateway once) before you
need it.

---

## Diagnostics

```bash
voicebridge providers                     # what is available, and its licence
curl localhost:8000/ready                 # why a provider failed
curl localhost:8000/v1/sessions           # live queue depths and latency
curl localhost:8000/metrics               # Prometheus metrics
voicebridge translate clip.wav --json     # raw event stream for one file
voicebridge -v serve                      # debug logging
```

`GET /v1/sessions/{id}` reports every queue's depth, capacity, policy and drop
count — the fastest way to find which stage is the bottleneck.

## File translation and model workers (v0.2)

**`FFMPEG_MISSING`.** File jobs need `ffmpeg` and `ffprobe`. Install them, or set
`media.ffmpeg_path` / `media.ffprobe_path` (or `VOICEBRIDGE_FFMPEG` /
`VOICEBRIDGE_FFPROBE`). Live translation does not need FFmpeg.

**`CORRUPTED_MEDIA` / `NO_AUDIO_STREAM`.** ffprobe could not read the container,
or it has no audio track. Uploads are also rejected up front (`415`) when the
content signature does not match the extension.

**Jobs crawl and the machine swaps.** Several large models are resident at once.
On a 14 GB machine, Qwen3-TTS in fp32 alone was 6.5 GB; in bf16 it is about 4.8 GB.
Use `runtime.low_memory: true` (the translation LLM is freed before TTS),
`providers.tts.idle_unload_seconds`, `runtime.warmup: [asr]` (lazy loading), the
`fast` translation preset, or Piper TTS.

**`qwen3-tts worker failed to start`.** Run `scripts/setup_workers.sh qwen`, or
point `providers.tts.python` at an interpreter that has `qwen-tts` installed.
Worker stderr is logged at DEBUG level under `voicebridge.workers.client`.

**`Qwen3-TTS Base needs a reference voice`.** The Base checkpoint only clones.
Use voice `source` (the original speaker's audio is used as the reference) or
configure `providers.tts.voices.<name>.ref_audio`.

**`SEAMLESS_MODEL_ERROR`.** The Seamless engine was selected and failed; the
message says why (disabled, missing weights, unsupported language, worker
crash). VoiceBridge never retries with the cascade on its own; choose the
cascade explicitly if you want it.

**Subtitles contain "ご視聴ありがとうございました" / "Thanks for watching" over music.**
That is a Whisper hallucination. Make sure `speech_gate.enabled: true`; the ASR
validator rejects these stock phrases unless the gate saw dialogue there.

**A repeated phrase appears in the transcript ("the X is the X, and the X is...").**
Whisper decoding loops, which were observed when the previous chunk's transcript was
passed as the prompt. Keep `providers.asr.prompt_previous_text: false` (the
default); the validator rejects phrase loops.
