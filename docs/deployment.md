# Deployment

## Choosing a mode

| Mode | Use when | Security burden |
| --- | --- | --- |
| **Local** — gateway on `127.0.0.1` | One person, one machine | Minimal; nothing leaves the box |
| **Remote** — network-reachable gateway | Shared GPU, several clients | Token, origin allow-list, TLS all required |

Start local. Move to remote only when you actually need a shared GPU.

---

## Local

```bash
pip install 'voicebridge[asr-whisperlivekit,translation-local,tts-piper]'
voicebridge-gateway
```

Models download on first use into `HF_HOME` (default `~/.cache/huggingface`).
Expect several GB and a slow first session. `voicebridge providers` shows what
resolved successfully.

## Docker

```bash
cd deploy/compose
docker compose --profile mock up      # no models, no GPU
docker compose --profile cpu  up      # real models on CPU
docker compose --profile gpu  up      # real models on CUDA
```

Models live on the `models` volume, never in the image — they are large and
separately licensed.

GPU requires the NVIDIA Container Toolkit on the host:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Remote, with TLS

Browsers refuse `ws://` from an `https://` page, so a network deployment needs
TLS regardless of your own preference.

```bash
export VOICEBRIDGE_DOMAIN=vb.example.com
export VOICEBRIDGE_API_TOKEN="$(openssl rand -hex 32)"
docker compose --profile gpu --profile tls up -d
```

`deploy/caddy/Caddyfile` obtains and renews certificates automatically. Note
`flush_interval -1`: without it a proxy's default write timeout cuts an open but
briefly idle translation socket.

Then in the extension's **Options**, set the gateway URL to
`https://vb.example.com` and paste the token.

### Required for any network deployment

```bash
VOICEBRIDGE_API_TOKEN=<random>                     # otherwise wide open
VOICEBRIDGE_WS_ORIGINS="chrome-extension://<id>"   # otherwise any page can connect
```

The gateway logs a warning at startup if you bind publicly without these. Find
your extension id at `chrome://extensions`.

---

## Scaling

The pipeline runs in one process deliberately: splitting stages adds a network
hop to each step of a latency-critical path.

**Scale out before you split up.** Run N gateway replicas behind a load
balancer with session affinity — a WebSocket session is stateful and must reach
the process that owns it. This is simpler and faster than a service split, and
it is the right answer until one of the following is true:

* ASR needs a GPU while translation and TTS do not, and you are paying for idle
  GPU;
* stages have genuinely different scaling curves under your traffic;
* you need to update one model without restarting sessions.

When splitting becomes justified, the provider interfaces are the seam: implement
`ASREngine`/`TranslationEngine`/`TTSEngine` as thin RPC clients. Nothing in
`core/` changes. Budget for the added hop — measure it against
`docs/performance.md` rather than assuming it is free.

### Capacity

Depends entirely on model choice and hardware; measure with
`benchmarks/run_benchmark.py` rather than extrapolating. Bound what a single
gateway will accept:

```yaml
security:
  max_sessions: 8
  max_session_seconds: 14400
  max_audio_frames_per_second: 100
```

Sessions past `max_session_seconds` are reaped by a background task, so an
abandoned tab cannot hold a GPU slot indefinitely.

---

## Operations

| Endpoint | Use |
| --- | --- |
| `/health` | Liveness — process is up |
| `/ready` | Readiness — configured providers can actually be built |
| `/metrics` | Prometheus text exposition |

Use `/ready`, not `/health`, as the load-balancer gate: a gateway whose model
failed to load is alive but useless.

Watch:

* `voicebridge_sessions_active`
* `voicebridge_end_to_end_latency_seconds`
* `voicebridge_dropped_partial_results` — rising means sustained overload
* `tts_backlog_seconds` — rising means the dub is drifting behind

The GPU image's healthcheck has a 180 s start period because model loading is
slow; a shorter one restarts the container mid-load, forever.

## Backups

There is nothing to back up. VoiceBridge holds no database and persists nothing
by default. Back up your config and your model volume.
