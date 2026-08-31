# VoiceBridge documentation

Start with the [project README](../README.md) for installation and a 60-second
demo.

## Understanding the system

| Document | What it covers |
| --- | --- |
| [architecture.md](architecture.md) | Stage graph, the three invariants, and why Japanese/Korean drive the design |
| [protocol.md](protocol.md) | WebSocket message specification |
| [provider-contracts.md](provider-contracts.md) | The three provider ABCs and the rules implementations must follow |
| [reference-analysis.md](reference-analysis.md) | What was read in each reference repository, which APIs were verified, and what no reference provided |

## Running it

| Document | What it covers |
| --- | --- |
| [deployment.md](deployment.md) | Local, Docker, remote with TLS, scaling, operations |
| [performance.md](performance.md) | Benchmark methodology, tuning, and the human-evaluation protocol |
| [troubleshooting.md](troubleshooting.md) | Symptoms and fixes, extension through latency |

## Obligations

| Document | What it covers |
| --- | --- |
| [security.md](security.md) | Threat model and the checklist for network deployments |
| [privacy.md](privacy.md) | What exists in memory, for how long, and what recording implies |
| [license-matrix.md](license-matrix.md) | Every dependency, reference and model — **including the non-commercial ones** |

## The two things most likely to trip you up

1. **Model weights are often more restrictive than the code that loads them.**
   NLLB-200 is CC-BY-NC-4.0; Piper voices vary per voice. See
   [license-matrix.md](license-matrix.md); `voicebridge providers` reports this
   at runtime.
2. **Japanese and Korean must not be translated before the sentence ends.**
   Tense, negation and mood are sentence-final, so an early translation can
   assert the opposite of what was said. See
   [architecture.md](architecture.md#language-aware-segmentation).

## No published numbers

This project publishes no latency or quality benchmarks yet. The harness exists
(`benchmarks/run_benchmark.py`); it has not been run on a stated evaluation set
and reference hardware. See [performance.md](performance.md).
