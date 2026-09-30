#!/usr/bin/env bash
# Create the isolated worker virtualenvs used by model providers whose pinned
# dependencies conflict with the gateway environment.
#
#   scripts/setup_workers.sh qwen        # Qwen3-TTS + Qwen3-ASR  (Apache-2.0 models)
#   scripts/setup_workers.sh seamless    # Meta SeamlessStreaming (CC-BY-NC-4.0 models)
#
# Requires `uv` (https://docs.astral.sh/uv/). CPU wheels by default; set
# TORCH_INDEX=https://download.pytorch.org/whl/cu121 (qwen) or the matching
# fairseq2 CUDA index (seamless) for GPU builds.
# Model weights are NOT downloaded here.
set -euo pipefail
cd "$(dirname "$0")/.."
UV=${UV:-uv}
which_env=${1:?usage: setup_workers.sh qwen|seamless}

case "$which_env" in
  qwen)
    TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}
    $UV venv --python 3.12 .venv-workers/qwen
    $UV pip install --python .venv-workers/qwen/bin/python --index-url "$TORCH_INDEX" torch torchaudio
    # qwen-tts pins transformers==4.57.3 and qwen-asr pins 4.57.6; both run on 4.57.6.
    $UV pip install --python .venv-workers/qwen/bin/python transformers==4.57.6 accelerate==1.12.0 \
        librosa soundfile sox onnxruntime einops nagisa==0.2.11 soynlp==0.0.493 qwen-omni-utils numpy
    $UV pip install --python .venv-workers/qwen/bin/python --no-deps qwen-tts qwen-asr
    ;;
  seamless)
    # seamless_communication needs fairseq2 0.2.x, which ships wheels for
    # Python <= 3.11 and torch 2.1 only.
    FS2_INDEX=${FS2_INDEX:-https://fair.pkg.atmeta.com/fairseq2/whl/pt2.1.1/cpu}
    $UV venv --python 3.10 .venv-workers/seamless
    $UV pip install --python .venv-workers/seamless/bin/python \
        --index-url https://download.pytorch.org/whl/cpu torch==2.1.1 torchaudio==2.1.1
    $UV pip install --python .venv-workers/seamless/bin/python \
        --extra-index-url "$FS2_INDEX" --index-strategy unsafe-best-match \
        "fairseq2==0.2.1" "fairseq2n==0.2.1+cpu" "numpy<2" soundfile \
        "git+https://github.com/facebookresearch/seamless_communication.git"
    ;;
  *) echo "unknown worker env: $which_env" >&2; exit 2 ;;
esac
echo "worker env ready: .venv-workers/$which_env"
