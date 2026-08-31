# VoiceBridge — CPU image.
#
# Default build is mock mode: it starts on any machine with no models and no
# GPU, which is what makes `docker compose up` a working demo rather than a
# 20-minute download. Pass EXTRAS at build time to bake in real models.
#
#   docker build -f deploy/docker/Dockerfile .
#   docker build -f deploy/docker/Dockerfile --build-arg EXTRAS="[translation-local,tts-piper]" .

FROM python:3.12-slim AS base

# FFmpeg is only needed for containerised/compressed input. The browser
# extension sends raw PCM, so this stays a small convenience rather than a
# requirement -- see voicebridge/core/audio/normalizer.py.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface \
    VOICEBRIDGE_HOST=0.0.0.0 \
    VOICEBRIDGE_PORT=8000

WORKDIR /app

# Dependency layer first so source edits do not invalidate the install.
COPY pyproject.toml README.md LICENSE ./
COPY voicebridge ./voicebridge

ARG EXTRAS=""
RUN pip install --upgrade pip && pip install ".${EXTRAS}"

COPY config ./config

# Models live on a mounted volume, never in the image: they are large and
# separately licensed.
RUN mkdir -p /models && useradd --create-home --uid 10001 voicebridge \
 && chown -R voicebridge:voicebridge /models /app
USER voicebridge

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["voicebridge-gateway"]
