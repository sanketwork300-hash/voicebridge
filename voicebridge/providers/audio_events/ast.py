"""Audio event classification with the Audio Spectrogram Transformer (AST).

Model: ``MIT/ast-finetuned-audioset-10-10-0.4593`` (BSD-3-Clause), a 527-label
AudioSet tagger loaded through ``transformers`` (``ASTFeatureExtractor`` +
``ASTForAudioClassification``). AudioSet is multi-label, so every label gets an
independent sigmoid score; "Speech" and "Music" are routinely both high in
anime/drama where dialogue sits on a soundtrack.

That property drives the output format. A single argmax label would discard
dialogue whenever the music bed is louder, so :meth:`classify` returns the
*top* category as ``event_type`` and additionally reports every category score
in ``SpeechDecision.scores``. The speech gate then decides on the dialogue score
first, and only vetoes audio when a non-speech category is confident *and*
dialogue is weak.

The mapping from the AudioSet ontology to VoiceBridge's categories is by label
index and was built by reading this checkpoint's ``config.json`` ``id2label``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

import numpy as np

from voicebridge.core.types import AudioChunk, AudioEventType, SpeechDecision
from voicebridge.providers.base import (
    AudioEventCapabilities,
    AudioEventClassifier,
    ProviderUnavailable,
)
from voicebridge.providers.registry import audio_event_registry
from voicebridge.runtime import resolve_device

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"


def _r(a: int, b: int) -> set[int]:
    return set(range(a, b + 1))


#: AudioSet label index -> VoiceBridge category. Unlisted indices are SFX
#: (tools, vehicles, impacts, animals, "Sound effect", ...).
CATEGORY_INDICES: dict[AudioEventType, set[int]] = {
    # Speech, Male/Female/Child speech, Conversation, Narration, Speech
    # synthesizer, Shout, Yell, Whispering -- shouted words are still words.
    AudioEventType.DIALOGUE: {0, 1, 2, 3, 4, 5, 7, 8, 11, 15},
    # Babbling, Bellow, Whoop, Battle cry, Children shouting, Screaming,
    # Laughter..Sigh, Humming..Sniff (groan, breathing, cough, sneeze, ...).
    AudioEventType.VOCAL_NON_SPEECH: {6, 9, 10, 12, 13, 14} | _r(16, 26) | _r(37, 50),
    # Singing family and the whole Music subtree.
    AudioEventType.MUSIC: _r(27, 36) | _r(137, 282),
    # Chatter, Crowd, Hubbub, Children playing; natural ambience; acoustic
    # environment and noise labels.
    AudioEventType.BACKGROUND: _r(68, 71) | _r(283, 299) | _r(506, 526),
    AudioEventType.SILENCE: {500},
}


class ASTAudioEventClassifier(AudioEventClassifier):
    name = "ast"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = "auto",
        silence_rms: float = 0.003,
        **_: object,
    ):
        self.model_name = model
        self.device_pref = device
        self.silence_rms = float(silence_rms)
        self._model: Any = None
        self._extractor: Any = None
        self._device: str = "cpu"
        self._index: dict[int, AudioEventType] = {}
        self._lock = threading.Lock()

    @property
    def capabilities(self) -> AudioEventCapabilities:
        return AudioEventCapabilities(
            name=self.name, event_types=[event.value for event in AudioEventType]
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import ASTFeatureExtractor, ASTForAudioClassification
            except ImportError as exc:
                raise ProviderUnavailable(
                    "The AST audio event classifier needs torch and transformers: "
                    "pip install 'voicebridge[speech-gate]'"
                ) from exc
            self._device = resolve_device(self.device_pref)
            logger.info("loading audio event classifier %s on %s", self.model_name, self._device)
            self._extractor = ASTFeatureExtractor.from_pretrained(self.model_name)
            model = ASTForAudioClassification.from_pretrained(self.model_name)
            model.eval().to(self._device)
            self._torch = torch
            for event, indices in CATEGORY_INDICES.items():
                for idx in indices:
                    self._index[idx] = event
            self._model = model

    async def warmup(self) -> None:
        await asyncio.to_thread(self._load)

    def score_batch(self, windows: list[np.ndarray]) -> list[dict[str, float]]:
        """Category scores for a batch of float32 16 kHz windows (<= ~10 s each)."""
        self._load()
        feats = self._extractor(
            [w.astype(np.float32, copy=False) for w in windows],
            sampling_rate=16000,
            return_tensors="pt",
        )
        with self._torch.inference_mode():
            logits = self._model(feats["input_values"].to(self._device)).logits
            probs = self._torch.sigmoid(logits).float().cpu().numpy()
        out = []
        for row, window in zip(probs, windows, strict=True):
            scores = {event.value: 0.0 for event in AudioEventType}
            for idx, p in enumerate(row):
                event = self._index.get(idx, AudioEventType.SFX)
                if p > scores[event.value]:
                    scores[event.value] = float(p)
            rms = float(np.sqrt(np.mean(np.square(window)))) if window.size else 0.0
            if rms < self.silence_rms:
                scores[AudioEventType.SILENCE.value] = max(
                    scores[AudioEventType.SILENCE.value], 0.99
                )
            scores[AudioEventType.UNKNOWN.value] = 0.0
            out.append(scores)
        return out

    def decision_from_scores(
        self, scores: dict[str, float], start: float, end: float
    ) -> SpeechDecision:
        top = max(scores, key=lambda k: scores[k])
        confidence = scores[top]
        if confidence < 0.10:
            top, confidence = AudioEventType.UNKNOWN.value, 1.0 - confidence
        return SpeechDecision(
            should_transcribe=top in (AudioEventType.DIALOGUE.value, AudioEventType.UNKNOWN.value),
            event_type=top,
            confidence=confidence,
            start_time=start,
            end_time=end,
            reason=f"ast top={top}:{confidence:.2f} speech={scores[AudioEventType.DIALOGUE.value]:.2f}",
            scores=scores,
        )

    async def classify(self, chunk: AudioChunk) -> SpeechDecision:
        audio = np.frombuffer(chunk.data, dtype="<i2").astype(np.float32) / 32768.0
        scores = (await asyncio.to_thread(self.score_batch, [audio]))[0]
        return self.decision_from_scores(scores, chunk.start, chunk.end)


audio_event_registry.register("ast", ASTAudioEventClassifier)
audio_event_registry.register("audioset", ASTAudioEventClassifier)
