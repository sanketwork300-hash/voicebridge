"""Speech gate decision logic and ASR hallucination validation."""

import asyncio

import numpy as np

from voicebridge.core.asr_validation import ASRValidationEngine
from voicebridge.core.pipeline.speech_gate import SpeechGate, SpeechGateConfig, merge_decisions
from voicebridge.core.types import AudioChunk, AudioEventType, SpeechDecision, TranscriptSegment

D = AudioEventType.DIALOGUE.value
M = AudioEventType.MUSIC.value
V = AudioEventType.VOCAL_NON_SPEECH.value


def scores(**kw):
    base = {e.value: 0.0 for e in AudioEventType}
    base.update({k.upper(): v for k, v in kw.items()})
    return base


def test_dialogue_over_music_is_transcribed():
    gate = SpeechGate(config=SpeechGateConfig())
    d = gate.combine(0, 2, 0.9, True, scores(dialogue=0.55, music=0.95))
    assert d.should_transcribe and d.event_type == D


def test_confident_music_is_rejected():
    gate = SpeechGate(config=SpeechGateConfig())
    d = gate.combine(0, 2, 0.9, True, scores(dialogue=0.05, music=0.92))
    assert not d.should_transcribe and d.event_type == M


def test_laughter_is_rejected():
    gate = SpeechGate(config=SpeechGateConfig())
    d = gate.combine(0, 2, 0.9, True, scores(dialogue=0.1, vocal_non_speech=0.8))
    assert not d.should_transcribe and d.event_type == V


def test_uncertain_audio_goes_to_asr_conservatively():
    gate = SpeechGate(config=SpeechGateConfig())
    d = gate.combine(0, 2, 0.6, True, scores(dialogue=0.2, sfx=0.4))
    assert d.should_transcribe and d.event_type == AudioEventType.UNKNOWN.value


def test_vad_no_speech_rejects_without_classifier():
    gate = SpeechGate(config=SpeechGateConfig())
    d = gate.combine(0, 2, 0.1, False, None)
    assert not d.should_transcribe


def test_merge_decisions_joins_adjacent_same_verdict():
    ds = [SpeechDecision(True, D, 0.8, 0, 2), SpeechDecision(True, D, 0.6, 2, 4),
          SpeechDecision(False, M, 0.9, 4, 6)]
    merged = merge_decisions(ds)
    assert [(m.start_time, m.end_time) for m in merged] == [(0, 4), (4, 6)]


class _FakeVAD:
    name = "fake"

    async def detect(self, chunk):
        loud = np.abs(np.frombuffer(chunk.data, "<i2")).mean() > 100
        return SpeechDecision(loud, D if loud else "SILENCE", 0.9, chunk.start, chunk.end)


def test_realtime_gate_replaces_rejected_audio_with_equal_length_silence():
    gate = SpeechGate(vad=_FakeVAD(), config=SpeechGateConfig(window_seconds=0.4))
    quiet = AudioChunk(data=np.zeros(3200, "<i2").tobytes(), start=0.0)
    loud = AudioChunk(data=(np.ones(3200) * 3000).astype("<i2").tobytes(), start=0.2)

    async def run():
        out = await gate.gate_chunk(quiet)
        assert out == []  # still buffering the window
        return await gate.gate_chunk(loud)

    out = asyncio.run(run())
    # Window average is loud -> forwarded unchanged, same total length.
    assert sum(len(c.data) for c, _ in out) == 12800
    assert all(d.should_transcribe for _, d in out)

    gate2 = SpeechGate(vad=_FakeVAD(), config=SpeechGateConfig(window_seconds=0.2))
    out2 = asyncio.run(gate2.gate_chunk(quiet))
    assert len(out2) == 1 and out2[0][0].data == bytes(6400)  # silence, same length
    assert not out2[0][1].should_transcribe


# ---------------------------------------------------------------- validation

def seg(text, start=0.0, end=3.0):
    return TranscriptSegment(text=text, start=start, end=end)


def test_validator_accepts_normal_dialogue():
    r = ASRValidationEngine().validate(seg("昨日のことだけど、彼には言わないで。"),
                                       {"no_speech_prob": 0.02, "avg_logprob": -0.2})
    assert r.valid and r.confidence > 0.7


def test_validator_rejects_stock_phrase_without_dialogue_evidence():
    # Measured: Whisper emits this on laughter/music with no_speech_prob 0.00.
    r = ASRValidationEngine().validate(seg("ご視聴ありがとうございました"),
                                       {"no_speech_prob": 0.0, "avg_logprob": -0.3})
    assert not r.valid and r.reason == "known_hallucination_phrase"


def test_validator_keeps_stock_phrase_with_dialogue_evidence():
    d = SpeechDecision(True, D, 0.9, 0, 3)
    r = ASRValidationEngine().validate(seg("ご視聴ありがとうございました"),
                                       {"no_speech_prob": 0.01, "avg_logprob": -0.2}, d)
    assert r.valid


def test_validator_rejects_cjk_loops_and_repetition():
    assert ASRValidationEngine().validate(seg("あ、あ、あ、あ、あ、あ、あ")).reason == "repetition"
    assert ASRValidationEngine().validate(seg("ははははははははは")).reason == "repetition"


def test_validator_rejects_decoder_signals_and_density():
    v = ASRValidationEngine()
    assert v.validate(seg("hello there"), {"no_speech_prob": 0.95}).reason == "high_no_speech_probability"
    assert v.validate(seg("hello there"), {"avg_logprob": -2.2}).reason == "low_average_log_probability"
    assert v.validate(seg("x" * 200, 0, 1)).reason == "abnormal_text_density"


def test_validator_rejects_language_mismatch_and_music_event():
    v = ASRValidationEngine()
    r = v.validate(seg("this is english"), {"language": "en", "language_probability": 0.97},
                   expected_language="ja")
    assert r.reason == "language_mismatch"
    r = v.validate(seg("歌詞のような文"), None, SpeechDecision(False, M, 0.9, 0, 3))
    assert r.reason == "non_dialogue_audio_event"


def test_segment_spanning_pauses_is_not_vetoed_by_one_silent_window():
    from voicebridge.core.pipeline.speech_gate import decision_for_span

    # 0.6 s gate windows: speech, pause, speech, speech; the segment spans all.
    ds = [SpeechDecision(True, D, 0.7, 0.0, 0.6), SpeechDecision(False, "SILENCE", 0.95, 0.6, 1.2),
          SpeechDecision(True, D, 0.9, 1.2, 1.8), SpeechDecision(True, D, 0.8, 1.8, 2.4)]
    d = decision_for_span(ds, 0.0, 2.4)
    assert d.should_transcribe and d.confidence == 0.9
    # A segment that lies wholly in rejected audio keeps the rejection.
    music = [SpeechDecision(False, M, 0.9, 10.0, 10.6), SpeechDecision(False, M, 0.9, 10.6, 11.2),
             SpeechDecision(False, M, 0.9, 11.2, 11.8), SpeechDecision(False, M, 0.9, 11.8, 12.4)]
    assert decision_for_span(ds + music, 10.6, 12.0).event_type == M
    assert decision_for_span([], 0, 1) is None
