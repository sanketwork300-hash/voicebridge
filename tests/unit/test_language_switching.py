"""Language-detection stabilisation (Flow E)."""

from voicebridge.core.session.language import LanguageDetectionStabilizer


def test_first_detection_is_accepted_immediately():
    s = LanguageDetectionStabilizer()
    assert s.observe("ja", 0.95) == "ja"
    assert s.current == "ja"


def test_single_spurious_detection_does_not_switch():
    s = LanguageDetectionStabilizer()
    s.note_audio(10)
    s.observe("ja", 0.95)
    assert s.observe("en", 0.95) is None
    assert s.current == "ja"


def test_switch_requires_sustained_evidence():
    s = LanguageDetectionStabilizer(min_evidence=3)
    s.note_audio(10)
    s.observe("ja", 0.9)
    assert s.observe("en", 0.9) is None
    assert s.observe("en", 0.9) is None
    assert s.observe("en", 0.9) == "en"
    assert s.current == "en"


def test_evidence_streak_resets_on_contrary_evidence():
    s = LanguageDetectionStabilizer(min_evidence=3)
    s.note_audio(10)
    s.observe("ja", 0.9)
    s.observe("en", 0.9)
    s.observe("en", 0.9)
    s.observe("ja", 0.9)          # resets the streak
    assert s.observe("en", 0.9) is None
    assert s.current == "ja"


def test_low_confidence_detections_are_ignored():
    s = LanguageDetectionStabilizer(min_confidence=0.6)
    s.note_audio(10)
    s.observe("ja", 0.9)
    for _ in range(10):
        assert s.observe("ko", 0.2) is None
    assert s.current == "ja"


def test_switch_requires_minimum_audio():
    s = LanguageDetectionStabilizer(min_evidence=2, min_seconds=5.0)
    s.observe("ja", 0.9)
    s.note_audio(1.0)
    for _ in range(5):
        s.observe("en", 0.9)
    assert s.current == "ja"       # not enough audio yet
    s.note_audio(10.0)
    s.observe("en", 0.9)
    assert s.current == "en"


def test_auto_and_empty_are_ignored():
    s = LanguageDetectionStabilizer()
    assert s.observe("auto") is None
    assert s.observe(None) is None
    assert s.current is None
