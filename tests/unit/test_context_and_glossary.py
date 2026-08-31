"""Context window bounding, glossary protection and honorific policy."""

import pytest

from voicebridge.core.context.glossary import SessionGlossary
from voicebridge.core.context.honorifics import apply_policy, detect_honorifics
from voicebridge.core.context.window import ContextEntry, ContextWindow
from voicebridge.core.types import HonorificPolicy, NameRendering


class TestContextWindow:
    def test_bounded_by_segment_count(self):
        w = ContextWindow(max_segments=3, max_characters=10_000)
        for i in range(10):
            w.add(ContextEntry(f"source {i}", f"target {i}"))
        assert len(w) == 3
        assert w.source_context() == ["source 7", "source 8", "source 9"]

    def test_bounded_by_character_count(self):
        w = ContextWindow(max_segments=100, max_characters=20)
        for _ in range(10):
            w.add(ContextEntry("x" * 10, "y" * 10))
        assert w.source_length <= 20

    def test_rejects_zero_size(self):
        with pytest.raises(ValueError):
            ContextWindow(max_segments=0)

    def test_never_grows_without_bound(self):
        w = ContextWindow(max_segments=4, max_characters=200)
        for i in range(1000):
            w.add(ContextEntry(f"segment number {i}", "t"))
        assert len(w) <= 4


class TestGlossary:
    @pytest.fixture
    def glossary(self):
        return SessionGlossary.from_payload({"terms": [
            {"source": "五条悟", "target": "Satoru Gojo", "romanized": "Satoru Gojo"},
            {"source": "領域展開", "target": "Domain Expansion"},
            {"source": "방탄소년단", "target": "BTS"},
        ]})

    def test_terms_survive_translation_as_sentinels(self, glossary):
        protected, mapping = glossary.protect("五条悟が領域展開を使った")
        assert "五条悟" not in protected
        assert len(mapping) == 2
        restored = glossary.restore(protected, mapping, NameRendering.TRANSLATE)
        assert "Satoru Gojo" in restored and "Domain Expansion" in restored

    def test_longest_term_wins(self):
        g = SessionGlossary.from_payload({"terms": [
            {"source": "五条", "target": "Gojo"},
            {"source": "五条悟", "target": "Satoru Gojo"},
        ]})
        protected, mapping = g.protect("五条悟です")
        assert g.restore(protected, mapping, NameRendering.TRANSLATE).startswith("Satoru Gojo")

    def test_preserve_rendering_keeps_original_script(self, glossary):
        protected, mapping = glossary.protect("五条悟")
        assert glossary.restore(protected, mapping, NameRendering.PRESERVE) == "五条悟"

    def test_romanize_rendering_shows_both(self, glossary):
        protected, mapping = glossary.protect("五条悟")
        out = glossary.restore(protected, mapping, NameRendering.PRESERVE_AND_ROMANIZE)
        assert out == "五条悟 (Satoru Gojo)"

    def test_dropped_sentinel_does_not_leak_to_the_user(self, glossary):
        _, mapping = glossary.protect("五条悟")
        out = glossary.restore("the model  VBX9X mangled it", mapping, NameRendering.TRANSLATE)
        assert "VBX" not in out

    def test_empty_glossary_is_a_noop(self):
        g = SessionGlossary()
        text, mapping = g.protect("anything at all")
        assert text == "anything at all" and mapping == {}

    def test_korean_term(self, glossary):
        protected, mapping = glossary.protect("방탄소년단의 무대")
        assert "BTS" in glossary.restore(protected, mapping, NameRendering.TRANSLATE)

    def test_malformed_entries_are_skipped(self):
        g = SessionGlossary.from_payload({"terms": [
            {"source": "", "target": "x"},
            {"source": "y"},
            "not a dict",
            {"source": "ok", "target": "fine"},
        ]})
        assert len(g) == 1


class TestHonorifics:
    def test_preserved_when_policy_asks(self):
        out = apply_policy("田中さんはどこ", "Where is Mr. Tanaka?",
                           HonorificPolicy.PRESERVE_HONORIFICS)
        assert out == "Where is Tanaka-san?"

    def test_natural_english_leaves_translation_alone(self):
        out = apply_policy("田中さんはどこ", "Where is Mr. Tanaka?",
                           HonorificPolicy.NATURAL_ENGLISH)
        assert out == "Where is Mr. Tanaka?"

    def test_sensei_maps_from_teacher(self):
        out = apply_policy("佐藤先生が来た", "Teacher Sato arrived.",
                           HonorificPolicy.PRESERVE_HONORIFICS)
        assert "Sato-sensei" in out

    def test_ambiguous_source_is_left_untouched(self):
        """Two different honorifics cannot be attributed without alignment."""
        out = apply_policy("田中さんと佐藤ちゃん", "Mr. Tanaka and Miss Sato",
                           HonorificPolicy.PRESERVE_HONORIFICS)
        assert out == "Mr. Tanaka and Miss Sato"

    def test_no_honorific_in_source_is_a_noop(self):
        out = apply_policy("これはペンです", "This is a pen.",
                           HonorificPolicy.PRESERVE_HONORIFICS)
        assert out == "This is a pen."

    def test_detects_korean_terms(self):
        assert detect_honorifics("선배님 안녕하세요", "ko") == ["-seonbae"]
