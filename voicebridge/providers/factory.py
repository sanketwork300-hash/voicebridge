"""Provider construction shared by realtime sessions and file jobs.

Models are large, so every provider instance is process-wide and cached by its
*effective configuration*: a realtime session and a file job that both use
``contextual`` with the same model share one loaded model; a job that asks for
a different model gets its own instance. Nothing is constructed until it is
first asked for (lazy loading), and :meth:`unload` releases instances.

Configuration lookup for ``kind`` (``asr``/``translation``/``tts``/``s2st``):

* ``providers.<kind>`` -- the selected provider and its options; options for a
  specific provider may also be nested under its name
  (``providers.asr.whisperlivekit.model``);
* ``providers.fallback_<kind>`` -- used when a job names that provider, and for
  ``tts`` also as the per-language fallback (see ``LanguageRoutedTTS``);
* ``providers.<kind>_presets.<preset>`` -- named presets such as the
  translation-quality levels ``fast`` / ``balanced`` / ``high_quality``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from voicebridge.providers.base import ProviderUnavailable
from voicebridge.providers.registry import (
    ProviderSet,
    asr_registry,
    audio_event_registry,
    load_builtin_providers,
    s2st_registry,
    translation_registry,
    tts_registry,
    vad_registry,
)
from voicebridge.runtime import resolve_device

logger = logging.getLogger(__name__)

REGISTRIES = {
    "asr": asr_registry,
    "translation": translation_registry,
    "tts": tts_registry,
    "s2st": s2st_registry,
    "vad": vad_registry,
    "audio_event": audio_event_registry,
}


class ProviderFactory:
    def __init__(self, provider_config: dict[str, Any] | None = None,
                 runtime: dict[str, Any] | None = None):
        load_builtin_providers()
        self.config = provider_config or {}
        self.runtime = runtime or {}
        self._cache: dict[str, Any] = {}

    # -- configuration -------------------------------------------------------

    def resolve(self, kind: str, name: str | None = None, preset: str | None = None
                ) -> tuple[str, dict[str, Any]]:
        if preset:
            presets = self.config.get(f"{kind}_presets") or {}
            if preset not in presets:
                raise ProviderUnavailable(f"unknown {kind} preset {preset!r}")
            cfg = dict(presets[preset])
            return str(cfg.pop("provider")), cfg
        candidates = [self.config.get(kind) or {}, self.config.get(f"fallback_{kind}") or {}]
        base = dict(candidates[0])
        selected = str(base.get("provider") or base.get("name") or "mock")
        wanted = name or selected
        cfg: dict[str, Any] = {}
        for cand in candidates:
            if str(cand.get("provider") or cand.get("name") or "") == wanted:
                cfg = dict(cand)
                break
        cfg.pop("provider", None)
        cfg.pop("name", None)
        nested = base.get(wanted)
        for key in list(cfg):
            if isinstance(cfg[key], dict) and key in REGISTRIES[kind].names():
                cfg.pop(key)  # options nested for *other* providers
        if isinstance(nested, dict):
            cfg.update(nested)
        # Runtime defaults apply unless the provider config overrides them.
        if kind in ("asr", "translation", "tts", "s2st", "audio_event"):
            for key in ("device", "dtype"):
                if key in self.runtime and key not in cfg and self.runtime[key] != "auto":
                    cfg[key] = self.runtime[key]
        return wanted, cfg

    # -- construction ---------------------------------------------------------

    def get(self, kind: str, name: str | None = None, preset: str | None = None,
            **overrides: Any) -> Any:
        provider, cfg = self.resolve(kind, name, preset)
        cfg.update({k: v for k, v in overrides.items() if v is not None})
        key = f"{kind}:{provider}:{json.dumps(cfg, sort_keys=True, default=str)}"
        instance = self._cache.get(key)
        if instance is None:
            instance = REGISTRIES[kind].create(provider, **cfg)
            self._cache[key] = instance
        return instance

    def tts(self, name: str | None = None) -> Any:
        primary = self.get("tts", name)
        fallback_cfg = self.config.get("fallback_tts") or {}
        fallback_name = fallback_cfg.get("provider")
        if fallback_name and fallback_name != getattr(primary, "name", None):
            from voicebridge.providers.tts.routed import LanguageRoutedTTS

            key = f"tts-routed:{id(primary)}"
            routed = self._cache.get(key)
            if routed is None:
                try:
                    routed = LanguageRoutedTTS(primary, self.get("tts", fallback_name))
                except ProviderUnavailable as exc:
                    logger.warning("fallback TTS unavailable: %s", exc)
                    return primary
                self._cache[key] = routed
            return routed
        return primary

    def speech_gate_providers(self, realtime: bool = False) -> tuple[Any, Any]:
        """VAD and event classifier for the speech gate.

        ``realtime``: live sessions use the classifier only when
        ``speech_gate.realtime_event_classifier`` allows it. The default
        ``auto`` enables it on GPU/MPS and disables it on CPU, where the AST
        model measured 0.7-0.85 s per 0.6 s window (slower than real time);
        the live gate then relies on Silero VAD plus ASR validation.
        """
        gate = self.config.get("speech_gate") or {}
        if not gate.get("enabled"):
            return None, None
        vad = classifier = None
        if gate.get("vad"):
            opts = gate.get("vad_options") or {}
            vad = self._cached("vad", str(gate["vad"]), opts)
        event = gate.get("event_classifier")
        if realtime and event:
            mode = str(gate.get("realtime_event_classifier", "auto")).lower()
            if mode in ("false", "off", "no") or (
                    mode == "auto" and resolve_device(self.runtime.get("device", "auto")) == "cpu"):
                event = None
        if event:
            name = event if isinstance(event, str) else "ast"
            opts = dict(gate.get("classifier_options") or {})
            if "device" in self.runtime and self.runtime["device"] != "auto":
                opts.setdefault("device", self.runtime["device"])
            classifier = self._cached("audio_event", name, opts)
        return vad, classifier

    def _cached(self, kind: str, name: str, opts: dict[str, Any]) -> Any:
        key = f"{kind}:{name}:{json.dumps(opts, sort_keys=True, default=str)}"
        if key not in self._cache:
            self._cache[key] = REGISTRIES[kind].create(name, **opts)
        return self._cache[key]

    def s2st(self, name: str | None = None) -> Any:
        return self.get("s2st", name)

    def translation_for(self, target_language: str | None, name: str | None = None,
                        preset: str | None = None) -> Any:
        """Translation provider for a target language.

        An explicit provider or preset wins. Otherwise
        ``providers.translation_by_target.<lang>`` overrides the default, so a
        model that benchmarks badly for one language is not used for it.
        """
        override = (self.config.get("translation_by_target") or {}).get(target_language or "")
        if name or preset or not override:
            return self.get("translation", name, preset)
        cfg = dict(override)
        provider = str(cfg.pop("provider"))
        key = f"translation:{provider}:{json.dumps(cfg, sort_keys=True, default=str)}"
        if key not in self._cache:
            self._cache[key] = REGISTRIES["translation"].create(provider, **cfg)
        return self._cache[key]

    def provider_set(self, wants_tts: bool = True, asr: str | None = None,
                     translation: str | None = None, translation_preset: str | None = None,
                     tts: str | None = None, target_language: str | None = None,
                     realtime: bool = False) -> ProviderSet:
        tts_engine = None
        if wants_tts:
            try:
                tts_engine = self.tts(tts)
            except ProviderUnavailable as exc:
                logger.warning("TTS unavailable, continuing without it: %s", exc)
        vad, classifier = self.speech_gate_providers(realtime)
        return ProviderSet(
            asr=self.get("asr", asr),
            translation=self.translation_for(target_language, translation, translation_preset),
            tts=tts_engine,
            vad=vad,
            audio_event=classifier,
        )

    async def warmup(self, kinds: tuple[str, ...] | list[str] | None = None) -> list[str]:
        """Load the providers listed in ``runtime.warmup`` (default: ASR only).

        Everything else loads lazily on first use, so an idle server does not
        hold every model in memory.
        """
        kinds = kinds if kinds is not None else tuple(self.runtime.get("warmup") or ("asr",))
        failed = []
        for kind in kinds:
            try:
                provider = self.tts() if kind == "tts" else self.get(kind)
                await provider.warmup()
            except Exception as exc:
                logger.warning("%s provider warmup failed: %s", kind, exc)
                failed.append(f"{kind}: {exc}")
        return failed

    async def unload(self) -> None:
        for instance in list(self._cache.values()):
            close = getattr(instance, "close", None) or getattr(instance, "shutdown", None)
            if close is not None:
                try:
                    await close()
                except Exception as exc:  # pragma: no cover
                    logger.debug("error closing provider: %s", exc)
        self._cache.clear()
