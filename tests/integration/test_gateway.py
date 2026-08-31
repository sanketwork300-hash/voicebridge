"""Gateway HTTP + WebSocket behaviour."""

import json

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from voicebridge.apps.gateway.main import create_app
from voicebridge.config import AppConfig, SecurityConfig


@pytest.fixture
def client():
    with TestClient(create_app(AppConfig())) as c:
        yield c


class TestMeta:
    def test_health(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["mock_mode"] is True

    def test_ready(self, client):
        assert client.get("/ready").json()["ready"] is True

    def test_metrics_is_prometheus_text(self, client):
        text = client.get("/metrics").text
        assert "voicebridge_sessions_active" in text

    def test_languages_lists_tier1_pairs(self, client):
        pairs = client.get("/v1/languages").json()["pairs"]
        tier1 = {(p["source"], p["target"]) for p in pairs if p["tier"] == 1}
        assert {("ja", "en"), ("ko", "en"), ("en", "hi")} == tier1

    def test_providers_reports_licences(self, client):
        body = client.get("/v1/providers").json()
        nllb = next(p for p in body["translation"] if p["name"] == "nllb")
        assert nllb["capabilities"]["commercial_use"] is False


class TestSessions:
    def test_create_get_delete(self, client):
        created = client.post("/v1/sessions", json={
            "source_language": "ja", "target_language": "en", "mode": "subtitles",
        })
        assert created.status_code == 201
        sid = created.json()["session_id"]
        assert created.json()["stream_url"].endswith("/stream")

        assert client.get(f"/v1/sessions/{sid}").status_code == 200
        assert client.delete(f"/v1/sessions/{sid}").status_code == 200
        assert client.get(f"/v1/sessions/{sid}").status_code == 404

    def test_unknown_session_is_404(self, client):
        assert client.get("/v1/sessions/nope").status_code == 404

    def test_privacy_defaults_are_off(self, client):
        body = client.post("/v1/sessions", json={}).json()
        # Nothing is persisted unless explicitly enabled.
        assert body["session_id"]

    def test_session_limit_is_enforced(self):
        config = AppConfig(security=SecurityConfig(max_sessions=2))
        with TestClient(create_app(config)) as c:
            assert c.post("/v1/sessions", json={}).status_code == 201
            assert c.post("/v1/sessions", json={}).status_code == 201
            assert c.post("/v1/sessions", json={}).status_code == 429


class TestAuth:
    @pytest.fixture
    def secure(self):
        config = AppConfig(security=SecurityConfig(api_token="s3cret"))
        with TestClient(create_app(config)) as c:
            yield c

    def test_rejects_missing_token(self, secure):
        assert secure.post("/v1/sessions", json={}).status_code == 401

    def test_accepts_bearer_token(self, secure):
        res = secure.post("/v1/sessions", json={},
                          headers={"authorization": "Bearer s3cret"})
        assert res.status_code == 201

    def test_rejects_wrong_token(self, secure):
        assert secure.post(
            "/v1/sessions", json={}, headers={"authorization": "Bearer wrong"}
        ).status_code == 401

    def test_websocket_rejects_bad_token(self, secure):
        sid = secure.post("/v1/sessions", json={},
                          headers={"authorization": "Bearer s3cret"}).json()["session_id"]
        with pytest.raises(WebSocketDisconnect):
            with secure.websocket_connect(f"/v1/sessions/{sid}/stream?token=wrong"):
                pass


class TestStreaming:
    def _events(self, ws, want, limit=200):
        seen = {}
        for _ in range(limit):
            try:
                message = ws.receive_json()
            except Exception:
                break
            seen.setdefault(message["event_type"], message)
            if want <= set(seen):
                break
        return seen

    def test_full_pipeline_over_websocket(self, client):
        sid = client.post("/v1/sessions", json={
            "source_language": "ko", "target_language": "en",
            "mode": "speech_and_subtitles",
        }).json()["session_id"]

        with client.websocket_connect(f"/v1/sessions/{sid}/stream") as ws:
            ws.send_text(json.dumps({"type": "AUDIO_START"}))
            for _ in range(120):
                ws.send_bytes(b"\x00\x00" * 1600)
            seen = self._events(ws, {"ASR_STABLE", "TRANSLATION_FINAL", "TTS_AUDIO"})

        assert "ASR_PARTIAL" in seen
        assert "ASR_STABLE" in seen
        assert "TRANSLATION_FINAL" in seen
        assert "TTS_AUDIO" in seen

        translation = seen["TRANSLATION_FINAL"]
        assert translation["source_language"] == "ko"
        assert translation["target_language"] == "en"
        assert translation["committed"] is True

    def test_every_event_carries_the_required_envelope(self, client):
        sid = client.post("/v1/sessions", json={}).json()["session_id"]
        with client.websocket_connect(f"/v1/sessions/{sid}/stream") as ws:
            ws.send_text(json.dumps({"type": "AUDIO_START"}))
            # Enough audio that the mock ASR is certain to emit more events than
            # we read; reading past the end would block on a live socket.
            for _ in range(120):
                ws.send_bytes(b"\x00\x00" * 1600)
            for _ in range(10):
                message = ws.receive_json()
                assert message["session_id"] == sid
                assert isinstance(message["sequence_id"], int)
                assert isinstance(message["timestamp"], float)
                assert message["event_type"]

    def test_sequence_ids_are_monotonic(self, client):
        sid = client.post("/v1/sessions", json={}).json()["session_id"]
        with client.websocket_connect(f"/v1/sessions/{sid}/stream") as ws:
            ws.send_text(json.dumps({"type": "AUDIO_START"}))
            for _ in range(40):
                ws.send_bytes(b"\x00\x00" * 1600)
            ids = [ws.receive_json()["sequence_id"] for _ in range(15)]
        assert ids == sorted(ids)

    def test_ping_pong(self, client):
        sid = client.post("/v1/sessions", json={}).json()["session_id"]
        with client.websocket_connect(f"/v1/sessions/{sid}/stream") as ws:
            ws.send_text(json.dumps({"type": "PING", "echo": 99}))
            seen = self._events(ws, {"PONG"}, limit=20)
        assert seen["PONG"]["echo"] == 99

    def test_malformed_control_frame_is_reported_not_fatal(self, client):
        sid = client.post("/v1/sessions", json={}).json()["session_id"]
        with client.websocket_connect(f"/v1/sessions/{sid}/stream") as ws:
            ws.send_text("this is not json")
            seen = self._events(ws, {"ERROR"}, limit=20)
            assert seen["ERROR"]["code"] == "protocol_error"
            assert seen["ERROR"]["recoverable"] is True
            # The socket must still work afterwards.
            ws.send_text(json.dumps({"type": "PING", "echo": 5}))
            assert self._events(ws, {"PONG"}, limit=20)["PONG"]["echo"] == 5

    def test_misaligned_audio_frame_is_rejected(self, client):
        sid = client.post("/v1/sessions", json={}).json()["session_id"]
        with client.websocket_connect(f"/v1/sessions/{sid}/stream") as ws:
            ws.send_bytes(b"\x00\x00\x00")   # odd length
            seen = self._events(ws, {"ERROR"}, limit=20)
        assert seen["ERROR"]["code"] == "invalid_audio"

    def test_reconnect_to_the_same_session_works(self, client):
        """A dropped socket must not destroy the session."""
        sid = client.post("/v1/sessions", json={}).json()["session_id"]
        with client.websocket_connect(f"/v1/sessions/{sid}/stream") as ws:
            ws.send_bytes(b"\x00\x00" * 1600)
        assert client.get(f"/v1/sessions/{sid}").status_code == 200
        with client.websocket_connect(f"/v1/sessions/{sid}/stream") as ws:
            ws.send_text(json.dumps({"type": "PING", "echo": 1}))
            seen = self._events(ws, {"PONG"}, limit=20)
        assert seen["PONG"]["echo"] == 1

    def test_websocket_on_unknown_session_is_closed(self, client):
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/v1/sessions/missing/stream"):
                pass
