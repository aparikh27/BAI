import json

from fastapi.testclient import TestClient

from backend.api import speech_api
from main import app


def test_speech_stream_route_registered():
    router_paths = [route.path for route in speech_api.speech_router.routes]
    assert "/stream-speech" in router_paths


def test_speech_stream_emits_sse_payload(monkeypatch):
    class FakeStreamer:
        def stream_and_transcribe(self, stop_event=None):
            yield "move forward"

    class FakeBrain:
        def process_task(self, text):
            return '[{"action":"move","target":"forward"}]'

    monkeypatch.setattr(
        speech_api,
        "get_speech_components",
        lambda: (FakeStreamer(), FakeBrain()),
    )

    client = TestClient(app)
    with client.stream("GET", "/api/stream-speech") as response:
        assert response.status_code == 200
        payload = b"".join(response.iter_bytes()).decode("utf-8")
        events = [
            json.loads(line.removeprefix("data: "))
            for line in payload.splitlines()
            if line.startswith("data: ")
        ]
        assert events == [
            {"transcript": "move forward"},
            {"command": '[{"action":"move","target":"forward"}]'},
        ]
