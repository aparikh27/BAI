import json

from fastapi.testclient import TestClient

from backend.api import speech_api
from backend.main import app
from messaging import Message, MessageStatus, MessageType


class FakeCoordinator:
    """Minimal Coordinator stand-in for route tests."""

    def __init__(self):
        self.dispatch_calls = []

    def dispatch(self, message: Message) -> Message:
        self.dispatch_calls.append(message)

        if message.receiver == "Audio" and message.action == "transcribe":
            return Message(
                sender="Audio",
                receiver=message.sender,
                action=message.action,
                payload={"text": "move forward"},
                status=MessageStatus.SUCCESS,
                message_type=MessageType.RESPONSE,
                parent_id=message.request_id,
            )

        if message.receiver == "Planner" and message.action == "create_plan":
            plan = [{"action": "move", "target": "forward"}]
            return Message(
                sender="Planner",
                receiver=message.sender,
                action=message.action,
                payload={"text": message.payload.get("text"), "plan": plan},
                status=MessageStatus.SUCCESS,
                message_type=MessageType.RESPONSE,
                parent_id=message.request_id,
            )

        if message.receiver == "Executor" and message.action == "execute":
            return Message(
                sender="Executor",
                receiver=message.sender,
                action=message.action,
                payload={
                    "completed_steps": message.payload.get("plan", []),
                    "total_steps": len(message.payload.get("plan", [])),
                },
                status=MessageStatus.SUCCESS,
                message_type=MessageType.RESPONSE,
                parent_id=message.request_id,
            )

        return Message(
            sender="coordinator",
            receiver=message.sender,
            action=message.action,
            payload={},
            status=MessageStatus.ERROR,
            message_type=MessageType.RESPONSE,
            parent_id=message.request_id,
            error=f"Unexpected dispatch: {message.receiver}/{message.action}",
        )


def test_speech_stream_route_registered():
    router_paths = [route.path for route in speech_api.speech_router.routes]
    assert "/stream-speech" in router_paths


def test_speech_stream_emits_sse_payload(monkeypatch):
    class FakeStreamer:
        def stream_audio(self, stop_event=None):
            yield object()

    fake_coordinator = FakeCoordinator()

    monkeypatch.setattr(
        speech_api,
        "ContinuousAudioStream",
        lambda chunk_duration=3: FakeStreamer(),
    )
    speech_api.set_coordinator(fake_coordinator)

    try:
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
                {"command": '[{"action": "move", "target": "forward"}]'},
            ]
    finally:
        speech_api.set_coordinator(None)


def test_speech_stream_dispatches_through_coordinator(monkeypatch):
    class FakeStreamer:
        def stream_audio(self, stop_event=None):
            yield object()

    fake_coordinator = FakeCoordinator()

    monkeypatch.setattr(
        speech_api,
        "ContinuousAudioStream",
        lambda chunk_duration=3: FakeStreamer(),
    )
    speech_api.set_coordinator(fake_coordinator)

    try:
        client = TestClient(app)
        with client.stream("GET", "/api/stream-speech") as response:
            assert response.status_code == 200
            _ = b"".join(response.iter_bytes())
    finally:
        speech_api.set_coordinator(None)

    receivers = [msg.receiver for msg in fake_coordinator.dispatch_calls]
    assert receivers == ["Audio", "Planner", "Executor"]
