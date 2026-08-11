"""
events.py — Fan-out hub for dashboard telemetry
───────────────────────────────────────────────
The speech SSE endpoint originally owned its event queue directly, which meant
only the microphone loop that created it could ever emit to the dashboard.
Anything else that wants to report progress — a replayed audio file, per-step
execution updates, task completion — had no way in.

``EventHub`` decouples producers from the SSE response: the endpoint subscribes,
any number of producers publish, and each subscriber gets its own queue so a
slow client cannot stall the pipeline.

A short replay buffer is kept for the *current run* so a dashboard that
subscribes a beat late (or that ``EventSource`` silently reconnects) still
renders the events it missed.
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from typing import Any


class EventHub:
    """Thread-safe publish/subscribe fan-out for dashboard events."""

    def __init__(self, replay_size: int = 50):
        self._subscribers: set[queue.Queue] = set()
        self._lock = threading.Lock()
        self._replay: deque[dict[str, Any]] = deque(maxlen=replay_size)

    def subscribe(self) -> queue.Queue:
        """Registers a new subscriber, pre-loaded with the current run's events."""
        subscriber: queue.Queue = queue.Queue()
        with self._lock:
            for event in self._replay:
                subscriber.put(event)
            self._subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue) -> None:
        with self._lock:
            self._subscribers.discard(subscriber)

    def publish(self, event: dict[str, Any]) -> None:
        """Broadcasts one event to every current subscriber."""
        with self._lock:
            self._replay.append(event)
            targets = list(self._subscribers)

        for subscriber in targets:
            # Unbounded queues: publishing never blocks the producing thread.
            subscriber.put(event)

    def start_run(self) -> None:
        """Clears replay history so a new run does not resurface stale state."""
        with self._lock:
            self._replay.clear()

    def end_run(self) -> None:
        """Drops the replay buffer once a run reaches a terminal state.

        Replay exists so a dashboard that subscribes a beat late still sees an
        *in-flight* run. Retaining it after the run finishes means the next
        client to connect is immediately handed a completed/failed task state
        for work it never watched — which renders as a stale "Task Completed"
        banner the moment the page loads.
        """
        with self._lock:
            self._replay.clear()

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)


# Single process-wide hub shared by the API routers.
dashboard_events = EventHub()
