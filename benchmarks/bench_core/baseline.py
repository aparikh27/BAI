"""The pure-Python legacy runtime the hybrid one is measured against.

This is the control arm. It implements, in idiomatic Python, exactly what
EMBER does for AgentCore: accept command messages, route them by topic onto
queues, hand them to worker tasks that actuate and acknowledge, publish
periodic telemetry back, and run a watchdog that stops the robot if the
commanding side goes quiet. Same semantics, same acknowledgement points, same
watchdog deadline — different implementation technology.

Getting this arm right is what makes the whole comparison honest, so three
choices are worth stating up front.

**It uses `asyncio.Queue`, not `queue.Queue`.** The brief asks for asyncio
queues specifically, and it is also what a pure-Python version of this
subsystem would use: the legacy path is I/O-bound message routing, which is
what asyncio is for.

**It really serializes.** `agents/messaging/helpers.py` already carries
`message_to_dict`/`message_from_dict`, so a Message crossing a boundary in
the legacy framework becomes a dict and then bytes. Skipping that here would
delete the exact cost the binary protocol replaces.

**It offers two transports.** `in_process` is the architecture AgentCore
actually had before EMBER: everything in one interpreter, no socket. That is
the real baseline, and it is the one the headline comparison uses.
`loopback` puts the same Python runtime behind a TCP socket, matching the
hybrid arm's transport exactly. Reporting both separates "what did the binary
protocol and the C++ runtime buy" from "what did merely moving to a socket
cost", which a single-transport comparison would blur together — and the
loopback arm is the fairer of the two to EMBER's critics, since it hands the
Python arm the same syscall overhead rather than pretending IPC is free.
"""

from __future__ import annotations

import asyncio
import json
import pickle
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from agents.messaging.ember_wire import (
    AckResult,
    CommandOp,
    TelemetryFrame,
    TelemetrySubsystem,
    fletcher16,
    fnv1a_32,
)
from benchmarks.bench_core import diagnostics as diag
from benchmarks.bench_core.diagnostics import RuntimeDiagnostics

# ---------------------------------------------------------------------------
# Codecs
# ---------------------------------------------------------------------------


class LegacyCodec(Protocol):
    """Object <-> bytes, the pure-Python counterpart of EMBER's
    `Serializer`/`FrameCodec` pair."""

    name: str

    def encode(self, obj: dict[str, Any]) -> bytes: ...
    def decode(self, payload: bytes) -> dict[str, Any]: ...


class JsonCodec:
    """`json` with the most compact separators. The separators matter: the
    default `", "`/`": "` add two bytes per field, which on a 5-field command
    message is a ~15% wire-size penalty that has nothing to do with JSON
    itself and would overstate the binary protocol's advantage."""

    name = "json"

    def encode(self, obj: dict[str, Any]) -> bytes:
        return json.dumps(obj, separators=(",", ":")).encode("utf-8")

    def decode(self, payload: bytes) -> dict[str, Any]:
        return json.loads(payload.decode("utf-8"))


class PickleCodec:
    """`pickle` at the highest protocol.

    Faster than JSON for Python-to-Python, and included because it is what a
    Python-only design would reach for once JSON showed up in a profile. It is
    also unusable across the language boundary this bridge actually spans,
    which is itself part of the result: the fastest pure-Python option is the
    one that cannot talk to the C++ side at all.
    """

    name = "pickle"

    def encode(self, obj: dict[str, Any]) -> bytes:
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    def decode(self, payload: bytes) -> dict[str, Any]:
        return pickle.loads(payload)


class ChecksummedJsonCodec(JsonCodec):
    """JSON plus a Fletcher-16 trailer, computed in Python.

    The like-for-like comparison against EMBER's framing: without it, the
    binary arm would be paying for integrity checking that the JSON arm
    silently skips, and the difference would be credited to the encoding
    rather than to the checksum.
    """

    name = "json+fletcher16"

    def encode(self, obj: dict[str, Any]) -> bytes:
        body = super().encode(obj)
        return body + struct.pack(">H", fletcher16(body))

    def decode(self, payload: bytes) -> dict[str, Any]:
        if len(payload) < 2:
            raise ValueError("payload too short to carry a checksum")
        body, trailer = payload[:-2], payload[-2:]
        (expected,) = struct.unpack(">H", trailer)
        if expected != fletcher16(body):
            raise ValueError("checksum mismatch")
        return super().decode(body)


CODECS: dict[str, LegacyCodec] = {
    "json": JsonCodec(),
    "json+fletcher16": ChecksummedJsonCodec(),
    "pickle": PickleCodec(),
}


# ---------------------------------------------------------------------------
# Length-prefixed framing (the legacy transport's answer to StreamFrameReader)
# ---------------------------------------------------------------------------

_LENGTH_PREFIX = struct.Struct(">I")


def frame_payload(payload: bytes) -> bytes:
    return _LENGTH_PREFIX.pack(len(payload)) + payload


class LengthPrefixedReader:
    """Reassembles length-prefixed payloads out of a TCP byte stream.

    The pure-Python counterpart of `StreamFrameReader`, and deliberately the
    naive version: a 4-byte length and no magic, so there is nothing to resync
    against. That asymmetry is real and worth seeing in the results — EMBER's
    framing costs more per frame and buys the ability to recover from a
    corrupt byte, which this one structurally cannot do.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes) -> None:
        self._buffer.extend(data)

    def try_extract(self) -> bytes | None:
        if len(self._buffer) < _LENGTH_PREFIX.size:
            return None
        (length,) = _LENGTH_PREFIX.unpack_from(self._buffer, 0)
        total = _LENGTH_PREFIX.size + length
        if len(self._buffer) < total:
            return None
        payload = bytes(self._buffer[_LENGTH_PREFIX.size : total])
        del self._buffer[:total]
        return payload

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# Message shapes
# ---------------------------------------------------------------------------

MSG_COMMAND = "cmd"
MSG_ACK = "ack"
MSG_TELEMETRY = "tel"
MSG_HEARTBEAT = "hb"


def command_message(sequence: int, op: CommandOp, param: float, request_id: str) -> dict[str, Any]:
    """The legacy wire object. Field-for-field equivalent to `CommandFrame`,
    so the two arms carry the same information and the size/latency delta is
    attributable to the encoding rather than to one side sending less."""
    return {
        "t": MSG_COMMAND,
        "sequence": sequence,
        "timestamp_ns": time.monotonic_ns(),
        "op": int(op),
        "param": param,
        "request_id_hash": fnv1a_32(request_id) if request_id else 0,
    }


def ack_message(sequence: int, result: AckResult, request_id_hash: int) -> dict[str, Any]:
    return {
        "t": MSG_ACK,
        "ack_sequence": sequence,
        "result": int(result),
        "request_id_hash": request_id_hash,
    }


def telemetry_message(
    sequence: int,
    subsystem: TelemetrySubsystem,
    value_a: float = 0.0,
    value_b: float = 0.0,
    code: int = 0,
    text: str = "",
) -> dict[str, Any]:
    return {
        "t": MSG_TELEMETRY,
        "sequence": sequence,
        "timestamp_ns": time.monotonic_ns(),
        "subsystem": int(subsystem),
        "value_a": value_a,
        "value_b": value_b,
        "code": code,
        "text": text,
    }


def telemetry_frame_from_message(message: dict[str, Any]) -> TelemetryFrame:
    """Rebuilds the shared `TelemetryFrame` type from a legacy dict.

    Both arms hand the rest of the system the same object, which is what lets
    one unmodified `EmberRobotDriver` sit on top of either client — the driver
    and everything above it is held constant across the comparison, so any
    measured difference is in the runtime below it.
    """
    return TelemetryFrame(
        sequence=int(message["sequence"]),
        subsystem=TelemetrySubsystem(int(message["subsystem"])),
        timestamp_ns=int(message["timestamp_ns"]),
        value_a=float(message["value_a"]),
        value_b=float(message["value_b"]),
        code=int(message["code"]),
        text=str(message["text"]),
    )


# Topic routing, mirroring `topic_for()` in edge/bridge/ember_bridge_adapter.cpp
# exactly — same seven topics, same names, so both arms perform the same
# number of routing decisions against the same key space.
TOPIC_FOR_OP: dict[CommandOp, str] = {
    CommandOp.MOVE_FORWARD: "cmd/motion/move_forward",
    CommandOp.TURN: "cmd/motion/turn",
    CommandOp.STOP: "cmd/motion/stop",
    CommandOp.RAISE_ARM: "cmd/manipulator/raise_arm",
    CommandOp.LOWER_ARM: "cmd/manipulator/lower_arm",
    CommandOp.GRAB_ITEM: "cmd/manipulator/grab_item",
    CommandOp.RELEASE_ITEM: "cmd/manipulator/release_item",
}

MOTION_OPS = frozenset({CommandOp.MOVE_FORWARD, CommandOp.TURN})


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LegacyRuntimeConfig:
    codec_name: str = "json"
    transport: str = "in_process"  # "in_process" | "loopback"
    host: str = "127.0.0.1"
    port: int = 0
    actuation_us: int = 0
    telemetry_hz: float = 0.0
    telemetry_value: float = 1.0
    # Matches BridgeConfig's defaults so the two watchdogs are comparable.
    watchdog_deadline_s: float = 0.25
    watchdog_period_s: float = 0.05
    max_buffered_telemetry: int = 256
    command_timeout_s: float = 2.0
    heartbeat_interval_s: float = 0.1

    @property
    def codec(self) -> LegacyCodec:
        return CODECS[self.codec_name]


@dataclass
class _PendingCommand:
    request_id: str
    future: asyncio.Future
    sent_ns: int = field(default_factory=time.perf_counter_ns)


# ---------------------------------------------------------------------------
# The runtime
# ---------------------------------------------------------------------------


class LegacyRuntime:
    """Pure-Python equivalent of EMBER's Coordinator + subsystem workers +
    telemetry publisher + watchdog, running on one asyncio event loop.

    Owns its own loop on a background thread for the same reason
    `EmberBridgeClient` does: the agent framework above it
    (`agents/agent/base_agent.py` and everything under it) is synchronous, and
    the comparison has to keep that boundary identical in both arms or it
    would be measuring an API change rather than a runtime change.
    """

    def __init__(self, config: LegacyRuntimeConfig, diagnostics: RuntimeDiagnostics):
        self.config = config
        self.diagnostics = diagnostics
        self.codec = config.codec

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stopping = threading.Event()

        # One queue per topic, exactly as EMBER gives each Subscriber its own
        # ThreadSafeQueue rather than sharing one and filtering.
        self._topic_queues: dict[str, asyncio.Queue] = {}
        self._worker_tasks: list[asyncio.Task] = []
        self._service_tasks: list[asyncio.Task] = []

        self._sequence_lock = threading.Lock()
        self._sequence = 0
        self._telemetry_sequence = 0

        self._pending: dict[int, _PendingCommand] = {}

        self._telemetry_probe = diagnostics.contention_probe(diag.TELEMETRY_CACHE_LOCK)
        self._latest_telemetry: dict[int, TelemetryFrame] = {}
        self._on_telemetry: Callable[[TelemetryFrame], None] | None = None

        self._last_inbound_ns = time.monotonic_ns()
        self._motion_in_flight = False
        self._watchdog_stops = diagnostics.counter("watchdog_stops")

        self._server: asyncio.AbstractServer | None = None
        self._client_writer: asyncio.StreamWriter | None = None
        self._server_writer: asyncio.StreamWriter | None = None
        self._connected = threading.Event()

        self.actual_port = config.port

    # ---- lifecycle ----------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run_loop, name="legacy-runtime", daemon=True)
        self._thread.start()
        if not self._ready.wait(10.0):
            raise RuntimeError("LegacyRuntime failed to start within 10s")

    def stop(self) -> None:
        # Cleared before the loop is torn down, not after: send_command() gates
        # on is_connected(), and leaving it true through shutdown lets a caller
        # reach a closed event loop and get a RuntimeError instead of the
        # REJECTED that EmberBridgeClient returns in the same situation. The
        # two arms have to fail identically or the comparison is between two
        # different contracts.
        self._stopping.set()
        self._ready.clear()
        self._connected.clear()
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(lambda: None)  # wake the loop promptly
            except RuntimeError:
                pass  # already closed; nothing to wake
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        finally:
            self._loop.close()

    async def _main(self) -> None:
        for topic in TOPIC_FOR_OP.values():
            self._topic_queues[topic] = asyncio.Queue()
            self._worker_tasks.append(asyncio.create_task(self._subsystem_worker(topic)))

        if self.config.transport == "loopback":
            await self._start_loopback()

        self._service_tasks.append(asyncio.create_task(self._watchdog_task()))
        if self.config.telemetry_hz > 0:
            self._service_tasks.append(asyncio.create_task(self._telemetry_task()))

        self._ready.set()

        # Poll `_stopping` rather than waiting on an asyncio.Event set from
        # another thread: the flag is a threading.Event so `stop()` works
        # whether or not the loop is currently accepting call_soon_threadsafe.
        while not self._stopping.is_set():
            await asyncio.sleep(0.02)

        for task in self._worker_tasks + self._service_tasks:
            task.cancel()
        await asyncio.gather(*self._worker_tasks, *self._service_tasks, return_exceptions=True)

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        if self._client_writer is not None:
            self._client_writer.close()

    # ---- loopback transport -------------------------------------------

    async def _start_loopback(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_connection, self.config.host, self.config.port
        )
        self.actual_port = self._server.sockets[0].getsockname()[1]

        reader, writer = await asyncio.open_connection(self.config.host, self.actual_port)
        self._client_writer = writer
        self._service_tasks.append(asyncio.create_task(self._client_reader_task(reader)))
        self._connected.set()

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Server side of the loopback link: the legacy analogue of
        `EmberBridgeAdapter::reader_loop`."""
        self._server_writer = writer
        framer = LengthPrefixedReader()
        try:
            while not self._stopping.is_set():
                chunk = await reader.read(4096)
                if not chunk:
                    break
                self.diagnostics.counter(diag.BYTES_RECEIVED).increment(len(chunk))
                framer.feed(chunk)
                while (payload := framer.try_extract()) is not None:
                    await self._handle_inbound_payload(payload)
        except (ConnectionError, OSError, asyncio.CancelledError):
            pass
        finally:
            writer.close()
            self._server_writer = None

    async def _client_reader_task(self, reader: asyncio.StreamReader) -> None:
        framer = LengthPrefixedReader()
        try:
            while not self._stopping.is_set():
                chunk = await reader.read(4096)
                if not chunk:
                    break
                framer.feed(chunk)
                while (payload := framer.try_extract()) is not None:
                    self._handle_client_payload(payload)
        except (ConnectionError, OSError, asyncio.CancelledError):
            pass

    async def _send_to_client(self, message: dict[str, Any]) -> None:
        payload = self.codec.encode(message)
        if self.config.transport == "loopback":
            writer = self._server_writer
            if writer is None:
                return
            writer.write(frame_payload(payload))
            self.diagnostics.counter(diag.BYTES_SENT).increment(len(payload) + 4)
            await writer.drain()
        else:
            # In-process: the payload is still encoded above, because the
            # legacy framework serialized messages regardless of whether they
            # crossed a socket — dropping the encode here would credit the
            # in-process arm with a saving it never had.
            self._handle_client_payload(payload)

    # ---- inbound handling ---------------------------------------------

    async def _handle_inbound_payload(self, payload: bytes) -> None:
        self._last_inbound_ns = time.monotonic_ns()
        try:
            message = self.codec.decode(payload)
        except Exception:
            self.diagnostics.counter(diag.FRAMES_CORRUPT).increment()
            return

        kind = message.get("t")
        if kind == MSG_HEARTBEAT:
            return
        if kind != MSG_COMMAND:
            return

        await self._route_command(message)

    async def _route_command(self, message: dict[str, Any]) -> None:
        op = CommandOp(int(message["op"]))
        if op in MOTION_OPS:
            self._motion_in_flight = True
        elif op == CommandOp.STOP:
            self._motion_in_flight = False

        topic = TOPIC_FOR_OP[op]
        queue = self._topic_queues[topic]
        queue.put_nowait(message)
        self.diagnostics.gauge(diag.INBOUND_QUEUE_DEPTH).set(queue.qsize())

        # Acknowledged on receipt, before actuation — matching
        # handle_command_frame()'s Accepted ack, so the two arms' measured
        # round trips end at the same semantic point.
        await self._send_to_client(
            ack_message(int(message["sequence"]), AckResult.ACCEPTED, int(message["request_id_hash"]))
        )

    async def _subsystem_worker(self, topic: str) -> None:
        """One worker per topic, the analogue of the per-Subscriber threads in
        edge/tests/integration/bench_harness_server.cpp."""
        queue = self._topic_queues[topic]
        actuation_s = self.config.actuation_us / 1e6
        while True:
            message = await queue.get()
            self.diagnostics.gauge(diag.INBOUND_QUEUE_DEPTH).set(queue.qsize())
            if actuation_s > 0:
                await asyncio.sleep(actuation_s)
            self.diagnostics.rate(diag.TASK_THROUGHPUT).mark()
            await self._send_to_client(
                ack_message(
                    int(message["sequence"]),
                    AckResult.COMPLETED,
                    int(message["request_id_hash"]),
                )
            )

    def _handle_client_payload(self, payload: bytes) -> None:
        """Client side of the link: resolves pending commands and caches
        telemetry. The counterpart of `EmberBridgeClient._handle_decoded`."""
        try:
            message = self.codec.decode(payload)
        except Exception:
            self.diagnostics.counter(diag.FRAMES_CORRUPT).increment()
            return

        kind = message.get("t")
        if kind == MSG_ACK:
            pending = self._pending.pop(int(message["ack_sequence"]), None)
            if pending is not None and not pending.future.done():
                pending.future.set_result(AckResult(int(message["result"])))
                self.diagnostics.counter(diag.COMMANDS_ACKED).increment()
        elif kind == MSG_TELEMETRY:
            frame = telemetry_frame_from_message(message)
            with self._telemetry_probe.guard():
                self._latest_telemetry[int(frame.subsystem)] = frame
            self.diagnostics.counter(diag.TELEMETRY_RECEIVED).increment()
            self.diagnostics.rate(diag.TELEMETRY_LOOP).mark()
            if self._on_telemetry is not None:
                try:
                    self._on_telemetry(frame)
                except Exception:
                    pass

    # ---- telemetry + watchdog ------------------------------------------

    async def _telemetry_task(self) -> None:
        period = 1.0 / self.config.telemetry_hz
        # Absolute deadlines, for the same reason the C++ harness uses them:
        # sleeping for `period` after each publish folds the publish cost into
        # the interval and reports a rate slower than the runtime can achieve.
        next_deadline = time.perf_counter()
        buffered = 0
        while True:
            self._telemetry_sequence += 1
            message = telemetry_message(
                self._telemetry_sequence,
                TelemetrySubsystem.MOTION,
                value_a=self.config.telemetry_value,
            )

            # Drop-oldest above the buffer cap, matching
            # BridgeConfig::max_buffered_telemetry_frames. In-process there is
            # no real queue to overflow, so the cap is applied to the number
            # of samples published within one scheduling slice.
            buffered += 1
            if buffered > self.config.max_buffered_telemetry:
                self.diagnostics.counter("telemetry_dropped").increment()
                buffered = self.config.max_buffered_telemetry
            else:
                await self._send_to_client(message)

            next_deadline += period
            delay = next_deadline - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
                buffered = 0
            else:
                # Behind schedule: yield without sleeping so the loop stays
                # responsive, and let the next iteration catch up.
                next_deadline = time.perf_counter()
                await asyncio.sleep(0)

    async def _watchdog_task(self) -> None:
        """Pure-Python equivalent of `EmberBridgeAdapter::watchdog_tick`.

        Same deadline, same self-issued stop. The point of running it in both
        arms is that its reaction time is a safety property, not a performance
        one: a watchdog that fires late because the interpreter was busy is a
        robot that keeps moving, and the tail of this measurement is the
        number that matters.
        """
        deadline_ns = int(self.config.watchdog_deadline_s * 1e9)
        while True:
            await asyncio.sleep(self.config.watchdog_period_s)
            if not self._motion_in_flight:
                continue
            if time.monotonic_ns() - self._last_inbound_ns <= deadline_ns:
                continue

            self._motion_in_flight = False
            self._watchdog_stops.increment()
            stop_message = command_message(0, CommandOp.STOP, 0.0, "")
            await self._route_command(stop_message)

    # ---- public, synchronous surface (mirrors EmberBridgeClient) --------

    def wait_connected(self, timeout_s: float | None = None) -> bool:
        if self.config.transport == "in_process":
            return self._ready.wait(timeout_s)
        return self._connected.wait(timeout_s)

    def is_connected(self) -> bool:
        if self.config.transport == "in_process":
            return self._ready.is_set()
        return self._connected.is_set()

    def set_telemetry_callback(self, callback: Callable[[TelemetryFrame], None] | None) -> None:
        self._on_telemetry = callback

    def send_command(self, op: CommandOp, param: float = 0.0, request_id: str = "") -> AckResult:
        """Blocks until acknowledged or `command_timeout_s` elapses, returning
        REJECTED rather than raising — identical contract to
        `EmberBridgeClient.send_command`, so `EmberRobotDriver` runs unmodified
        on top of this client and the arms differ only below the driver."""
        if self._loop is None or self._stopping.is_set() or not self.is_connected():
            self.diagnostics.counter(diag.COMMANDS_REJECTED).increment()
            return AckResult.REJECTED

        with self._sequence_lock:
            self._sequence += 1
            sequence = self._sequence

        self.diagnostics.counter(diag.COMMANDS_SENT).increment()
        future = asyncio.run_coroutine_threadsafe(
            self._send_command_async(sequence, op, param, request_id), self._loop
        )
        try:
            return future.result(timeout=self.config.command_timeout_s)
        except Exception:
            self.diagnostics.counter(diag.COMMANDS_TIMED_OUT).increment()
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._pending.pop, sequence, None)
            return AckResult.REJECTED

    async def _send_command_async(
        self, sequence: int, op: CommandOp, param: float, request_id: str
    ) -> AckResult:
        message = command_message(sequence, op, param, request_id)
        ack_future = self._loop.create_future()  # type: ignore[union-attr]
        self._pending[sequence] = _PendingCommand(request_id=request_id, future=ack_future)
        self.diagnostics.gauge(diag.PENDING_COMMANDS).set(len(self._pending))

        payload = self.codec.encode(message)
        if self.config.transport == "loopback":
            writer = self._client_writer
            if writer is None:
                raise ConnectionError("legacy runtime not connected")
            writer.write(frame_payload(payload))
            await writer.drain()
        else:
            await self._handle_inbound_payload(payload)

        result = await ack_future
        self.diagnostics.gauge(diag.PENDING_COMMANDS).set(len(self._pending))
        return result

    def get_latest_telemetry(self, subsystem: TelemetrySubsystem) -> TelemetryFrame | None:
        with self._telemetry_probe.guard():
            return self._latest_telemetry.get(int(subsystem))

    def seed_telemetry(self, subsystem: TelemetrySubsystem, value_a: float) -> None:
        """Injects one telemetry sample directly.

        Used to give the baseline arm the same last-known sensor value the
        hybrid arm gets from EMBER's periodic Motion stream, for workloads
        that read `get_distance_to_front()` but are not themselves measuring
        the telemetry path.
        """
        self._telemetry_sequence += 1
        frame = telemetry_frame_from_message(
            telemetry_message(self._telemetry_sequence, subsystem, value_a=value_a)
        )
        with self._telemetry_probe.guard():
            self._latest_telemetry[int(subsystem)] = frame

    @property
    def watchdog_stop_count(self) -> int:
        return self._watchdog_stops.value

    def suspend_heartbeat(self) -> None:
        """Simulates AgentCore going quiet (a long inference call, a GC pause)
        so the watchdog's reaction can be timed. In-process there is no
        heartbeat task to stop, so `_last_inbound_ns` is simply frozen far
        enough in the past that the next watchdog tick sees the deadline as
        already missed."""
        self._last_inbound_ns = time.monotonic_ns() - int(self.config.watchdog_deadline_s * 2e9)
