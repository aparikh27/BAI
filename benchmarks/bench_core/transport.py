"""Subsystem micro-benchmarks: data transport/serialization and event handling.

These measure the Python side of each boundary. Their C++ counterparts live
in `edge/benchmarks/bench_bridge.cpp`, which measures the same operations
against the same frame types at the same sizes using the same percentile
definitions, so the two reports can be read side by side.

Two measurement notes that shape everything here:

* Encode/decode of a 21-byte struct runs faster than Windows'
  `perf_counter_ns` can resolve, so these use `time_batched` rather than
  `time_each_call`. The per-call means are meaningful; the tails from these
  particular rows are not, and are not quoted as determinism evidence.

* The queue benchmarks compare `asyncio.Queue` and `queue.Queue` against each
  other and against EMBER's `ThreadSafeQueue` figures from the C++ suite.
  `asyncio.Queue` is not thread-safe and `queue.Queue` is; they solve
  different problems, and the reason both appear is that the legacy runtime
  used the former while the boundary EMBER replaced genuinely needs the
  latter's guarantees. Comparing only one would flatter whichever was chosen.
"""

from __future__ import annotations

import asyncio
import collections
import queue
import threading
import time
from typing import Callable

from agents.messaging.ember_wire import (
    AckFrame,
    AckResult,
    CommandFrame,
    CommandOp,
    MsgType,
    StreamFrameReader,
    TelemetryFrame,
    TelemetrySubsystem,
    pack_frame,
    unpack_frame,
)
from benchmarks.bench_core.baseline import (
    CODECS,
    LengthPrefixedReader,
    ack_message,
    command_message,
    frame_payload,
    telemetry_message,
)
from benchmarks.bench_core.resources import GcProbe
from benchmarks.bench_core.stats import (
    BenchRow,
    compute_stats,
    throughput_from_stats,
    time_batched,
)

BINARY_ARM = "Binary (EMBER wire)"
PYTHON_ARM = "Pure Python"

FAULT_TEXT = "motor_left: encoder disagreement beyond tolerance"


def _row(
    suite: str,
    name: str,
    arm: str,
    stats,
    unit: str = "ops/sec",
    notes: str = "",
) -> BenchRow:
    return BenchRow(
        suite=suite,
        name=name,
        arm=arm,
        latency=stats,
        throughput_value=throughput_from_stats(stats),
        throughput_unit=unit,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Sample messages — the same logical content in both encodings
# ---------------------------------------------------------------------------


def sample_command_frame() -> CommandFrame:
    return CommandFrame(
        sequence=424242,
        op=CommandOp.MOVE_FORWARD,
        timestamp_ns=1_700_000_000_123_456_789,
        param=1.25,
        request_id_hash=0xDEADBEEF,
    )


def sample_telemetry_frame(text: str = "") -> TelemetryFrame:
    return TelemetryFrame(
        sequence=99,
        subsystem=TelemetrySubsystem.MOTION if not text else TelemetrySubsystem.FAULT,
        timestamp_ns=1_700_000_000_987_654_321,
        value_a=0.85,
        value_b=11.4,
        code=0xE001,
        text=text,
    )


def sample_ack_frame() -> AckFrame:
    return AckFrame(ack_sequence=424242, result=AckResult.ACCEPTED, request_id_hash=0xDEADBEEF)


def sample_command_dict() -> dict:
    message = command_message(424242, CommandOp.MOVE_FORWARD, 1.25, "bench-request")
    message["timestamp_ns"] = 1_700_000_000_123_456_789  # pinned, so sizes are reproducible
    return message


def sample_telemetry_dict(text: str = "") -> dict:
    message = telemetry_message(
        99,
        TelemetrySubsystem.MOTION if not text else TelemetrySubsystem.FAULT,
        value_a=0.85,
        value_b=11.4,
        code=0xE001,
        text=text,
    )
    message["timestamp_ns"] = 1_700_000_000_987_654_321
    return message


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def serialization_rows(iterations: int, warmup: int, batch_size: int = 200) -> list[BenchRow]:
    """Encode/decode of one command and one telemetry message, in each
    encoding, measured at the framed level (checksum and all) — because a
    framed packet, not a bare struct, is what actually crosses the boundary."""
    suite = "Serialization"
    rows: list[BenchRow] = []

    command_frame = sample_command_frame()
    command_dict = sample_command_dict()
    telemetry_plain = sample_telemetry_frame()
    telemetry_text = sample_telemetry_frame(FAULT_TEXT)
    telemetry_plain_dict = sample_telemetry_dict()
    telemetry_text_dict = sample_telemetry_dict(FAULT_TEXT)
    ack_frame = sample_ack_frame()

    # --- binary (the hybrid arm's Python-side cost) ---------------------

    def binary_command_encode() -> None:
        pack_frame(int(MsgType.COMMAND), command_frame.to_bytes())

    packed_command = pack_frame(int(MsgType.COMMAND), command_frame.to_bytes())

    def binary_command_decode() -> None:
        decoded = unpack_frame(packed_command)
        if decoded is not None:
            CommandFrame.from_bytes(decoded.payload)

    rows.append(
        _row(
            suite,
            "Command encode (framed)",
            BINARY_ARM,
            time_batched(iterations, warmup, binary_command_encode, batch_size),
            notes=f"{len(packed_command)}B wire",
        )
    )
    rows.append(
        _row(
            suite,
            "Command decode (framed+verify)",
            BINARY_ARM,
            time_batched(iterations, warmup, binary_command_decode, batch_size),
            notes=f"{len(packed_command)}B wire",
        )
    )

    for label, frame in (("(no text)", telemetry_plain), ("(48B text)", telemetry_text)):
        packed = pack_frame(int(MsgType.TELEMETRY), frame.to_bytes())

        def encode(frame=frame) -> None:
            pack_frame(int(MsgType.TELEMETRY), frame.to_bytes())

        def decode(packed=packed) -> None:
            decoded = unpack_frame(packed)
            if decoded is not None:
                TelemetryFrame.from_bytes(decoded.payload)

        rows.append(
            _row(
                suite,
                f"Telemetry encode {label}",
                BINARY_ARM,
                time_batched(iterations, warmup, encode, batch_size),
                notes=f"{len(packed)}B wire",
            )
        )
        rows.append(
            _row(
                suite,
                f"Telemetry decode {label}",
                BINARY_ARM,
                time_batched(iterations, warmup, decode, batch_size),
                notes=f"{len(packed)}B wire",
            )
        )

    packed_ack = pack_frame(int(MsgType.ACK), ack_frame.to_bytes())

    def binary_ack_roundtrip() -> None:
        decoded = unpack_frame(pack_frame(int(MsgType.ACK), ack_frame.to_bytes()))
        if decoded is not None:
            AckFrame.from_bytes(decoded.payload)

    rows.append(
        _row(
            suite,
            "Ack encode+decode (framed)",
            BINARY_ARM,
            time_batched(iterations, warmup, binary_ack_roundtrip, batch_size),
            notes=f"{len(packed_ack)}B wire",
        )
    )

    # --- pure-Python codecs ---------------------------------------------

    for codec_name, codec in CODECS.items():
        encoded_command = frame_payload(codec.encode(command_dict))

        def encode_command(codec=codec) -> None:
            frame_payload(codec.encode(command_dict))

        def decode_command(codec=codec, encoded=encoded_command) -> None:
            reader = LengthPrefixedReader()
            reader.feed(encoded)
            payload = reader.try_extract()
            if payload is not None:
                codec.decode(payload)

        rows.append(
            _row(
                suite,
                "Command encode (framed)",
                f"{PYTHON_ARM} [{codec_name}]",
                time_batched(iterations, warmup, encode_command, batch_size),
                notes=f"{len(encoded_command)}B wire",
            )
        )
        rows.append(
            _row(
                suite,
                "Command decode (framed+verify)",
                f"{PYTHON_ARM} [{codec_name}]",
                time_batched(iterations, warmup, decode_command, batch_size),
                notes=f"{len(encoded_command)}B wire",
            )
        )

        for label, message in (
            ("(no text)", telemetry_plain_dict),
            ("(48B text)", telemetry_text_dict),
        ):
            encoded_telemetry = frame_payload(codec.encode(message))

            def encode_telemetry(codec=codec, message=message) -> None:
                frame_payload(codec.encode(message))

            def decode_telemetry(codec=codec, encoded=encoded_telemetry) -> None:
                reader = LengthPrefixedReader()
                reader.feed(encoded)
                payload = reader.try_extract()
                if payload is not None:
                    codec.decode(payload)

            rows.append(
                _row(
                    suite,
                    f"Telemetry encode {label}",
                    f"{PYTHON_ARM} [{codec_name}]",
                    time_batched(iterations, warmup, encode_telemetry, batch_size),
                    notes=f"{len(encoded_telemetry)}B wire",
                )
            )
            rows.append(
                _row(
                    suite,
                    f"Telemetry decode {label}",
                    f"{PYTHON_ARM} [{codec_name}]",
                    time_batched(iterations, warmup, decode_telemetry, batch_size),
                    notes=f"{len(encoded_telemetry)}B wire",
                )
            )

    return rows


def stream_reassembly_rows(iterations: int, warmup: int, batch_size: int = 50) -> list[BenchRow]:
    """Byte-stream reassembly, whole-frame and one-byte-at-a-time.

    The one-byte case is not a strawman: it is what a small MTU, a Nagle
    interaction or a congested link produces, and it is the case where the
    magic-scan-and-resync design of `StreamFrameReader` costs the most
    relative to a bare length prefix.
    """
    suite = "Stream Reassembly"
    rows: list[BenchRow] = []

    packed = pack_frame(int(MsgType.COMMAND), sample_command_frame().to_bytes())
    legacy_packed = frame_payload(CODECS["json"].encode(sample_command_dict()))

    def run_binary(chunk: int) -> Callable[[], None]:
        def inner() -> None:
            reader = StreamFrameReader()
            for offset in range(0, len(packed), chunk):
                reader.feed(packed[offset : offset + chunk])
                while reader.try_extract() is not None:
                    pass

        return inner

    def run_legacy(chunk: int) -> Callable[[], None]:
        def inner() -> None:
            reader = LengthPrefixedReader()
            for offset in range(0, len(legacy_packed), chunk):
                reader.feed(legacy_packed[offset : offset + chunk])
                while reader.try_extract() is not None:
                    pass

        return inner

    for chunk, label in ((4096, "whole frame per feed"), (1, "1 byte per feed")):
        rows.append(
            _row(
                suite,
                f"Reassemble {label}",
                BINARY_ARM,
                time_batched(iterations, warmup, run_binary(chunk), batch_size),
                unit="frames/sec",
                notes=f"{len(packed)}B frame, magic+checksum resync",
            )
        )
        rows.append(
            _row(
                suite,
                f"Reassemble {label}",
                f"{PYTHON_ARM} [json]",
                time_batched(iterations, warmup, run_legacy(chunk), batch_size),
                unit="frames/sec",
                notes=f"{len(legacy_packed)}B frame, length prefix only (no resync)",
            )
        )

    return rows


def wire_size_table() -> dict[str, str]:
    """Bytes on the wire per message type, per encoding.

    Size is not a latency figure, but on a real link it becomes one, and it is
    the one dimension where the binary protocol's advantage is exact rather
    than statistical.
    """
    sizes: dict[str, str] = {}
    command_frame = sample_command_frame()
    command_dict = sample_command_dict()

    sizes["Command / binary"] = f"{len(pack_frame(int(MsgType.COMMAND), command_frame.to_bytes()))} B"
    for name, codec in CODECS.items():
        sizes[f"Command / {name}"] = f"{len(frame_payload(codec.encode(command_dict)))} B"

    for label, text in (("Telemetry (no text)", ""), ("Telemetry (48B text)", FAULT_TEXT)):
        frame = sample_telemetry_frame(text)
        message = sample_telemetry_dict(text)
        sizes[f"{label} / binary"] = (
            f"{len(pack_frame(int(MsgType.TELEMETRY), frame.to_bytes()))} B"
        )
        for name, codec in CODECS.items():
            sizes[f"{label} / {name}"] = f"{len(frame_payload(codec.encode(message)))} B"

    ack = sample_ack_frame()
    sizes["Ack / binary"] = f"{len(pack_frame(int(MsgType.ACK), ack.to_bytes()))} B"
    for name, codec in CODECS.items():
        message = ack_message(424242, AckResult.ACCEPTED, 0xDEADBEEF)
        sizes[f"Ack / {name}"] = f"{len(frame_payload(codec.encode(message)))} B"

    return sizes


# ---------------------------------------------------------------------------
# Event handling / queues
# ---------------------------------------------------------------------------


def _run_on_loop(coro_factory, loop: asyncio.AbstractEventLoop):
    return loop.run_until_complete(coro_factory())


def asyncio_queue_rows(iterations: int, warmup: int, batch_size: int = 200) -> list[BenchRow]:
    """`asyncio.Queue` put/get round trip, measured inside the event loop.

    Timed in-loop rather than via `run_coroutine_threadsafe` on purpose: the
    latter would measure the thread handoff, which is a separate benchmark
    below. This one isolates the queue.
    """
    suite = "Event Handling"
    rows: list[BenchRow] = []

    loop = asyncio.new_event_loop()
    try:
        item = sample_command_dict()

        async def batch_roundtrip(count: int) -> None:
            q: asyncio.Queue = asyncio.Queue()
            for _ in range(count):
                q.put_nowait(item)
                await q.get()

        # Warm-up runs first so the loop's own lazy setup is out of the way.
        loop.run_until_complete(batch_roundtrip(warmup))

        samples: list[float] = []
        batches = max(1, iterations // batch_size)
        for _ in range(batches):
            start = time.perf_counter_ns()
            loop.run_until_complete(batch_roundtrip(batch_size))
            samples.append((time.perf_counter_ns() - start) / batch_size)

        rows.append(
            _row(
                suite,
                "asyncio.Queue put+get",
                PYTHON_ARM,
                compute_stats(samples),
                unit="msgs/sec",
                notes="single-threaded, in-loop",
            )
        )
    finally:
        loop.close()

    return rows


def thread_queue_rows(iterations: int, warmup: int, batch_size: int = 200) -> list[BenchRow]:
    """`queue.Queue` and `collections.deque`, the thread-safe Python options.

    `queue.Queue` is the closest stdlib analogue to EMBER's `ThreadSafeQueue`
    (mutex + condition variable, blocking pop), so it is the fair Python
    comparator for the C++ queue rows. The bare `deque` is included as a lower
    bound: it is what the cost would be with no synchronization at all, which
    frames how much of `queue.Queue`'s cost is the safety rather than the
    data structure.
    """
    suite = "Event Handling"
    rows: list[BenchRow] = []
    item = sample_command_dict()

    def thread_queue_roundtrip() -> None:
        q: queue.Queue = queue.Queue()
        q.put(item)
        q.get()

    rows.append(
        _row(
            suite,
            "queue.Queue put+get",
            PYTHON_ARM,
            time_batched(iterations, warmup, thread_queue_roundtrip, batch_size),
            unit="msgs/sec",
            notes="single-threaded, uncontended",
        )
    )

    def deque_roundtrip() -> None:
        d: collections.deque = collections.deque()
        d.append(item)
        d.popleft()

    rows.append(
        _row(
            suite,
            "collections.deque append+popleft",
            PYTHON_ARM,
            time_batched(iterations, warmup, deque_roundtrip, batch_size),
            unit="msgs/sec",
            notes="no synchronization (lower bound)",
        )
    )

    return rows


def queue_contention_rows(
    producer_counts: tuple[int, ...] = (1, 2, 4, 8),
    per_producer: int = 20000,
) -> list[BenchRow]:
    """`queue.Queue` push latency at N producers and one consumer.

    The direct counterpart of `bench_queue_contention` in bench_bridge.cpp:
    same shape, same producer counts, same per-producer count, so the P99
    inflation from 1 to 8 producers is comparable across languages. Under the
    GIL the shape of the answer is different in kind, not just in degree, and
    seeing both is the point.
    """
    suite = "Lock Contention"
    rows: list[BenchRow] = []
    payload = sample_command_dict()

    for producers in producer_counts:
        q: queue.Queue = queue.Queue()
        consumed = 0
        stop_sentinel = object()

        def consume() -> None:
            nonlocal consumed
            while True:
                item = q.get()
                if item is stop_sentinel:
                    return
                consumed += 1

        consumer = threading.Thread(target=consume, name="bench-consumer", daemon=True)
        consumer.start()

        per_thread: list[list[float]] = [[] for _ in range(producers)]
        ready = threading.Barrier(producers + 1)

        def produce(index: int) -> None:
            samples = per_thread[index]
            clock = time.perf_counter_ns
            ready.wait()
            for _ in range(per_producer):
                start = clock()
                q.put(payload)
                samples.append(clock() - start)

        threads = [
            threading.Thread(target=produce, args=(i,), name=f"bench-producer-{i}", daemon=True)
            for i in range(producers)
        ]
        for thread in threads:
            thread.start()

        ready.wait()
        wall_start = time.perf_counter_ns()
        for thread in threads:
            thread.join()
        wall_ns = time.perf_counter_ns() - wall_start

        q.put(stop_sentinel)
        consumer.join(timeout=5.0)

        all_samples = [s for samples in per_thread for s in samples]
        stats = compute_stats(all_samples)
        total_ops = producers * per_producer
        rows.append(
            BenchRow(
                suite=suite,
                name=f"queue.Queue put {producers}P/1C",
                arm=PYTHON_ARM,
                latency=stats,
                throughput_value=total_ops / (wall_ns / 1e9) if wall_ns else 0.0,
                throughput_unit="msgs/sec",
                notes=f"{total_ops:,} pushes, {consumed:,} consumed",
            )
        )

    return rows


def cross_thread_handoff_rows(iterations: int, warmup: int) -> list[BenchRow]:
    """Producer-thread to consumer-thread wake-up latency.

    This is the number that actually governs how quickly a telemetry frame
    arriving on a network thread reaches the code that reacts to it, and it is
    dominated by thread wake-up rather than by the queue's data structure —
    which is why it is measured separately from the uncontended put/get rows
    above. Both mechanisms the legacy runtime could use are covered:
    `queue.Queue` (blocking pop) and `asyncio.Queue` fed by
    `call_soon_threadsafe`, which is how a real asyncio design gets data in
    from a non-loop thread.
    """
    suite = "Event Handling"
    rows: list[BenchRow] = []

    # --- queue.Queue blocking handoff ---------------------------------
    q: queue.Queue = queue.Queue()
    received: list[float] = []
    stop = object()

    def consume_blocking() -> None:
        while True:
            item = q.get()
            if item is stop:
                return
            received.append(time.perf_counter_ns() - item)

    consumer = threading.Thread(target=consume_blocking, daemon=True)
    consumer.start()
    for _ in range(warmup):
        q.put(time.perf_counter_ns())
    time.sleep(0.05)
    received.clear()
    for _ in range(iterations):
        q.put(time.perf_counter_ns())
        # A brief pause between sends keeps the consumer genuinely asleep
        # between items. Without it the consumer stays hot in `get()` and the
        # measurement reports queue throughput rather than wake-up latency.
        time.sleep(0.0002)
    time.sleep(0.1)
    q.put(stop)
    consumer.join(timeout=5.0)

    rows.append(
        _row(
            suite,
            "Cross-thread handoff (queue.Queue)",
            PYTHON_ARM,
            compute_stats(received),
            unit="handoffs/sec",
            notes=f"{len(received)} samples, consumer parked between items",
        )
    )

    # --- asyncio.Queue fed via call_soon_threadsafe --------------------
    loop = asyncio.new_event_loop()
    async_received: list[float] = []
    loop_ready = threading.Event()

    def run_loop() -> None:
        asyncio.set_event_loop(loop)
        loop.call_soon(loop_ready.set)
        loop.run_forever()

    loop_thread = threading.Thread(target=run_loop, daemon=True)
    loop_thread.start()
    loop_ready.wait(5.0)

    async_queue: asyncio.Queue = asyncio.Queue()

    async def consume_async() -> None:
        while True:
            sent_ns = await async_queue.get()
            if sent_ns is None:
                return
            async_received.append(time.perf_counter_ns() - sent_ns)

    consume_task = asyncio.run_coroutine_threadsafe(consume_async(), loop)

    for _ in range(warmup):
        loop.call_soon_threadsafe(async_queue.put_nowait, time.perf_counter_ns())
    time.sleep(0.05)
    async_received.clear()
    for _ in range(iterations):
        loop.call_soon_threadsafe(async_queue.put_nowait, time.perf_counter_ns())
        time.sleep(0.0002)
    time.sleep(0.1)
    loop.call_soon_threadsafe(async_queue.put_nowait, None)
    try:
        consume_task.result(timeout=5.0)
    except Exception:
        pass
    loop.call_soon_threadsafe(loop.stop)
    loop_thread.join(timeout=5.0)
    loop.close()

    rows.append(
        _row(
            suite,
            "Cross-thread handoff (asyncio.Queue)",
            PYTHON_ARM,
            compute_stats(async_received),
            unit="handoffs/sec",
            notes=f"{len(async_received)} samples, via call_soon_threadsafe",
        )
    )

    return rows


def gc_pressure_summary(sample_count: int = 40000, buffer_depth: int = 256) -> dict[str, str]:
    """Memory and garbage-collection cost of each encoding.

    Two figures per codec, because they answer different questions.

    **Retained bytes per buffered message** is measured with `tracemalloc`
    while `buffer_depth` encoded messages are held alive at once — deliberately
    the same 256 that `BridgeConfig::max_buffered_telemetry_frames` allows, so
    the number reads directly as "what a full outbound buffer costs in this
    encoding". This is the figure that differentiates the codecs.

    **Collections per 10k round trips** is measured with the same
    `gc.callbacks` probe the resource suite uses. On every codec here it comes
    out at zero, and that is the finding rather than a broken measurement:
    CPython reclaims these objects by reference counting the moment they go out
    of scope, so the generational counter — which is incremented on allocation
    and decremented on deallocation — never climbs toward its threshold. Codec
    churn does not drive collection; only objects that *survive* do. An earlier
    version of this function reported `len(gc.get_objects())` deltas, which was
    zero for a subtler and less interesting reason (encoder output is `bytes`,
    which the collector does not track at all).
    """
    import gc
    import tracemalloc

    summary: dict[str, str] = {}
    command_frame = sample_command_frame()
    command_dict = sample_command_dict()

    def retained_bytes(fn: Callable[[], object]) -> float:
        gc.collect()
        tracemalloc.start()
        baseline = tracemalloc.get_traced_memory()[0]
        held = [fn() for _ in range(buffer_depth)]
        current = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        # `held` is read after the measurement so the list cannot be optimized
        # away and its contents cannot be collected before `current` is taken.
        assert len(held) == buffer_depth
        return (current - baseline) / buffer_depth

    def collections_per_10k(fn: Callable[[], object]) -> tuple[float, float]:
        gc.collect()
        with GcProbe() as probe:
            for _ in range(sample_count):
                fn()
        report = probe.report
        scale = 10000 / sample_count
        return report.total_collections * scale, report.total_pause_ns / 1e6 * scale

    def describe(label: str, encode: Callable[[], object], roundtrip: Callable[[], object]) -> None:
        per_message = retained_bytes(encode)
        collections, pause_ms = collections_per_10k(roundtrip)
        summary[label] = (
            f"{per_message:,.0f} B retained/msg "
            f"({per_message * buffer_depth / 1024:,.1f} KiB for a {buffer_depth}-frame buffer), "
            f"{collections:.1f} collections/10k round trips, {pause_ms:.3f} ms GC pause/10k"
        )

    packed_binary = pack_frame(int(MsgType.COMMAND), command_frame.to_bytes())

    def binary_roundtrip() -> None:
        pack_frame(int(MsgType.COMMAND), command_frame.to_bytes())
        decoded = unpack_frame(packed_binary)
        if decoded is not None:
            CommandFrame.from_bytes(decoded.payload)

    describe(
        "Command / binary",
        lambda: pack_frame(int(MsgType.COMMAND), command_frame.to_bytes()),
        binary_roundtrip,
    )

    for name, codec in CODECS.items():
        encoded = frame_payload(codec.encode(command_dict))

        def roundtrip(codec=codec, encoded=encoded) -> None:
            frame_payload(codec.encode(command_dict))
            reader = LengthPrefixedReader()
            reader.feed(encoded)
            payload = reader.try_extract()
            if payload is not None:
                codec.decode(payload)

        describe(
            f"Command / {name}",
            lambda codec=codec: frame_payload(codec.encode(command_dict)),
            roundtrip,
        )

    return summary
