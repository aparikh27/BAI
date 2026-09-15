"""Integration tests for the hybrid AgentCore + EMBER runtime.

Two concerns, in two halves:

**Payload integrity across the IPC boundary.** Unit tests on each side prove
that each half is self-consistent; neither can prove the two halves agree.
These tests drive the real compiled EMBER runtime over a real loopback socket
and check the bytes, including a byte-for-byte comparison of what each
language's encoder produces for the same logical frame. ADR-005 records that
this cross-check was performed once, by hand, with a throwaway program; here
it is a permanent test, because a field-offset disagreement introduced later
would otherwise be invisible until a robot moved the wrong distance.

**Failure recovery.** A bridge between a deterministic C++ core and a Python
process doing multi-second inference exists precisely because things go wrong,
so the failure paths deserve at least as much coverage as the happy one:
corrupt checksums, truncated and fragmented frames, garbage on the wire,
telemetry overflow, and a peer that disappears without warning. Each test
asserts both halves of recovery — that the bad input was rejected, *and* that
the connection still works afterwards. Rejecting a corrupt frame by wedging
the link is not recovery.
"""

from __future__ import annotations

import socket
import struct
import threading
import time

import pytest

from agents.messaging import MessageStatus
from agents.messaging.ember_wire import (
    AckFrame,
    AckResult,
    CommandFrame,
    CommandOp,
    MsgType,
    StreamFrameReader,
    TelemetryFrame,
    TelemetrySubsystem,
    fletcher16,
    fnv1a_32,
    pack_frame,
    unpack_frame,
)
from benchmarks.bench_core.workload import run_perception_to_execution

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _connect(port: int, timeout: float = 5.0) -> socket.socket:
    """A raw client socket to the EMBER harness.

    Several tests need to put bytes on the wire that `EmberBridgeClient` would
    never produce — a corrupt checksum, a truncated frame, pure garbage — so
    they bypass the client entirely rather than adding fault-injection hooks
    to production code that would then exist only for tests.
    """
    sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    sock.settimeout(timeout)
    return sock


def _drain(sock: socket.socket, duration: float = 0.4) -> list:
    """Collects every complete frame arriving within `duration`.

    Time-bounded rather than count-bounded on purpose: the assertions that
    matter are usually about a frame *not* arriving, and a read that blocks
    until N frames appear cannot express that.
    """
    reader = StreamFrameReader()
    frames = []
    deadline = time.monotonic() + duration
    sock.settimeout(0.1)
    while time.monotonic() < deadline:
        try:
            chunk = sock.recv(4096)
        except socket.timeout:
            continue
        except OSError:
            break
        if not chunk:
            break
        reader.feed(chunk)
        while (frame := reader.try_extract()) is not None:
            frames.append(frame)
    return frames


def _drain_until(sock: socket.socket, predicate, timeout: float = 5.0):
    """Reads frames until `predicate(ack)` matches one, returning
    (matched_ack_or_None, seconds_elapsed).

    Distinct from `_drain`, which always consumes its whole window: when the
    question is *how long* something took, a fixed-duration read reports the
    window rather than the latency.
    """
    reader = StreamFrameReader()
    start = time.perf_counter()
    deadline = start + timeout
    sock.settimeout(0.1)
    while time.perf_counter() < deadline:
        try:
            chunk = sock.recv(65536)
        except socket.timeout:
            continue
        except OSError:
            break
        if not chunk:
            break
        reader.feed(chunk)
        while (frame := reader.try_extract()) is not None:
            if frame.msg_type != int(MsgType.ACK):
                continue
            ack = AckFrame.from_bytes(frame.payload)
            if ack is not None and predicate(ack):
                return ack, time.perf_counter() - start
    return None, time.perf_counter() - start


def _acks(frames) -> list[AckFrame]:
    decoded = [AckFrame.from_bytes(f.payload) for f in frames if f.msg_type == int(MsgType.ACK)]
    return [ack for ack in decoded if ack is not None]


def _command_packet(sequence: int, op: CommandOp, param: float = 0.0, request_id: str = "") -> bytes:
    frame = CommandFrame(
        sequence=sequence,
        op=op,
        timestamp_ns=1_700_000_000_000_000_000,
        param=param,
        request_id_hash=fnv1a_32(request_id) if request_id else 0,
    )
    return pack_frame(int(MsgType.COMMAND), frame.to_bytes())


def _corrupt_checksum(packet: bytes) -> bytes:
    """Flips one bit in the checksum trailer, leaving the payload valid.

    Corrupting the checksum rather than the payload isolates what is being
    tested: the receiver must reject the frame on integrity grounds alone,
    with a payload that would otherwise decode perfectly well.
    """
    body, trailer = packet[:-2], packet[-2:]
    (checksum,) = struct.unpack(">H", trailer)
    return body + struct.pack(">H", checksum ^ 0x0001)


def _corrupt_payload(packet: bytes) -> bytes:
    """Flips a bit inside the payload, leaving the checksum trailer alone —
    the shape a real single-bit line error takes."""
    data = bytearray(packet)
    data[8] ^= 0x40
    return bytes(data)


# ===========================================================================
# Part 1: payload integrity across the Python/C++ boundary
# ===========================================================================


class TestWireFormatParity:
    """Byte-for-byte agreement between `agents/messaging/ember_wire.py` and
    `edge/bridge/bridge_frames.hpp` + `edge/serialization/serializer.hpp`."""

    def test_python_and_cpp_encode_telemetry_identically(self, ember_server):
        cases = [
            (TelemetrySubsystem.MOTION, 1, 1_700_000_000_123_456_789, 0.85, 0.0, 0, ""),
            (TelemetrySubsystem.BATTERY, 7, 42, 18.0, 11.1, 0, ""),
            (TelemetrySubsystem.FAULT, 65535, 0, 0.0, 0.0, 57345, "motor_left:overheat"),
            (TelemetrySubsystem.THERMAL, 2**31, 2**40, -273.15, 1e-8, 999, "cpu"),
        ]

        for subsystem, sequence, timestamp_ns, value_a, value_b, code, text in cases:
            cpp_bytes = ember_server.pack_telemetry(
                int(subsystem), sequence, timestamp_ns, value_a, value_b, code, text
            )
            assert cpp_bytes is not None, "EMBER did not answer PACK_TELEMETRY"

            python_bytes = pack_frame(
                int(MsgType.TELEMETRY),
                TelemetryFrame(
                    sequence=sequence,
                    subsystem=subsystem,
                    timestamp_ns=timestamp_ns,
                    value_a=value_a,
                    value_b=value_b,
                    code=code,
                    text=text,
                ).to_bytes(),
            )
            assert python_bytes == cpp_bytes, (
                f"encoder disagreement for {subsystem.name}: "
                f"python={python_bytes.hex()} cpp={cpp_bytes.hex()}"
            )

            # And Python can read back what C++ produced, with the float
            # values surviving the big-endian round trip exactly (they are
            # IEEE-754 singles on both sides, so equality is the right
            # assertion here, not approximate comparison).
            decoded = unpack_frame(cpp_bytes)
            assert decoded is not None
            frame = TelemetryFrame.from_bytes(decoded.payload)
            assert frame is not None
            assert frame.sequence == sequence
            assert frame.timestamp_ns == timestamp_ns
            assert frame.code == code
            assert frame.text == text
            assert struct.pack(">f", frame.value_a) == struct.pack(">f", value_a)
            assert struct.pack(">f", frame.value_b) == struct.pack(">f", value_b)

    def test_python_and_cpp_encode_acks_identically(self, ember_server):
        for sequence, result, request_hash in (
            (0, AckResult.ACCEPTED, 0),
            (1, AckResult.COMPLETED, 0xDEADBEEF),
            (2**32 - 1, AckResult.FAULTED, 2**32 - 1),
        ):
            cpp_bytes = ember_server.pack_ack(sequence, int(result), request_hash)
            assert cpp_bytes is not None
            python_bytes = pack_frame(
                int(MsgType.ACK),
                AckFrame(
                    ack_sequence=sequence, result=result, request_id_hash=request_hash
                ).to_bytes(),
            )
            assert python_bytes == cpp_bytes

    def test_cpp_decodes_python_encoded_commands_field_for_field(self, ember_server):
        """The other direction, and the one that would catch a field-offset
        slip: EMBER reports the values it decoded, so agreement is checked on
        the fields rather than merely on "it didn't crash"."""
        cases = [
            (1, CommandOp.MOVE_FORWARD, 1.25, "req-a"),
            (2**32 - 1, CommandOp.TURN, -180.0, "req-b"),
            (12345, CommandOp.STOP, 0.0, ""),
            (7, CommandOp.GRAB_ITEM, 3.4028234663852886e38, "req-max-float"),
        ]

        for sequence, op, param, request_id in cases:
            request_hash = fnv1a_32(request_id) if request_id else 0
            packet = _command_packet(sequence, op, param, request_id)
            fields = ember_server.unpack_command(packet)
            assert fields is not None, f"EMBER rejected a valid Python-encoded {op.name}"

            assert int(fields["msg_type"]) == int(MsgType.COMMAND)
            assert int(fields["sequence"]) == sequence
            assert int(fields["timestamp_ns"]) == 1_700_000_000_000_000_000
            assert int(fields["op"]) == int(op)
            assert int(fields["request_id_hash"]) == request_hash
            # param is a float32 on the wire; compare through the same
            # narrowing so the assertion isn't testing float64 formatting.
            assert struct.pack(">f", fields["param"]) == struct.pack(">f", param)

    def test_cpp_rejects_frames_python_would_also_reject(self, ember_server):
        """Both sides must agree on what is *invalid*, not just on what is
        valid — a receiver that accepts frames the sender considers malformed
        is exactly how a corrupt command reaches an actuator."""
        valid = _command_packet(1, CommandOp.MOVE_FORWARD, 1.0)

        for label, bad in (
            ("corrupt checksum", _corrupt_checksum(valid)),
            ("corrupt payload", _corrupt_payload(valid)),
            ("bad magic", b"XX" + valid[2:]),
            ("truncated", valid[:-3]),
            ("wrong payload length for a CommandFrame", pack_frame(int(MsgType.COMMAND), b"\x00" * 10)),
        ):
            assert unpack_frame(bad) is None or CommandFrame.from_bytes(
                unpack_frame(bad).payload  # type: ignore[union-attr]
            ) is None, f"Python accepted {label}"
            assert ember_server.unpack_command(bad) is None, f"EMBER accepted {label}"

    def test_fletcher16_matches_across_languages_via_real_frames(self, ember_server):
        """`fletcher16` has no direct control command, but every PACKED reply
        carries EMBER's checksum in its trailer — so recomputing it in Python
        over the same region checks the algorithm itself, including the
        modulo-255 wraparound that a naive implementation gets wrong only for
        long payloads."""
        long_text = "".join(chr(0x41 + (i % 26)) for i in range(600))
        cpp_bytes = ember_server.pack_telemetry(
            int(TelemetrySubsystem.FAULT), 1, 2, 0.0, 0.0, 1, long_text
        )
        assert cpp_bytes is not None

        body, trailer = cpp_bytes[:-2], cpp_bytes[-2:]
        (cpp_checksum,) = struct.unpack(">H", trailer)
        assert fletcher16(body) == cpp_checksum
        assert len(body) > 255, "payload must exceed 255 bytes to exercise the modulo wraparound"


class TestPayloadIntegrityUnderLoad:
    """Integrity at volume through the real client, not just for a handful of
    hand-picked frames."""

    def test_every_command_is_acked_exactly_once_with_its_own_sequence(self, ember_server):
        sock = _connect(ember_server.port)
        try:
            count = 200
            for sequence in range(1, count + 1):
                sock.sendall(_command_packet(sequence, CommandOp.MOVE_FORWARD, sequence / 10.0))

            acks = _acks(_drain(sock, duration=3.0))
            accepted = [a.ack_sequence for a in acks if a.result == AckResult.ACCEPTED]

            assert sorted(accepted) == list(range(1, count + 1))
            assert len(accepted) == len(set(accepted)), "a sequence was acked more than once"
        finally:
            sock.close()

    def test_param_survives_the_round_trip_through_the_real_driver(self, hybrid_rig):
        """`move_forward(distance)` must arrive at the C++ side as the same
        float32. This goes through the whole stack — driver, client, socket,
        adapter, Coordinator, subsystem worker — and reads the value back off
        EMBER's own log line."""
        distances = [0.0, 0.5, 1.25, 12.75, 1000.5]
        mark = hybrid_rig.server.line_count()

        for distance in distances:
            hybrid_rig.driver.move_forward(distance)

        line = hybrid_rig.server.wait_for_line(
            lambda text: text.startswith("CMD_RECEIVED") and "param=1000.500000" in text,
            timeout=5.0,
            since=mark,
        )
        assert line is not None, "EMBER never logged the final move_forward"

        logged = [
            float(part.split("=", 1)[1])
            for text in hybrid_rig.server.all_lines()[mark:]
            if text.startswith("CMD_RECEIVED")
            for part in text.split()
            if part.startswith("param=")
        ]
        for expected in distances:
            assert any(
                struct.pack(">f", value) == struct.pack(">f", expected) for value in logged
            ), f"distance {expected} never reached EMBER intact (saw {logged})"

    def test_request_id_hash_reaches_ember_and_matches_python(self, ember_server):
        """The correlation key ADR-005 uses to tie a command back to an
        AgentCore request. A hash that disagreed across the boundary would
        break log correlation silently — nothing would fail, the traces would
        just stop lining up."""
        request_id = "b3f1c9a2-4e7d-4a11-9f2c-5d6e7a8b9c01"
        packet = _command_packet(9, CommandOp.TURN, 45.0, request_id)
        fields = ember_server.unpack_command(packet)
        assert fields is not None
        assert int(fields["request_id_hash"]) == fnv1a_32(request_id)

    def test_full_pipeline_reaches_ember_with_every_motor_command(self, hybrid_rig):
        """The complete Perception -> Execution path, asserted against what
        the C++ side logged rather than only against the Message that came
        back — a pipeline can return SUCCESS while the commands went nowhere."""
        hybrid_rig.start_telemetry_stream(100.0)
        try:
            assert hybrid_rig.wait_for_telemetry(timeout_s=5.0), "no Motion telemetry from EMBER"
            mark = hybrid_rig.server.line_count()

            result = run_perception_to_execution(hybrid_rig.coordinator, frame_index=0)
            assert result.status == MessageStatus.SUCCESS, result.error
            assert result.payload["total_steps"] == 1

            # The ack that unblocked Python is sent on receipt, before the
            # subsystem worker prints its line, so the last CMD_RECEIVED can
            # still be in flight when the pipeline returns.
            hybrid_rig.server.wait_for_line(
                lambda text: text.startswith("CMD_COMPLETED"), timeout=3.0, since=mark
            )
            time.sleep(0.3)
            received = [
                text
                for text in hybrid_rig.server.all_lines()[mark:]
                if text.startswith("CMD_RECEIVED")
            ]

            topics = [
                part.split("=", 1)[1]
                for text in received
                for part in text.split()
                if part.startswith("topic=")
            ]
            # _get_object issues: stop, move_forward, lower_arm, grab_item,
            # raise_arm, turn, move_forward, turn.
            assert topics.count("cmd/motion/move_forward") == 2
            assert topics.count("cmd/motion/turn") == 2
            assert topics.count("cmd/motion/stop") == 1
            assert topics.count("cmd/manipulator/lower_arm") == 1
            assert topics.count("cmd/manipulator/grab_item") == 1
            assert topics.count("cmd/manipulator/raise_arm") == 1
        finally:
            hybrid_rig.stop_telemetry_stream()

    def test_telemetry_values_arrive_intact_at_the_driver(self, hybrid_rig):
        hybrid_rig.start_telemetry_stream(200.0)
        try:
            assert hybrid_rig.wait_for_telemetry(timeout_s=5.0)
            distance = hybrid_rig.driver.get_distance_to_front()
            # BENCH_DISTANCE_M, as a float32 round trip.
            assert struct.pack(">f", distance) == struct.pack(">f", 1.25)
        finally:
            hybrid_rig.stop_telemetry_stream()


# ===========================================================================
# Part 2: error injection and failure recovery
# ===========================================================================


class TestCorruptedFrameRecovery:
    def test_corrupt_checksum_is_dropped_and_the_link_recovers(self, ember_server):
        """The core resync guarantee: one bad frame must not be actuated and
        must not wedge the stream."""
        sock = _connect(ember_server.port)
        try:
            before = ember_server.stats()

            sock.sendall(_corrupt_checksum(_command_packet(1, CommandOp.MOVE_FORWARD, 1.0)))
            time.sleep(0.3)
            mid = ember_server.stats()
            assert mid["commands_received"] == before["commands_received"], (
                "EMBER accepted a frame with a bad checksum"
            )

            # The connection must still carry a good frame afterwards.
            sock.sendall(_command_packet(2, CommandOp.MOVE_FORWARD, 2.0))
            acks = _acks(_drain(sock, duration=2.0))
            assert any(a.ack_sequence == 2 for a in acks), "link did not recover after a bad frame"

            after = ember_server.stats()
            assert after["commands_received"] == before["commands_received"] + 1
            assert after["commands_routed"] == before["commands_routed"] + 1
        finally:
            sock.close()

    def test_garbage_prefix_is_resynced_past(self, ember_server):
        """Random bytes ahead of a valid frame — what a partial write from a
        crashed peer, or a reconnect landing mid-frame, actually looks like."""
        sock = _connect(ember_server.port)
        try:
            before = ember_server.stats()

            # "EM" appears inside the garbage, and the two bytes after it parse
            # as a length far larger than anything that follows - so the framer
            # correctly waits for a frame that will never complete, and only the
            # reader loop's stall detection can break the deadlock. Recovery
            # therefore takes longer than the reassembly stall timeout (500ms),
            # not just a round trip.
            sock.sendall(b"\x00\xff\x13\x37EM\x99garbage" + _command_packet(5, CommandOp.STOP))
            acks = _acks(_drain(sock, duration=4.0))

            assert any(a.ack_sequence == 5 for a in acks), "EMBER failed to resync past garbage"
            after = ember_server.stats()
            assert after["commands_received"] == before["commands_received"] + 1
            assert after["reassembly_resyncs"] > before["reassembly_resyncs"], (
                "the frame was recovered without the stall detector firing; this test "
                "is no longer exercising the path it claims to"
            )
        finally:
            sock.close()

    def test_bogus_length_header_does_not_stall_the_connection(self, ember_server):
        """A frame claiming a 65535-byte payload that never arrives.

        The failure this guards against is a stall, not a crash: a reader that
        waits for a length the sender lied about waits forever, and the
        connection is dead without anything reporting an error. Recovery here
        depends on the reader resyncing byte-by-byte rather than trusting the
        header.
        """
        sock = _connect(ember_server.port)
        try:
            before = ember_server.stats()
            sock.sendall(b"EM" + struct.pack(">BH", int(MsgType.COMMAND), 0xFFFF) + b"\x01\x02\x03")
            time.sleep(0.2)
            sock.sendall(_command_packet(11, CommandOp.RAISE_ARM))

            # The claimed 65535-byte payload is legal on this wire, so nothing
            # about the header can be rejected on inspection - the connection
            # recovers only because the reader loop notices it has stopped
            # making progress and resyncs. That takes longer than the 500ms
            # stall timeout.
            acks = _acks(_drain(sock, duration=5.0))
            assert any(a.ack_sequence == 11 for a in acks), (
                "a bogus length header stalled the connection"
            )
            after = ember_server.stats()
            assert after["commands_received"] >= before["commands_received"] + 1
            assert after["reassembly_resyncs"] > before["reassembly_resyncs"]
        finally:
            sock.close()

    def test_fragmented_frame_is_reassembled_exactly_once(self, ember_server):
        """One frame delivered one byte at a time, with a pause between each.

        Every intermediate state is a partial frame, so a reader that acted on
        a prefix would emit a phantom command; a reader that lost track would
        emit none. Exactly one is the only correct answer.
        """
        sock = _connect(ember_server.port)
        try:
            before = ember_server.stats()
            packet = _command_packet(21, CommandOp.TURN, 90.0)
            for byte in packet:
                sock.sendall(bytes([byte]))
                time.sleep(0.001)

            acks = _acks(_drain(sock, duration=2.0))
            # Two acks per command are expected by design - Accepted on receipt
            # and Completed once the subsystem worker finishes - so the
            # exactly-once property is asserted on the Accepted ack, which is
            # the one that corresponds to the frame being decoded.
            accepted = [
                a for a in acks if a.ack_sequence == 21 and a.result == AckResult.ACCEPTED
            ]
            assert len(accepted) == 1, (
                f"expected exactly one Accepted ack for the fragmented frame, got {acks}"
            )
            assert ember_server.stats()["commands_received"] == before["commands_received"] + 1
        finally:
            sock.close()

    def test_two_frames_in_one_write_are_both_delivered(self, ember_server):
        """The opposite packing hazard: TCP coalescing two frames into a
        single recv(), where a reader that handles one frame per read would
        silently drop the second."""
        sock = _connect(ember_server.port)
        try:
            sock.sendall(
                _command_packet(31, CommandOp.LOWER_ARM) + _command_packet(32, CommandOp.GRAB_ITEM)
            )
            acks = _acks(_drain(sock, duration=2.0))
            sequences = {a.ack_sequence for a in acks}
            assert {31, 32} <= sequences
        finally:
            sock.close()

    def test_valid_frame_with_an_undecodable_payload_is_counted_separately(self, ember_server):
        """A frame whose checksum is fine but whose payload is the wrong size
        for a CommandFrame. It is a peer bug, not a corrupted link, and the
        adapter's counters distinguish the two — a distinction that only earns
        its keep if something checks it."""
        sock = _connect(ember_server.port)
        try:
            before = ember_server.stats()
            sock.sendall(pack_frame(int(MsgType.COMMAND), b"\x00" * 7))
            time.sleep(0.3)
            after = ember_server.stats()

            assert after["commands_malformed"] == before["commands_malformed"] + 1
            assert after["commands_received"] == before["commands_received"]
            assert after["commands_routed"] == before["commands_routed"]

            sock.sendall(_command_packet(41, CommandOp.STOP))
            assert any(a.ack_sequence == 41 for a in _acks(_drain(sock, duration=2.0)))
        finally:
            sock.close()

    def test_python_framer_resyncs_past_a_corrupt_frame(self):
        """The Python twin of the same guarantee, tested without a subprocess
        so it runs on a machine with no compiler."""
        good_one = _command_packet(1, CommandOp.MOVE_FORWARD, 1.0)
        good_two = _command_packet(2, CommandOp.TURN, 90.0)

        reader = StreamFrameReader()
        reader.feed(_corrupt_checksum(good_one) + good_two)

        frames = []
        while (frame := reader.try_extract()) is not None:
            frames.append(frame)

        assert len(frames) == 1, "expected the corrupt frame to be dropped and the good one kept"
        recovered = CommandFrame.from_bytes(frames[0].payload)
        assert recovered is not None and recovered.sequence == 2


class TestQueueOverflowAndBackpressure:
    def test_telemetry_overflow_drops_oldest_and_never_drops_acks(self, ember_server):
        """A telemetry burst far larger than the outbound buffer, with nothing
        reading it.

        Two guarantees at once, and the second is the important one: telemetry
        is shed under backpressure by design, but an Ack is a command's receipt
        and must survive. A test that only checked the drop counter would pass
        just as happily on an implementation that dropped acks too.
        """
        sock = _connect(ember_server.port)
        try:
            ember_server.reset_stats()

            # Burst well past max_telemetry_buffer (256) without draining.
            ember_server.telemetry_burst(count=3000, interval_us=0, timeout=60.0)
            time.sleep(0.3)

            stats = ember_server.stats()
            assert stats["telemetry_published"] == 3000
            assert stats["telemetry_dropped"] > 0, (
                "no telemetry was shed despite bursting past the buffer cap"
            )
            assert stats["outbound_queue_high_water"] <= 300, (
                "the outbound queue grew past its configured cap "
                f"({stats['outbound_queue_high_water']})"
            )

            # The command path must still work, and its ack must arrive.
            sock.sendall(_command_packet(101, CommandOp.STOP))
            acks = _acks(_drain(sock, duration=5.0))
            assert any(a.ack_sequence == 101 for a in acks), "an Ack was lost to telemetry backpressure"
        finally:
            sock.close()

    def test_queue_depth_returns_to_zero_once_the_consumer_drains(self, ember_server):
        sock = _connect(ember_server.port)
        try:
            ember_server.reset_stats()
            ember_server.telemetry_burst(count=500, interval_us=0)
            _drain(sock, duration=1.5)
            time.sleep(0.3)

            stats = ember_server.stats()
            assert stats["outbound_queue_high_water"] > 0, "the burst never queued anything"
            assert stats["outbound_queue_depth"] == 0, (
                f"queue did not drain (depth={stats['outbound_queue_depth']})"
            )
        finally:
            sock.close()

    def test_burst_does_not_block_the_command_path(self, ember_server):
        """The isolation property the drop-oldest policy exists to provide: a
        saturated telemetry stream must not delay a command being routed."""
        sock = _connect(ember_server.port)
        try:
            burst = threading.Thread(
                target=ember_server.telemetry_burst, args=(4000, 0), daemon=True
            )
            burst.start()

            sock.sendall(_command_packet(201, CommandOp.STOP))
            ack, elapsed = _drain_until(
                sock, lambda a: a.ack_sequence == 201, timeout=10.0
            )
            burst.join(timeout=60.0)

            assert ack is not None, "the command was never acked during a telemetry burst"
            # The Ack sits behind at most max_buffered_telemetry_frames (256)
            # frames of ~32 bytes, so a functioning writer clears it in
            # milliseconds. A second is generous enough not to be flaky and
            # tight enough to catch the failure this guards against, which is
            # the Ack being stuck behind an unbounded backlog.
            assert elapsed < 1.0, f"command ack took {elapsed:.3f}s during a telemetry burst"

            # An Ack enqueued mid-burst sits in the same queue the shed loop
            # pops from, so surviving at all depends on that loop stepping over
            # non-sheddable frames rather than discarding whatever is at the
            # head. Asserting the counter pins the mechanism, not just the
            # outcome: without it this test would still pass on a build that
            # merely happened not to shed while the Ack was queued.
            stats = ember_server.stats()
            assert stats["telemetry_dropped"] > 0, (
                "no shedding occurred; this test is not exercising the path it claims to"
            )
            assert stats["acks_preserved"] > 0, (
                "the Ack survived without the shed loop preserving it - the ordering "
                "that made this pass is incidental"
            )
        finally:
            sock.close()


class TestConnectionLossRecovery:
    def test_abrupt_disconnect_allows_a_fresh_connection(self, ember_server):
        """A peer that vanishes without a clean close, then reconnects.

        This is the scenario that caught the one-way-shutdown bug recorded in
        ADR-005 §3: the adapter used to reuse a single outbound queue across
        connections, so every reconnect after the first silently dropped all
        outbound frames. A single-connection test cannot see it — only a
        second connection can.
        """
        first = _connect(ember_server.port)
        first.sendall(_command_packet(1, CommandOp.MOVE_FORWARD, 1.0))
        assert any(a.ack_sequence == 1 for a in _acks(_drain(first, duration=2.0)))

        # SO_LINGER with a zero timeout makes close() send RST instead of FIN,
        # which is what a crashed process looks like rather than a polite exit.
        first.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        first.close()
        time.sleep(0.5)

        second = _connect(ember_server.port, timeout=10.0)
        try:
            second.sendall(_command_packet(2, CommandOp.TURN, 45.0))
            acks = _acks(_drain(second, duration=3.0))
            assert any(a.ack_sequence == 2 for a in acks), (
                "EMBER did not serve a reconnecting peer - outbound frames were dropped"
            )
            assert ember_server.stats()["connections_accepted"] >= 2
        finally:
            second.close()

    def test_three_consecutive_reconnects_all_work(self, ember_server):
        """Once is luck. The queue-per-connection fix has to hold for every
        connection, not just the second."""
        for attempt in range(3):
            sock = _connect(ember_server.port, timeout=10.0)
            try:
                sequence = 100 + attempt
                sock.sendall(_command_packet(sequence, CommandOp.STOP))
                acks = _acks(_drain(sock, duration=3.0))
                assert any(a.ack_sequence == sequence for a in acks), (
                    f"reconnect #{attempt + 1} received no ack"
                )
            finally:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                sock.close()
            time.sleep(0.4)

    def test_disconnected_client_rejects_rather_than_raising(self, hybrid_rig):
        """`send_command` on a dead connection must return REJECTED, not raise
        a socket exception — `EmberRobotDriver` turns REJECTED into a
        RuntimeError that `Coordinator.dispatch` already knows how to convert
        into an ERROR Message, and a raw OSError would escape that path."""
        hybrid_rig.client.stop()
        time.sleep(0.2)

        result = hybrid_rig.client.send_command(CommandOp.MOVE_FORWARD, 1.0)
        assert result == AckResult.REJECTED

        with pytest.raises(RuntimeError):
            hybrid_rig.driver.move_forward(1.0)

    def test_server_death_surfaces_as_an_error_message_not_a_crash(self, hybrid_rig):
        """End-to-end failure propagation: EMBER dies mid-session, and the
        framework turns that into an ERROR Message the Planner could act on
        rather than an unhandled exception."""
        hybrid_rig.server.kill()
        time.sleep(0.5)

        result = run_perception_to_execution(hybrid_rig.coordinator, frame_index=0)
        assert result.status == MessageStatus.ERROR
        assert result.error, "an ERROR Message must carry a reason"

    def test_watchdog_stops_the_robot_when_the_peer_goes_quiet(self, ember_server):
        """EMBER's own safety net, with no Python participation at all: a
        motion command, then silence, and EMBER must issue its own stop.

        The client is a raw socket precisely so nothing heartbeats — the real
        `EmberBridgeClient` heartbeats every 100ms specifically to prevent
        this, which makes it the wrong tool for provoking it.
        """
        sock = _connect(ember_server.port)
        try:
            before = ember_server.stats()
            mark = ember_server.line_count()

            sock.sendall(_command_packet(1, CommandOp.MOVE_FORWARD, 5.0))
            assert any(a.ack_sequence == 1 for a in _acks(_drain(sock, duration=1.0)))

            # Stay connected and stay silent past the 300ms watchdog deadline.
            time.sleep(1.2)

            after = ember_server.stats()
            assert after["watchdog_stops"] > before["watchdog_stops"], (
                "the watchdog did not fire while motion was in flight and the peer was silent"
            )

            stop_line = ember_server.wait_for_line(
                lambda text: text.startswith("CMD_RECEIVED") and "topic=cmd/motion/stop" in text,
                timeout=2.0,
                since=mark,
            )
            assert stop_line is not None, "EMBER never routed its self-issued stop"
        finally:
            sock.close()

    def test_watchdog_does_not_fire_on_an_idle_but_healthy_connection(self, hybrid_rig):
        """The inverse, and the bug ADR-005 §5 records: an idle connection was
        being torn down by a 200ms recv() timeout before the watchdog deadline,
        which silently suppressed the auto-stop. A heartbeating client that is
        simply not sending commands must be left alone."""
        before = hybrid_rig.server.stats()
        time.sleep(1.5)  # well past the 300ms deadline, with heartbeats flowing
        after = hybrid_rig.server.stats()

        assert after["watchdog_stops"] == before["watchdog_stops"]
        assert after["heartbeats_received"] > before["heartbeats_received"], (
            "no heartbeats arrived; this test is not exercising what it claims to"
        )
        assert after["connections_accepted"] == before["connections_accepted"], (
            "an idle connection was torn down and re-accepted"
        )
        assert hybrid_rig.client.is_connected()


class TestBaselineParity:
    """The pure-Python arm must be a faithful control, not merely a fast one.

    If the two arms disagreed about what they acknowledge or how they fail, the
    benchmark comparison would be measuring two different contracts and its
    speedups would mean nothing — so the properties the comparison relies on
    are asserted rather than assumed.
    """

    def test_baseline_acks_commands_with_the_same_contract(self, baseline_rig):
        for op in (CommandOp.MOVE_FORWARD, CommandOp.TURN, CommandOp.STOP, CommandOp.GRAB_ITEM):
            assert baseline_rig.send_command(op, 1.0) == AckResult.ACCEPTED

    def test_baseline_runs_the_same_pipeline_to_success(self, baseline_rig):
        result = run_perception_to_execution(baseline_rig.coordinator, frame_index=0)
        assert result.status == MessageStatus.SUCCESS, result.error
        assert result.payload["total_steps"] == 1

    def test_baseline_rejects_rather_than_raising_when_stopped(self, baseline_rig):
        baseline_rig.stop()
        assert baseline_rig.client.send_command(CommandOp.MOVE_FORWARD, 1.0) == AckResult.REJECTED

    def test_baseline_watchdog_stops_motion_when_commands_go_quiet(self, baseline_rig):
        before = baseline_rig.watchdog_stop_count
        baseline_rig.send_command(CommandOp.MOVE_FORWARD, 1.0)
        baseline_rig.suspend_heartbeat()

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if baseline_rig.watchdog_stop_count > before:
                break
            time.sleep(0.02)

        assert baseline_rig.watchdog_stop_count > before, "the baseline watchdog never fired"

    def test_baseline_corrupt_payload_is_counted_not_crashed(self, baseline_rig):
        """The checksummed pure-Python codec must reject a tampered payload the
        same way EMBER does — returning, not raising."""
        from benchmarks.bench_core.baseline import CODECS

        codec = CODECS["json+fletcher16"]
        encoded = bytearray(codec.encode({"t": "cmd", "sequence": 1}))
        encoded[3] ^= 0x20  # corrupt the body, leave the trailer

        with pytest.raises(ValueError):
            codec.decode(bytes(encoded))


class TestDiagnosticsInstruments:
    """The diagnostics layer is what the report quotes, so its instruments are
    tested directly rather than trusted because their numbers looked plausible
    in a table."""

    def test_counters_are_accurate_under_concurrent_increments(self):
        """A plain `+=` on an int attribute is a read-modify-write that a
        thread switch can interleave. An undercount here would be
        indistinguishable from the counted thing not happening."""
        from benchmarks.bench_core.diagnostics import Counter

        counter = Counter("test")
        threads = [
            threading.Thread(target=lambda: [counter.increment() for _ in range(5000)])
            for _ in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert counter.value == 8 * 5000

    def test_gauge_remembers_its_peak_after_draining(self):
        from benchmarks.bench_core.diagnostics import Gauge

        gauge = Gauge("depth")
        for value in (1, 5, 200, 3, 0):
            gauge.set(value)

        assert gauge.value == 0
        assert gauge.peak == 200, "a queue that filled and drained must still report its high water"

    def test_contention_probe_records_waits_only_when_contended(self):
        from benchmarks.bench_core.diagnostics import ContentionProbe

        probe = ContentionProbe("test")
        for _ in range(100):
            with probe.guard():
                pass

        assert probe.acquisitions == 100
        assert probe.contended == 0, "an uncontended lock must not register contention"

        holder_ready = threading.Event()
        release = threading.Event()

        def hold() -> None:
            with probe.guard():
                holder_ready.set()
                release.wait(2.0)

        holder = threading.Thread(target=hold, daemon=True)
        holder.start()
        holder_ready.wait(2.0)

        waiter_done = threading.Event()

        def wait_for_lock() -> None:
            with probe.guard():
                pass
            waiter_done.set()

        waiter = threading.Thread(target=wait_for_lock, daemon=True)
        waiter.start()
        time.sleep(0.15)
        release.set()

        assert waiter_done.wait(3.0)
        holder.join(timeout=2.0)
        waiter.join(timeout=2.0)

        assert probe.contended == 1
        assert probe.wait_stats().max_ns > 0

    def test_percentiles_match_the_cpp_definition(self):
        """`compute_stats` must use the same linear-interpolated percentile as
        `ember::bench::percentile`, or every Python-vs-C++ row in the report
        carries a silent methodological skew."""
        from benchmarks.bench_core.stats import compute_stats, percentile

        samples = [float(i) for i in range(1, 101)]
        assert percentile(samples, 50.0) == pytest.approx(50.5)
        assert percentile(samples, 99.0) == pytest.approx(99.01)
        assert percentile(samples, 0.0) == 1.0
        assert percentile(samples, 100.0) == 100.0

        stats = compute_stats(samples)
        assert stats.count == 100
        assert stats.min_ns == 1.0
        assert stats.max_ns == 100.0
        assert stats.mean_ns == pytest.approx(50.5)

    def test_wcet_spikes_are_counted_relative_to_the_median(self):
        from benchmarks.bench_core.stats import compute_stats

        samples = [10.0] * 999 + [10_000.0]
        stats = compute_stats(samples, spike_factor=10.0)

        assert stats.wcet_spikes == 1
        assert stats.spike_threshold_ns == pytest.approx(100.0)

        # And the reason wcet_spikes exists as its own field: a single 1000x
        # outlier in 1000 samples leaves P99 - and therefore the determinism
        # index - completely unmoved. A report quoting only percentiles would
        # call this distribution perfectly deterministic.
        assert stats.p99_ns == 10.0
        assert stats.determinism_index == pytest.approx(1.0)
        assert stats.max_ns == 10_000.0

    def test_speedup_reports_not_measurable_rather_than_infinity(self):
        """A zero hybrid measurement means the clock could not resolve the
        operation. Reporting that as an infinite speedup would be the single
        most misleading number this suite could emit."""
        from benchmarks.bench_core.stats import Comparison, LatencyStats

        baseline = LatencyStats(count=1, p50_ns=100.0, mean_ns=100.0, p99_ns=100.0)
        hybrid = LatencyStats(count=1, p50_ns=0.0, mean_ns=0.0, p99_ns=0.0)
        comparison = Comparison(name="x", baseline=baseline, hybrid=hybrid)

        assert comparison.p50_speedup == 0.0
        assert comparison.mean_speedup == 0.0
