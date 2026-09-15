# ADR-005: EMBER <-> AgentCore/BAI Bridge

**Status:** Accepted (Initial Implementation)

**Date:** September 1, 2026

---

# Background

`edge/` (EMBER) is a C++20 deterministic middleware core: `MemoryPool` /
`ObjectPool` (fixed-block allocation), `EventBus` (synchronous, intra-thread,
type-erased pub/sub), `messaging::Coordinator` / `Subscriber` /
`ThreadSafeQueue` (topic-string pub/sub with per-subscriber queues, the
actual cross-thread delivery mechanism), a `Scheduler`/`Runtime` for
periodic deterministic tasks, and a binary `Serializer` + Fletcher-16
checksum for wire framing.

`agents/` (AgentCore/BAI) is a synchronous Python multi-agent framework:
`Agent`/`ExecutorAgent`/`WebotsExecutorAgent`, a `Message` dataclass, and a
`Coordinator` that dispatches `Message`s to agents via direct in-process
method calls. Nothing in `agents/`, `backend/`, or `rl/` uses asyncio,
sockets, ctypes, or any IPC library today — `WebotsExecutorAgent` talks
directly to Webots' own native Python bindings, not to `edge/`.

The two codebases have never been connected. This ADR defines and
implements that connection: AgentCore's plan/action output reaching
EMBER's actuation layer, and EMBER's sensor/fault telemetry reaching
AgentCore's Planner for re-planning — without slow Python inference ever
sharing a thread, a lock, or a blocking call with EMBER's deterministic
core.

---

# 1. Architecture & Data Flow Blueprint

## IPC mechanism: loopback TCP, framed with EMBER's own binary protocol

| Option | Verdict |
|---|---|
| pybind11 (embed Python or EMBER-as-extension) | Rejected. No binding target exists in `edge/CMakeLists.txt` today. Embedding AgentCore's GIL-bound, multi-second LLM/vision inference calls in the same process as EMBER's deterministic scheduler risks exactly the priority-inversion/blocking failure mode this bridge exists to prevent — a crash or hang in the Python interpreter would take EMBER down with it. |
| Shared memory (`MemoryPool`-backed ring buffer) | Rejected for v1. Lowest latency, but needs a second synchronization primitive (futex/eventfd/semaphore) EMBER doesn't have yet, same-host only, and command/telemetry rates here (motor commands, encoder feedback, fault events — not raw video) don't need it. Left as a documented Tier-2 upgrade path (see "Future Considerations") if a specific stream later demands it. |
| ZeroMQ | Rejected. Adds a new dependency to both a C++ build with zero third-party runtime deps today and a Python build with zero IPC deps today, for a pattern (pub/sub, req/rep) EMBER already has in `Coordinator`/`ThreadSafeQueue` — a raw framed socket reuses the existing `Serializer`/checksum code as-is instead.
| **Loopback TCP socket, EMBER's existing `Serializer::pack` frame format** | **Chosen.** Zero new dependencies on either side (`edge/bridge/tcp_socket.cpp` wraps Winsock2/BSD sockets directly; Python uses stdlib `asyncio`/`struct`). Full OS-process isolation — a Python crash/OOM/GIL-stall cannot corrupt or block EMBER's memory or threads. Cross-platform identically. Reuses `Serializer::pack` and `calculate_fletcher16` unchanged. |

EMBER is the long-lived process and owns the *listening* socket
(`TcpSocket::listen_and_accept`, `edge/bridge/tcp_socket.cpp`); AgentCore
connects to it as the client and reconnects (with exponential backoff) on
disconnect. This means an AgentCore restart/redeploy never requires
restarting EMBER, and EMBER's `accept_loop()` transparently accepts the
next connection — see `edge/bridge/ember_bridge_adapter.cpp`.

## Data flow

```text
 AgentCore/BAI (Python, synchronous agent framework)
 ─────────────────────────────────────────────────────────────
   WebotsExecutorAgent._execute_single_step()
        │  robot.move_forward(distance) / .turn() / .stop() / ...
        ▼
   EmberRobotDriver              (implements the RobotDriver Protocol
        │                         webot_execution.py already declares —
        │                         no framework code changes needed)
        ▼
   EmberBridgeClient.send_command()   (blocks the caller's thread;
        │                              internally hands off to an
        │                              asyncio loop on its own thread)
        ▼
   CommandFrame.to_bytes() -> Serializer.pack(CMD, ...) -> TCP socket
 ═══════════════════════════════════════════════════════════════ loopback
        ▼
 EMBER (C++20, deterministic core)
 ─────────────────────────────────────────────────────────────
   EmberBridgeAdapter::reader_loop()      (dedicated I/O thread)
        │  StreamFrameReader reassembles frames off the byte stream
        ▼
   handle_command_frame()
        │  builds ember::messaging::Message{topic="cmd/motion/move_forward", ...}
        │  coordinator_.publish(msg)         <- Goal 1: Python call -> Coordinator topic
        │  report_command_result(..., Accepted)   <- acks receipt immediately
        ▼
   Coordinator fans out to the Subscriber whose topic matches exactly
        ▼
   MotionSubsystem worker thread (Subscriber::wait_and_pop on its own
   ThreadSafeQueue<shared_ptr<const Message>> — this queue IS the
   ThreadSafeQueue the brief asks Python calls to map into)
        │  actuates via HAL (edge/hal/...)
        ▼
   report_command_result(seq, hash, Completed/Faulted)   <- optional, once
                                                              actuation finishes
```

```text
 EMBER internal state changes
 ─────────────────────────────────────────────────────────────
   e.g. BatteryLowEvent, HardwareFaultEvent, ThermalWarningEvent,
   TaskStateChangedEvent — published on whatever thread noticed them
        ▼
   ember::events::EventBus::publish<EventType>()    (synchronous, intra-thread)
        ▼
   EmberBridgeAdapter's typed subscription (forward_event_as_telemetry<T>)
        │  builds a TelemetryFrame, pushes to a per-connection outbound
        │  ThreadSafeQueue<vector<uint8_t>> — O(1), no I/O, so the
        │  publishing thread (which may be a control-loop thread) never
        │  blocks on the network
        ▼
   writer_loop() (dedicated I/O thread) drains the queue, sends over TCP
 ═══════════════════════════════════════════════════════════════ loopback
        ▼
 AgentCore/BAI
 ─────────────────────────────────────────────────────────────
   EmberBridgeClient._reader_task() decodes TelemetryFrame
        ├─ Motion/Manipulator samples -> cached (get_latest_telemetry),
        │  polled synchronously by EmberRobotDriver.get_distance_to_front()
        └─ Fault/Battery/Thermal/Connection -> EmberTelemetryRelay.on_telemetry()
               -> Message(action="telemetry_alert") -> Coordinator.dispatch()
               -> Planner agent                          <- Goal 2: re-planning
```

## Mapping Python calls onto Coordinator topics / EventBus types

`Coordinator::publish` matches `Message::topic` by **exact string
equality** (`coordinator.cpp:28`) — there is no prefix/wildcard matching —
so each `CommandOp` gets its own exact topic, and each subsystem's
`Subscriber` registers for exactly the topics it executes
(`topic_for()` in `edge/bridge/ember_bridge_adapter.cpp`):

| CommandOp | Coordinator topic |
|---|---|
| `MoveForward`, `Turn`, `Stop` | `cmd/motion/move_forward`, `cmd/motion/turn`, `cmd/motion/stop` |
| `RaiseArm`, `LowerArm`, `GrabItem`, `ReleaseItem` | `cmd/manipulator/raise_arm`, `.../lower_arm`, `.../grab_item`, `.../release_item` |

Outbound, `EmberBridgeAdapter::start()` subscribes to seven existing
`ember::events` types (`edge/events/events_list.hpp`) and forwards each as
a `TelemetryFrame`: `BatteryLowEvent`, `ThermalWarningEvent`,
`HardwareFaultEvent`, `TaskStateChangedEvent`, `ConnectionLostEvent`,
`ConnectionRestoredEvent`. No changes to `EventBus` or the event structs
were needed — the adapter is purely a new subscriber.

---

# 2. Binary Frame Protocol Specification

Every frame is `ember::serialization::Serializer::pack(msg_type, payload)`
(unchanged, `edge/serialization/serializer.hpp`) — big-endian throughout:

| Offset | Size | Field | Value |
|---|---|---|---|
| 0-1 | 2 | Magic | `"EM"` |
| 2 | 1 | `msg_type` | see `MsgType` below |
| 3-4 | 2 | Length | payload length, `u16` BE |
| 5..5+len-1 | len | Payload | one of the structs below |
| 5+len..+2 | 2 | Checksum | Fletcher-16 BE over bytes `[0, 5+len)` |

`FrameCodec::unpack` (new — `edge/serialization/frame_codec.hpp`) is the
validating inverse of `Serializer::pack`, which previously had none.
`StreamFrameReader` (same file; Python twin: `StreamFrameReader` in
`agents/messaging/ember_wire.py`) reassembles frames out of a raw byte
stream, since TCP has no message boundaries — it resyncs past a single
corrupt byte rather than dropping the whole buffered window.

```cpp
enum class MsgType : uint8_t { Command = 0x01, Telemetry = 0x02, Ack = 0x03, Heartbeat = 0x04 };
```

## CommandFrame (`MsgType::Command`, Python -> EMBER) — 21 bytes

| Offset | Size | Field | Type |
|---|---|---|---|
| 0 | 4 | `sequence` | `u32` BE |
| 4 | 8 | `timestamp_ns` | `u64` BE (producer steady-clock ns) |
| 12 | 1 | `op` | `u8` (`CommandOp`) |
| 13 | 4 | `param` | `float` BE (distance in m / angle in deg; 0 if unused) |
| 17 | 4 | `request_id_hash` | `u32` BE (FNV-1a of the originating AgentCore `Message.request_id`) |

`CommandOp`: `MoveForward=0x01, Turn=0x02, Stop=0x03, RaiseArm=0x04,
LowerArm=0x05, GrabItem=0x06, ReleaseItem=0x07`.

## TelemetryFrame (`MsgType::Telemetry`, EMBER -> Python) — 25 bytes + text

| Offset | Size | Field | Type |
|---|---|---|---|
| 0 | 4 | `sequence` | `u32` BE |
| 4 | 8 | `timestamp_ns` | `u64` BE |
| 12 | 1 | `subsystem` | `u8` (`TelemetrySubsystem`) |
| 13 | 4 | `value_a` | `float` BE |
| 17 | 4 | `value_b` | `float` BE |
| 21 | 2 | `code` | `u16` BE |
| 23 | 2 | `text_len` | `u16` BE |
| 25 | `text_len` | `text` | UTF-8 |

`TelemetrySubsystem`: `Motion=1` (`value_a`=distance-to-front m),
`Manipulator=2` (`value_a`=grip state), `Battery=3` (`value_a`=%,
`value_b`=voltage), `Thermal=4` (`value_a`=°C), `Fault=5`
(`code`=error_code, `text`=component: description), `TaskState=6`
(`text`="task_name|state"), `Connection=7` (`code`=0 lost / 1 restored,
`text`=endpoint name).

## AckFrame (`MsgType::Ack`, EMBER -> Python) — 9 bytes

| Offset | Size | Field | Type |
|---|---|---|---|
| 0 | 4 | `ack_sequence` | `u32` BE (echoes `CommandFrame.sequence`) |
| 4 | 1 | `result` | `u8` (`AckResult`) |
| 5 | 4 | `request_id_hash` | `u32` BE |

`AckResult`: `Accepted=0` (routed to a Subscriber — sent synchronously and
immediately by `handle_command_frame`, before actuation), `Completed=1` /
`Faulted=3` (sent later, separately, by a subsystem via
`report_command_result` once actuation finishes — not correlated back to
the original blocking `send_command()` call, see below), `Rejected=2`.

## Heartbeat (`MsgType::Heartbeat`) — empty payload

Sent by AgentCore every 100ms (`EmberBridgeClient._heartbeat_task`);
refreshes the watchdog deadline on the EMBER side without needing an
in-flight command.

## C++ <-> Python parity

C++: `edge/bridge/bridge_frames.hpp` (`CommandFrame`/`TelemetryFrame`/`AckFrame`,
each with `to_bytes()`/`from_bytes()`, manual field-by-field big-endian
writes — never `reinterpret_cast`/struct-punning, so the layout is immune
to compiler padding/alignment). Python: `agents/messaging/ember_wire.py`,
using `struct.Struct(">...")` rather than `ctypes.Structure` — `ctypes`
struct layout follows host alignment rules and does not guarantee the
padding-free layout the wire format requires; `struct` with a `">"` prefix
does. **This was cross-validated byte-for-byte**, not just by inspection: a
throwaway C++ program packed a `CommandFrame`/`TelemetryFrame`/`AckFrame`
with the real compiled `Serializer`, and the Python module decoded the
exact hex output and re-encoded it back to an identical byte string.

---

# 3. Bridging Code Implementation

## C++ (`edge/`)

| File | Role |
|---|---|
| `serialization/endian.hpp` | extended with `u32`/`u64` BE and `read_float_be` (only `u16`/`write_float_be` existed) |
| `serialization/frame_codec.hpp` | new — `FrameCodec::unpack` (the missing inverse of `Serializer::pack`) + `StreamFrameReader` |
| `bridge/bridge_frames.hpp` | `CommandFrame`/`TelemetryFrame`/`AckFrame`/enums |
| `bridge/tcp_socket.hpp/.cpp` | minimal Winsock2/BSD-socket wrapper (no networking dependency existed before) |
| `bridge/ember_bridge_adapter.hpp/.cpp` | the adapter: accept/reader/writer threads, `Coordinator`/`EventBus` wiring, watchdog |
| `tests/bridge_test.cpp` | gtest unit coverage (frame round-trips, checksum/magic rejection, stream reassembly/resync) |
| `tests/integration/pipeline_test_server.cpp` | standalone subprocess used by the Python integration suite — see §5 |

## Python (`agents/`)

| File | Role |
|---|---|
| `messaging/ember_wire.py` | wire-format mirror: `pack_frame`/`unpack_frame`/`StreamFrameReader`, `fletcher16`, `fnv1a_32`, frame dataclasses |
| `agent/execution_agent/ember_bridge_client.py` | `EmberBridgeClient` — background-thread asyncio TCP client exposing synchronous `send_command()`/`get_latest_telemetry()`/`set_telemetry_callback()` |
| `agent/execution_agent/ember_execution_agent.py` | `EmberRobotDriver` — implements the `RobotDriver` Protocol `webot_execution.py` already declares |
| `agent/execution_agent/ember_telemetry_relay.py` | `EmberTelemetryRelay` — pushes Fault/Battery/Thermal/Connection telemetry into `Coordinator.dispatch()` for the Planner |
| `tests/integration/` | cross-language integration suite — see §5 |

**No existing framework file was modified on the Python side.**
`WebotsExecutorAgent` depends only on the structural `RobotDriver`
Protocol it already declares (duck typing, `webot_execution.py:20`), so
running against real EMBER hardware instead of Webots is:

```python
driver = EmberRobotDriver(bridge_client)
executor = WebotsExecutorAgent(robot_driver=driver, world=world_model)  # unchanged class
```

`world_model` (bounding boxes, tracked objects) still comes from the
Vision Agent — EMBER has no concept of vision; it only serves
`get_distance_to_front()` off the last-known `Motion` telemetry sample.

### Verification performed (not just written — actually run)

1. Every new/changed C++ file compiles clean with `g++ -std=c++20 -Wall
   -Wextra` (no CMake available in this environment, so `ember_core`'s
   sources were compiled and linked directly).
2. A standalone end-to-end smoke test (`Coordinator` + `EventBus` +
   `Runtime` + `EmberBridgeAdapter`, a real `Subscriber`, a real
   `TcpSocket` client) exercises the command path, the Ack path, the
   telemetry path, and the watchdog auto-stop — 17/17 checks pass.
3. The C++ and Python wire encoders were cross-checked byte-for-byte (see
   above).
4. A persistent EMBER server binary with a real "motion worker" thread
   (pops commands off its `Subscriber`'s `ThreadSafeQueue`, actuates,
   calls `report_command_result`) was run in the background and driven by
   the real `EmberBridgeClient`/`EmberRobotDriver` — including three
   consecutive reconnects to the same long-lived server process.

This process caught real bugs before they could ship — see §5 for two more
found by the full pytest integration suite:

- **No Ack was ever sent.** The original adapter routed commands into the
  `Coordinator` but never produced an `AckFrame`, so `send_command()`
  would always time out. Fixed by acking `Accepted` synchronously on
  receipt, plus a `report_command_result()` hook subsystems call once
  actuation completes.
- **`outbound_queue_.shutdown()` is one-way.** The adapter originally
  shared one `ThreadSafeQueue` for its whole lifetime and shut it down at
  the end of every connection's `reader_loop()` — `ThreadSafeQueue` has no
  "reopen," so every connection *after the first* silently dropped all
  outbound Ack/Telemetry frames forever. A single-connection smoke test
  could never have caught this; it only showed up once the bridge was
  actually reconnected to, which is exactly the scenario the design
  promises to support. Fixed by giving each accepted connection its own
  queue (`std::atomic<std::shared_ptr<ThreadSafeQueue<...>>>`, replaced in
  `accept_loop()` per connection).

---

# 4. Safety & Real-Time Isolation Guarantees

**A slow/stalled/crashed Python process must never leave EMBER's control
loop blocked, and must never leave the robot moving.**

- **Process isolation, not thread isolation.** AgentCore and EMBER are
  separate OS processes connected only by a loopback socket. A Python GIL
  stall (a multi-second LLM call), a crash, or an OOM kill cannot corrupt
  EMBER's memory, cannot block an EMBER mutex, and cannot occupy an EMBER
  thread — the OS reclaims the socket and EMBER's `accept_loop()` simply
  waits for the next connection.

- **Command watchdog, enforced locally, independent of AgentCore's
  health.** `EmberBridgeAdapter` tracks `last_inbound_ns_` (refreshed by
  either a `CommandFrame` or a `HeartbeatFrame`) and `motion_in_flight_`.
  A task registered on `Runtime`'s own `Scheduler`
  (`schedule_task("ember_bridge_watchdog", ...)`, 20 Hz by default) checks
  every tick: if motion is in flight and nothing has arrived within
  `watchdog_deadline` (250ms default), EMBER publishes a synthetic Stop
  command to `cmd/motion/stop` itself — no round trip to Python, no
  dependency on Python responding at all. `EmberBridgeClient`'s 100ms
  heartbeat exists specifically to keep this from tripping during
  legitimate idle periods.

- **EventBus handlers stay O(1) and I/O-free.** `EventBus::publish` is
  **synchronous and intra-thread** (`EventBus.hpp:82-103`) — it runs the
  subscriber callback on whatever thread published the event, which may be
  a control-loop thread. The adapter's event subscriptions
  (`forward_event_as_telemetry`) only build a `TelemetryFrame` and push it
  onto a `ThreadSafeQueue` (a mutex lock + notify); the actual socket
  `send()` happens later, on a dedicated writer thread. A stalled Python
  peer therefore can only ever stall the writer thread — never the thread
  that published the event.

- **Inbound commands never execute on the network thread.**
  `handle_command_frame` only calls `Coordinator::publish`, which itself
  only pushes onto the target `Subscriber`'s own queue
  (`coordinator.cpp:34-38`) — actuation happens later, on that subsystem's
  own worker thread, at whatever scheduling priority that subsystem needs.
  Network jitter (a slow `recv()`, TCP retransmits) cannot delay actuation
  timing.

- **Bounded, drop-oldest telemetry backpressure.** `enqueue_telemetry`
  caps the outbound queue at `max_buffered_telemetry_frames` (256 default)
  and drops the oldest sample rather than growing unbounded or applying
  backpressure upstream into an EventBus publisher thread. `AckFrame`s
  (`enqueue_priority`) are never dropped this way — correctness of a
  command's receipt acknowledgment matters more than one stale telemetry
  sample.

- **The writer thread cannot block forever on a stalled peer, but an idle
  connection is not treated as a dead one.** `TcpSocket::set_timeouts` sets
  `SO_SNDTIMEO`/`SO_RCVTIMEO` (50ms/200ms default); a `send()` that can't
  complete in time fails the call, which drops the connection and lets
  `accept_loop()` recover — bounded worst case, not indefinite hang. A
  `recv()` *timeout*, though, is deliberately not treated the same way:
  `TcpSocket::recv_some` returns a distinct `-2` for "nothing arrived
  within the timeout" versus `-1`/`0` for a genuine error/close, and
  `reader_loop()` only tears the connection down on the latter. An earlier
  version conflated the two, which meant an idle connection (nothing to
  send, heartbeats spaced out for a test — or, in production, any GC/IO
  pause on the Python side longer than 200ms) got disconnected by
  `reader_loop()` itself well before the 250ms watchdog deadline, silently
  suppressing the very auto-stop this bridge exists to guarantee — see §5.

- **AgentCore never blocks on EMBER for longer than
  `command_timeout_s`.** `EmberBridgeClient.send_command()` (default 2s
  timeout) is the only synchronous boundary on the Python side; a timeout
  or disconnect returns `AckResult.REJECTED` rather than raising a network
  exception into `ExecutorAgent`, which is already set up (via
  `Coordinator.dispatch`'s existing `try/except`) to turn any agent
  exception into an `ERROR` `Message` the Planner can react to — no
  framework code needed to change to handle this.

- **Commands from Python are treated as untrusted input.** `FrameCodec`
  rejects anything with a bad magic/length/checksum without throwing
  (`std::optional`, not an exception) — malformed input from an external
  process is an expected condition on this boundary, not a bug. Range
  clamping of `param` (max safe distance/angle) belongs in each
  subsystem's HAL-facing code, not in the bridge — out of scope here since
  it depends on the specific hardware's limits.

---

# 5. Cross-Language Integration Test Suite

Unit tests on each side (`edge/tests/bridge_test.cpp`;
`agents/tests/unit/`) verify the two halves independently. Neither can
verify the seam. `agents/tests/integration/` closes that gap by driving
the real five-agent AgentCore pipeline against the real, compiled EMBER
runtime — not mocks of each other.

## `edge/tests/integration/pipeline_test_server.cpp`

A standalone (non-gtest) subprocess: the same `Coordinator` + `EventBus` +
`Runtime` + `EmberBridgeAdapter` wiring a real deployment's `main()` would
use, plus a stand-in worker thread that pops commands off each topic's
`Subscriber`, logs them, sleeps `actuation_ms` (simulated actuation time),
and acks `Completed`. It's driven over two channels rather than signals or
a second socket:

- **stdout** (line-based, unbuffered) — structured status events
  (`READY`, `CMD_RECEIVED seq=... topic=... op=... param=...`,
  `CMD_COMPLETED`, `INJECTED ...`) a Python fixture parses to assert on
  what actually happened C++-side, not just what Python believes it sent.
- **stdin** (line-based) — test-only control commands with no wire-protocol
  equivalent, for deterministically triggering EMBER-side events a test
  wants to observe flowing back (`INJECT_BATTERY_LOW`, `INJECT_FAULT`,
  `INJECT_THERMAL`) and for a portable clean shutdown (`QUIT`) that doesn't
  depend on signal delivery working the same way to a Windows vs. POSIX
  subprocess.

Built either via the CMake target `ember_pipeline_test_server` (added to
`edge/CMakeLists.txt` alongside the benchmark executables) or, if no
existing CMake build is found, directly via `g++` by
`agents/tests/integration/ember_test_server.py` — cached and rebuilt only
when a source file changes. Either way it's linked with `-static`:
deliberately, because a dev box can have more than one MinGW/MSYS2
toolchain on `PATH` at once (this one does — `/mingw64/bin` and a separate
UCRT64 install), and a dynamically-linked binary resolves its runtime DLLs
based on whichever the *launching* process's `PATH` favors at that moment.
That differs between running the binary by hand and launching it via
`subprocess.Popen` from a different parent, and a mismatch fails the
process at startup with no useful diagnostic — observed firsthand while
building this suite (`ember_server` fixture failing with an empty stdout
buffer and no error).

## `agents/tests/integration/`

- `ember_test_server.py` — binary discovery/build (above) and
  `EmberTestServerProcess`, the Python-side subprocess wrapper.
- `conftest.py` — `ember_server` (fresh subprocess + port per test),
  `ember_bridge_client`, `ember_robot_driver`, `ember_coordinator` (the
  same five-agent wiring as `agents/tests/conftest.py`'s
  `configured_coordinator`, with `WebotsExecutorAgent` driving
  `EmberRobotDriver` instead of `MockRobotDriver`), `ember_telemetry_relay`.
- `test_full_pipeline_ember_integration.py` — Audio → Vision → Planner →
  Executor through real EMBER for a `pick_up` plan, asserting the C++
  side's own `CMD_RECEIVED`/`CMD_COMPLETED` log lines, not just the
  pipeline's returned `Message`; a documented-limitation test pinning
  `get_object`'s current failure mode (no `MotionSubsystem` yet publishes
  Motion telemetry, so `get_distance_to_front()` always reads back 0.0 —
  see "Future Considerations"); and a reconnect test that kills and restarts
  the EMBER subprocess mid-session and re-runs the pipeline through the
  same `Coordinator`/`EmberBridgeClient`.
- `test_ember_telemetry_integration.py` — `INJECT_*` → `EmberTelemetryRelay`
  → `Coordinator.dispatch` → `Planner.handle_message`, including ordering
  across multiple injected events and a no-op check when no Planner is
  registered.
- `test_ember_safety_integration.py` — the watchdog auto-stop, verified by
  starving a client of heartbeats and asserting EMBER's own
  `CMD_RECEIVED ... topic=cmd/motion/stop` appears with no Python-side stop
  ever sent; the disconnected-client `REJECTED`/`RuntimeError` path, which
  needs no EMBER process at all.

Every test that needs the real binary skips (not fails) when no C++
toolchain and no prebuilt binary are available
(`EMBER_TEST_SERVER_BIN` env var, or a CMake build under `edge/build/`),
mirroring how `agents/tests/conftest.py` already stubs `whisper`/`cv2`/
`ultralytics`/`llama_cpp` when they aren't installed rather than failing
the whole suite.

## Bugs this suite caught (beyond the two in §3)

Actually running this — not just writing it — surfaced two more real
issues, on top of the two already found while building the bridge itself:

- **`reader_loop()` conflated a `recv()` timeout with a real disconnect**
  (fixed in `TcpSocket::recv_some` / `EmberBridgeAdapter::reader_loop`,
  detailed in §4). The watchdog integration test initially failed — not
  because the watchdog didn't fire, but because the connection had already
  been torn down by an idle 200ms `recv()` timeout *before* the 300ms
  watchdog deadline, and the watchdog task's `if (!connected_.load())
  return;` guard (meant for "nobody's connected, nothing to do") silently
  skipped firing on a connection that had just been killed out from under
  it. A single quiet moment on an otherwise-healthy connection was enough
  to trigger this — a real deployment could hit it under nothing more
  unusual than a heartbeat delayed by GC/IO pause. No single-language test
  could have caught this: it depends on the real timing relationship
  between two independent timeouts on two sides of a real socket.
- **A test-authoring bug in this suite's own `ember_coordinator` fixture**
  — it constructed `YOLOVisionAgent()` before `mock_yolo_model` had patched
  `YOLO`, so a real model loaded and then choked on the mocked
  `cv2.imread()` output instead of using the mock. Fixed by adding
  `mock_whisper_model`/`mock_yolo_model`/`mock_qwen_model` as (otherwise
  unused) fixture dependencies to force resolution order — the same
  pattern `agents/tests/conftest.py`'s `configured_coordinator` already
  uses for exactly this reason.

---

# Future Considerations

- A `MotionSubsystem`/`ManipulatorSubsystem` worker thread that actually
  drives `edge/hal/` from the `cmd/motion/*` / `cmd/manipulator/*` topics
  is the remaining integration point —
  `edge/tests/integration/pipeline_test_server.cpp`'s stand-in worker
  sketches the pattern (`Subscriber::pop` -> actuate ->
  `report_command_result`) but never touches real hardware.
- If a future sensor stream needs sub-millisecond latency (a high-rate
  IMU, e.g.), a `MemoryPool`-backed shared-memory ring buffer alongside
  this socket is the documented Tier-2 upgrade — not needed for the
  command/fault/battery/thermal telemetry this ADR covers.
- `AckResult::Completed`/`Faulted` are currently fire-and-forget (not
  correlated back to the original blocking `send_command()` call). If a
  caller needs to await physical completion rather than just receipt, that
  needs a second correlation table keyed on `request_id_hash` — deferred
  since no current caller needs it (`ExecutorAgent`'s sequential
  step-execution model already serializes on Python's side).
- Fixing the recv-timeout/watchdog race (§4, §5) means `reader_loop()` no
  longer treats an idle `recv()` timeout as a disconnect — correct for the
  watchdog, but it also means a peer that vanishes *without* a clean TCP
  close (a hard crash, not a graceful shutdown) is no longer detected by
  `reader_loop()` on its own; `EmberBridgeAdapter` would keep believing a
  connection is live until something actually fails a `send()` or a new
  `CommandFrame`/`Heartbeat` never arrives (which the watchdog already
  handles safely — it doesn't depend on `connected_` flipping false to do
  its job). What this doesn't yet solve: `accept_loop()` won't accept a
  *new* AgentCore connection until the old one's threads exit, so a
  hard-crashed peer with nothing generating outbound traffic (no telemetry
  in flight) could leave EMBER unable to accept a reconnecting AgentCore
  process indefinitely. `SO_KEEPALIVE`, or a periodic health probe
  independent of the command watchdog, would close this gap — not
  implemented here since it's a distinct concern from the race this ADR
  fixes, and no test in §5 currently exercises a true crash-without-FIN.
