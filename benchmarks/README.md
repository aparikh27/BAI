# Comparative Benchmarking & Diagnostics

Measures what moving AgentCore/BAI's low-level task processing, serialization and
messaging into the EMBER C++ runtime actually bought — in latency, throughput,
determinism and resource cost — against the pure-Python implementation it
replaced.

```bash
python -m benchmarks.compare_runtimes                                  # every suite
python -m benchmarks.compare_runtimes --suite e2e --iterations 2000
python -m benchmarks.compare_runtimes --markdown docs/perf.md --json perf.json
python -m benchmarks.compare_runtimes --suite micro --no-ember         # no compiler needed
```

A missing C++ toolchain is not an error: the baseline arm still runs and the
report states which comparisons were skipped and why.

---

## What makes the comparison fair

Everything above the robot driver is identical in both arms — the same
`Coordinator`, the same `PipelineManager`, the same real `WebotsExecutorAgent`
running its real `_get_object` servo logic, and literally the same
`EmberRobotDriver` class, because `LegacyRuntime` exposes the same synchronous
`send_command`/`get_latest_telemetry` surface `EmberBridgeClient` does. Only the
runtime beneath the driver differs.

Four deliberate choices are worth knowing before reading any number:

**Perception is stubbed at zero cost by default.** Whisper and YOLO inference
are identical in both arms and cost 10–100× more than everything being measured;
including them would bury the difference rather than change it. The stubs keep
the pipeline's real structure — four `Coordinator.dispatch` hops, real `Message`
construction, real payload merging — and `--perception-cost-ms` folds a
realistic constant back in when the question is "what fraction of a whole task
did this improve".

**The baseline really serializes.** `agents/messaging/helpers.py` already carries
`message_to_dict`/`message_from_dict`, so a Message crossing a boundary in the
legacy framework became a dict and then bytes. Skipping that would delete the
exact cost the binary protocol replaces.

**Two baseline transports.** `--transport in_process` is the architecture
AgentCore actually had before EMBER — one interpreter, no socket — and is the
default. `--transport loopback` puts the same Python runtime behind a TCP socket,
matching the hybrid arm's transport exactly. Running both separates "what did the
binary protocol and the C++ runtime buy" from "what did merely moving to a socket
cost". The `in_process` arm is the *harder* comparison for EMBER, since it hands
Python a free IPC boundary.

**Both arms are acknowledged at the same semantic point.** A measured round trip
ends when the runtime has accepted and routed the command — EMBER's `Accepted`
ack, the legacy runtime's `ACCEPTED` — not at physical actuation.

---

## Suites

| Suite | Measures |
|---|---|
| `e2e` | Perception→Execution task latency, single-command round trip, telemetry loop rate and inter-arrival determinism, concurrent multi-threaded load, watchdog reaction time, EMBER scheduler jitter/WCET |
| `micro` | Framed encode/decode per codec, stream reassembly, wire sizes, `asyncio.Queue`/`queue.Queue`/`deque`, cross-thread handoff, lock contention at 1/2/4/8 producers |
| `resources` | CPU seconds, peak RSS and GC pause behaviour under sustained load, sampled at 20 Hz across **every** process each arm uses |
| `diagnostics` | Live counters from both sides: task throughput, dropped telemetry, IPC queue depth, lock contention |

`resources` samples the hybrid arm across both processes on purpose. Offloading
work to a C++ runtime relocates its memory rather than eliminating it, and a
Python-only measurement would report that relocation as a saving. `psutil` is
used when present; otherwise the probes fall back to ctypes against
psapi/kernel32 on Windows and `/proc` on Linux, and the report records which
backend produced the numbers.

## Reading the columns

- **P99/P50** — a unitless determinism index. `1.0` is perfect; the larger it
  gets, the further the tail is from the typical case. It compares directly
  across two runtimes whose absolute latencies differ by an order of magnitude.
- **Spikes** — samples exceeding 10× the median (WCET spikes). A multiple of the
  median rather than a fixed budget is what makes the count comparable between
  arms; any absolute threshold would flatter one by construction. This column
  exists because percentiles hide rare outliers: a single 1000× spike in 1000
  samples leaves P99 completely unmoved.
- **Jitter** — peak-to-peak spread, which is what bounds a control loop's worst
  case, as opposed to stddev, which describes its typical one.
- **`n/a` speedup** — the hybrid measurement was zero, meaning the clock could
  not resolve the operation. Reported as "not measurable" rather than as an
  infinite speedup.

Percentiles use the linear-interpolated ("R7") definition, matching
`ember::bench::percentile` in `edge/benchmarks/bench_framework.hpp` exactly, so
Python and C++ rows in the same report are directly comparable.

## The C++ half

`edge/benchmarks/bench_bridge.cpp` measures the same operations on the C++ side
— the real `CommandFrame`/`TelemetryFrame`/`AckFrame` at their real sizes,
`StreamFrameReader` reassembly, `ThreadSafeQueue` at 1/2/4/8 producers,
`Coordinator::publish` fan-out and `EventBus::publish` — using the same
percentile definition and the same producer counts, so its table reads
side-by-side with the Python one.

```bash
cmake --build build --target run_benchmarks     # all suites -> benchmarks/results/BENCHMARKS.md
./bench_bridge --markdown=results/BENCHMARKS.md # just the bridge path
```

## Layout

```
compare_runtimes.py       CLI entry point and the suites
bench_core/
  stats.py                percentiles, WCET spikes, comparisons (twin of bench_framework.hpp)
  reporting.py            ASCII console tables + Markdown/JSON export
  resources.py            CPU / RSS / GC probes, psutil-optional
  diagnostics.py          Counter / RateMeter / Gauge / ContentionProbe
  baseline.py             the pure-Python legacy runtime (the control arm)
  ember_harness.py        locate, build and drive the compiled EMBER server
  workload.py             the shared perception-to-execution workload
  rigs.py                 BaselineRig / HybridRig — one interface over both arms
  transport.py            serialization and event-handling micro-benchmarks
```

The hybrid arm runs against `edge/tests/integration/bench_harness_server.cpp`,
built on demand via CMake or a cached `g++` invocation. It is a separate binary
from `ember_pipeline_test_server`: that one polls its Subscribers on a 2 ms sleep,
which is fine for correctness tests and fatal to a latency measurement.

## Platform note: Windows timer granularity

The harness requests a 1 ms timer quantum (`timeBeginPeriod`) and opts out of
background timer-resolution throttling (`SetProcessInformation`), and `READY`
reports whether each call succeeded. Even so, MinGW's
`std::this_thread::sleep_until` — which `ember::time::Rate::sleep()` uses —
does not benefit. Measured on this toolchain, 40 consecutive 5 ms waits:

| | default quantum | `timeBeginPeriod(1)` |
|---|---:|---:|
| `Sleep(5)` | 15.453 ms | 5.722 ms |
| `sleep_until(+5ms)` | 15.902 ms | 15.306 ms |

So a `Runtime` configured for 200 Hz ticks at roughly 63 Hz on Windows, and no
scheduled task can have a shorter effective period. The `e2e` suite reports this
as **scheduler rate attainment** and raises a footnote when it falls below 80%.
Fixing it means changing `Rate::sleep` to sleep short and finish on a spin, or to
use a waitable timer directly — a change to EMBER's core timing behaviour, out of
scope for a measurement tool. Pass `--timer-resolution-ms 0` to see the
unmitigated numbers.

## Tests

`tests/test_hybrid_pipeline.py` covers the seam this suite measures: byte-for-byte
codec parity between `ember_wire.py` and `bridge_frames.hpp`, payload integrity
at volume, and error injection (corrupt checksums, truncated and fragmented
frames, garbage on the wire, telemetry overflow, abrupt disconnects, watchdog
behaviour). It shares the rigs in `bench_core/`, so the thing it verifies is the
same thing the benchmarks measure.

```bash
pytest tests/test_hybrid_pipeline.py -q
```
