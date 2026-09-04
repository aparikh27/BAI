"""Comparative benchmark suite: pure-Python AgentCore vs. AgentCore + EMBER.

Measures what offloading low-level task processing, serialization and
messaging to the C++ runtime actually bought, across four suites:

    e2e          Perception-to-Execution latency, single-command round trip,
                 telemetry loop rate, execution determinism, watchdog reaction
    micro        Serialization, stream reassembly, queue/event handling,
                 lock contention, wire sizes
    resources    CPU, RSS and garbage-collection behaviour under sustained load
    diagnostics  Live runtime counters from both sides of the bridge

Both arms run the *same* Coordinator, the same pipeline and the same
`EmberRobotDriver`; only the runtime beneath the driver differs. See
`bench_core/workload.py` for what is held constant and why, and
`bench_core/baseline.py` for how the legacy arm is constructed.

Usage:

    python -m benchmarks.compare_runtimes                       # everything, default settings
    python -m benchmarks.compare_runtimes --suite e2e --iterations 500
    python -m benchmarks.compare_runtimes --markdown docs/perf.md --json perf.json
    python -m benchmarks.compare_runtimes --suite micro --no-ember

Exits non-zero only on a genuine failure. A missing C++ toolchain is not a
failure: the baseline arm still runs, and the report says plainly which
comparisons were skipped.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

# Running this file directly (`python benchmarks/compare_runtimes.py`) puts
# benchmarks/ on sys.path rather than the repository root, so the absolute
# `benchmarks.` / `agents.` imports below would fail. Prepending the repo root
# makes both invocation styles work, which matters because the -m form is not
# obvious to someone who just found this file.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents.messaging.ember_wire import CommandOp  # noqa: E402
from benchmarks.bench_core import transport  # noqa: E402
from benchmarks.bench_core.reporting import Report, Section, print_report  # noqa: E402
from benchmarks.bench_core.resources import (  # noqa: E402
    GcProbe,
    ResourceSampler,
    format_bytes,
    interpreter_allocation_note,
    probe_backend_name,
)
from benchmarks.bench_core.rigs import BaselineRig, HybridRig, Rig, ember_available  # noqa: E402
from benchmarks.bench_core.stats import (  # noqa: E402
    BenchRow,
    Comparison,
    compute_stats,
    throughput_from_stats,
)
from benchmarks.bench_core.workload import (  # noqa: E402
    WorkloadConfig,
    assert_pipeline_succeeded,
    motor_commands_per_iteration,
    run_perception_to_execution,
)

SUITES = ("e2e", "micro", "resources", "diagnostics")

# Telemetry rate used during the end-to-end suite. High enough that
# `get_distance_to_front()` always has a fresh sample, low enough that the
# telemetry stream is not itself the dominant load — the dedicated telemetry
# benchmark pushes it much harder.
E2E_TELEMETRY_HZ = 50.0


# ---------------------------------------------------------------------------
# End-to-end measurements
# ---------------------------------------------------------------------------


def measure_command_roundtrip(rig: Rig, iterations: int, warmup: int) -> BenchRow:
    """Latency of one motor command from issue to acknowledgement.

    The narrowest measurement of the boundary itself: `send_command` returns
    when the runtime has accepted and routed the command, which is the same
    semantic point in both arms (EMBER's Accepted ack, the legacy runtime's
    ACCEPTED ack). Alternating ops rather than repeating one avoids measuring
    a single topic's queue in a permanently warm state.
    """
    ops = (CommandOp.MOVE_FORWARD, CommandOp.TURN, CommandOp.RAISE_ARM, CommandOp.LOWER_ARM)

    for i in range(warmup):
        rig.send_command(ops[i % len(ops)], 0.5)

    clock = time.perf_counter_ns
    samples: list[float] = []
    for i in range(iterations):
        op = ops[i % len(ops)]
        start = clock()
        result = rig.send_command(op, 0.5)
        samples.append(clock() - start)
        if int(result) not in (0, 1):  # ACCEPTED / COMPLETED
            raise RuntimeError(f"{rig.arm}: command {op.name} was not accepted ({result!r})")

    stats = compute_stats(samples)
    return BenchRow(
        suite="End-to-End",
        name="Motor command round trip (issue -> ack)",
        arm=rig.arm,
        latency=stats,
        throughput_value=throughput_from_stats(stats),
        throughput_unit="cmds/sec",
        notes=f"{iterations:,} commands, {warmup:,} warm-up",
    )


def measure_pipeline_latency(rig: Rig, iterations: int, warmup: int) -> BenchRow:
    """Perception-to-Execution: synthetic frame/speech in, verified motor
    command frames out.

    One iteration is a full four-agent pipeline pass whose plan expands into
    eight acknowledged motor commands, so this is the headline number: what a
    whole task costs, not what one syscall costs.
    """
    for i in range(warmup):
        assert_pipeline_succeeded(run_perception_to_execution(rig.coordinator, i))

    clock = time.perf_counter_ns
    samples: list[float] = []
    for i in range(iterations):
        start = clock()
        result = run_perception_to_execution(rig.coordinator, i)
        samples.append(clock() - start)
        assert_pipeline_succeeded(result)

    stats = compute_stats(samples)
    commands = motor_commands_per_iteration()
    return BenchRow(
        suite="End-to-End",
        name="Perception -> Execution (full pipeline)",
        arm=rig.arm,
        latency=stats,
        throughput_value=throughput_from_stats(stats),
        throughput_unit="tasks/sec",
        notes=f"{iterations:,} tasks x {commands} motor commands",
    )


def measure_concurrent_load(rig: Rig, threads: int, per_thread: int) -> BenchRow:
    """Multi-threaded command load.

    Several agent threads driving the same runtime concurrently is the
    realistic shape — a telemetry relay dispatching into the Coordinator while
    an executor issues commands — and it is where the two arms diverge most
    sharply, because the legacy runtime funnels every one of them through a
    single event loop behind the GIL.
    """
    per_thread_samples: list[list[float]] = [[] for _ in range(threads)]
    barrier = threading.Barrier(threads + 1)
    errors: list[str] = []
    errors_lock = threading.Lock()

    def worker(index: int) -> None:
        samples = per_thread_samples[index]
        clock = time.perf_counter_ns
        ops = (CommandOp.MOVE_FORWARD, CommandOp.TURN, CommandOp.STOP)
        barrier.wait()
        for i in range(per_thread):
            op = ops[i % len(ops)]
            start = clock()
            result = rig.send_command(op, 0.25)
            samples.append(clock() - start)
            if int(result) not in (0, 1):
                with errors_lock:
                    errors.append(f"thread {index}: {op.name} -> {result!r}")
                return

    workers = [
        threading.Thread(target=worker, args=(i,), name=f"bench-load-{i}", daemon=True)
        for i in range(threads)
    ]
    for thread in workers:
        thread.start()

    barrier.wait()
    wall_start = time.perf_counter_ns()
    for thread in workers:
        thread.join()
    wall_ns = time.perf_counter_ns() - wall_start

    if errors:
        raise RuntimeError(f"{rig.arm}: concurrent load failed: {errors[:3]}")

    samples = [s for group in per_thread_samples for s in group]
    stats = compute_stats(samples)
    total = threads * per_thread
    return BenchRow(
        suite="End-to-End",
        name=f"Concurrent command load ({threads} threads)",
        arm=rig.arm,
        latency=stats,
        throughput_value=total / (wall_ns / 1e9) if wall_ns else 0.0,
        throughput_unit="cmds/sec",
        notes=f"{total:,} commands across {threads} threads",
    )


def measure_telemetry_loop(rig: Rig, target_hz: float, duration_s: float) -> tuple[BenchRow, dict]:
    """Sustained telemetry rate and inter-arrival determinism.

    Reports the achieved rate against the requested one and the distribution
    of gaps between consecutive frames. A high-frequency state stream is only
    as good as its worst gap: a loop that averages 500Hz but occasionally
    stalls for 40ms is not a 500Hz loop for any consumer that has to act on
    the samples.
    """
    rig.start_telemetry_stream(target_hz)
    try:
        if not rig.wait_for_telemetry(timeout_s=5.0):
            raise RuntimeError(f"{rig.arm}: no telemetry arrived within 5s at {target_hz}Hz")

        # Settle before recording: the first frames after a stream starts
        # include connection warm-up and one-off allocations.
        time.sleep(0.25)
        rig.start_recording_telemetry()
        time.sleep(duration_s)
        arrivals = rig.stop_recording_telemetry()
    finally:
        rig.stop_telemetry_stream()

    gaps = [float(b - a) for a, b in zip(arrivals, arrivals[1:])]
    stats = compute_stats(gaps)
    achieved_hz = (len(arrivals) - 1) / ((arrivals[-1] - arrivals[0]) / 1e9) if len(arrivals) > 1 else 0.0

    extras = {
        "requested_hz": f"{target_hz:,.1f}",
        "achieved_hz": f"{achieved_hz:,.1f}",
        "rate_attainment": f"{100.0 * achieved_hz / target_hz:.1f}%" if target_hz else "n/a",
        "frames_observed": f"{len(arrivals):,}",
    }
    return (
        BenchRow(
            suite="Telemetry",
            name=f"Telemetry inter-arrival gap @ {target_hz:,.0f}Hz",
            arm=rig.arm,
            latency=stats,
            throughput_value=achieved_hz,
            throughput_unit="frames/sec",
            notes=f"{duration_s:.1f}s window, {len(arrivals):,} frames",
        ),
        extras,
    )


def measure_watchdog_reaction(rig: Rig) -> dict[str, str]:
    """How quickly the runtime self-issues a stop once commands go quiet.

    A safety property rather than a performance one, and the reason it belongs
    in a determinism report: EMBER's watchdog runs on its own scheduler in its
    own process, so a stalled Python interpreter cannot delay it, whereas the
    legacy arm's watchdog is a coroutine on the same loop as everything else.
    """
    before = rig.watchdog_stop_count  # type: ignore[attr-defined]
    rig.send_command(CommandOp.MOVE_FORWARD, 1.0)  # arm the watchdog

    if isinstance(rig, BaselineRig):
        rig.suspend_heartbeat()
        deadline_hint = rig.config.watchdog_deadline_s
    else:
        # The hybrid arm's client heartbeats every 100ms, which by design
        # keeps EMBER's watchdog from tripping. Stopping the client is what
        # actually simulates AgentCore going away — and it also exercises the
        # case the watchdog exists for.
        rig.client.stop()
        deadline_hint = rig.watchdog_ms / 1000.0  # type: ignore[attr-defined]

    start = time.perf_counter()
    fired = False
    while time.perf_counter() - start < deadline_hint + 2.0:
        if rig.watchdog_stop_count > before:  # type: ignore[attr-defined]
            fired = True
            break
        time.sleep(0.01)
    elapsed = time.perf_counter() - start

    return {
        "watchdog deadline": f"{deadline_hint * 1000:.0f} ms",
        "watchdog fired": "yes" if fired else "no",
        "observed reaction": f"{elapsed * 1000:.1f} ms" if fired else "not observed",
    }


# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------


def _pair(name: str, baseline_row: BenchRow | None, hybrid_row: BenchRow | None) -> Comparison | None:
    if baseline_row is None or hybrid_row is None:
        return None
    return Comparison(name=name, baseline=baseline_row.latency, hybrid=hybrid_row.latency)


def run_e2e_suite(report: Report, args: argparse.Namespace, ember_ok: bool) -> None:
    workload = WorkloadConfig(perception_cost_s=args.perception_cost_ms / 1000.0)
    rows: list[BenchRow] = []
    comparisons: list[Comparison] = []
    key_values: dict[str, object] = {}
    footnotes: list[str] = []

    measured: dict[str, dict[str, BenchRow]] = {"baseline": {}, "hybrid": {}}

    # --- baseline arm ---------------------------------------------------
    baseline = BaselineRig(
        workload=workload,
        codec_name=args.codec,
        transport=args.transport,
        actuation_us=args.actuation_us,
        telemetry_hz=E2E_TELEMETRY_HZ,
    )
    with baseline:
        baseline.wait_for_telemetry(timeout_s=2.0)
        measured["baseline"]["rtt"] = measure_command_roundtrip(
            baseline, args.iterations, args.warmup
        )
        measured["baseline"]["pipeline"] = measure_pipeline_latency(
            baseline, max(20, args.iterations // 10), max(5, args.warmup // 10)
        )
        measured["baseline"]["load"] = measure_concurrent_load(
            baseline, args.threads, max(50, args.iterations // args.threads)
        )
        telemetry_row, telemetry_extras = measure_telemetry_loop(
            baseline, args.telemetry_hz, args.telemetry_duration
        )
        measured["baseline"]["telemetry"] = telemetry_row
        for key, value in telemetry_extras.items():
            key_values[f"baseline.telemetry.{key}"] = value
        for key, value in measure_watchdog_reaction(baseline).items():
            key_values[f"baseline.{key}"] = value

    # --- hybrid arm -----------------------------------------------------
    if ember_ok:
        hybrid = HybridRig(
            workload=workload,
            actuation_us=args.actuation_us,
            runtime_hz=args.runtime_hz,
            timer_resolution_ms=args.timer_resolution_ms,
            command_timeout_s=args.command_timeout_s,
        )
        with hybrid:
            hybrid.start_telemetry_stream(E2E_TELEMETRY_HZ)
            if not hybrid.wait_for_telemetry(timeout_s=5.0):
                raise RuntimeError("EMBER produced no Motion telemetry; cannot run the E2E suite")

            measured["hybrid"]["rtt"] = measure_command_roundtrip(
                hybrid, args.iterations, args.warmup
            )
            measured["hybrid"]["pipeline"] = measure_pipeline_latency(
                hybrid, max(20, args.iterations // 10), max(5, args.warmup // 10)
            )
            measured["hybrid"]["load"] = measure_concurrent_load(
                hybrid, args.threads, max(50, args.iterations // args.threads)
            )
            hybrid.stop_telemetry_stream()

            telemetry_row, telemetry_extras = measure_telemetry_loop(
                hybrid, args.telemetry_hz, args.telemetry_duration
            )
            measured["hybrid"]["telemetry"] = telemetry_row
            for key, value in telemetry_extras.items():
                key_values[f"hybrid.telemetry.{key}"] = value

            scheduler = hybrid.scheduler_stats()
            if scheduler.get("samples", 0) > 1:
                requested_period_ms = scheduler["period_us"] / 1000.0
                achieved_period_ms = scheduler["p50_ns"] / 1e6
                attainment = (
                    100.0 * requested_period_ms / achieved_period_ms
                    if achieved_period_ms > 0
                    else 0.0
                )
                key_values["ember.scheduler requested period"] = f"{requested_period_ms:.2f} ms"
                key_values["ember.scheduler P50 interval"] = f"{achieved_period_ms:.3f} ms"
                key_values["ember.scheduler P99 interval"] = f"{scheduler['p99_ns'] / 1e6:.3f} ms"
                key_values["ember.scheduler max interval (WCET)"] = (
                    f"{scheduler['max_ns'] / 1e6:.3f} ms"
                )
                key_values["ember.scheduler jitter (stddev)"] = (
                    f"{scheduler['stddev_ns'] / 1000:.1f} us"
                )
                key_values["ember.scheduler rate attainment"] = f"{attainment:.1f}%"

                # A scheduler that only reaches a fraction of its configured
                # rate is the single most actionable thing this suite can
                # report, and it is easy to miss inside a table of intervals -
                # so it becomes a footnote in its own right.
                if attainment < 80.0:
                    footnotes.append(
                        f"EMBER's scheduler reached only {attainment:.0f}% of its configured "
                        f"rate ({achieved_period_ms:.2f} ms actual vs "
                        f"{requested_period_ms:.2f} ms requested). On Windows this is a "
                        "platform limit rather than a design one: ember::time::Rate::sleep() "
                        "uses std::this_thread::sleep_until, which on this toolchain "
                        "quantizes to the ~15.6 ms system tick even with a 1 ms timer "
                        "quantum successfully requested - measured directly, Sleep(5) drops "
                        "to 5.7 ms under timeBeginPeriod(1) while sleep_until stays at 15.3 "
                        "ms. No scheduled task can run faster than that until Rate::sleep "
                        "changes; see the header of "
                        "edge/tests/integration/bench_harness_server.cpp."
                    )

            for key, value in measure_watchdog_reaction(hybrid).items():
                key_values[f"hybrid.{key}"] = value
    else:
        footnotes.append(
            "Hybrid arm skipped: no compiled EMBER runtime available on this machine."
        )

    labels = {
        "rtt": "Motor command round trip",
        "pipeline": "Perception -> Execution task",
        "load": f"Concurrent load ({args.threads} threads)",
        "telemetry": f"Telemetry inter-arrival @ {args.telemetry_hz:.0f}Hz",
    }
    for key, label in labels.items():
        baseline_row = measured["baseline"].get(key)
        hybrid_row = measured["hybrid"].get(key)
        if baseline_row:
            rows.append(baseline_row)
        if hybrid_row:
            rows.append(hybrid_row)
        comparison = _pair(label, baseline_row, hybrid_row)
        if comparison is not None:
            comparisons.append(comparison)

    footnotes.append(
        "Round-trip latency ends at command acknowledgement (routed and accepted), which is "
        "the same semantic point in both arms - not at physical actuation completion."
    )
    if args.transport == "in_process":
        footnotes.append(
            "The baseline arm ran in-process, with no socket at all, while the hybrid arm pays "
            "a full loopback round trip per command. That is the architecture AgentCore "
            "actually had, so it is the honest default - but it means a per-command latency "
            "figure favouring the baseline is measuring the absence of IPC, not the presence "
            "of a faster runtime. Re-run with --transport loopback to hold the transport "
            "constant; the telemetry-rate, determinism and watchdog figures are unaffected "
            "either way, since none of them are dominated by per-command syscall cost."
        )
    footnotes.append(
        "Telemetry gap statistics describe the interval between consecutive frames; a low P50 "
        "with a high P99 is a stream that is fast on average and unreliable when it matters."
    )
    if args.perception_cost_ms == 0:
        footnotes.append(
            "Perception agents are stubbed at zero cost so the pipeline figure isolates the "
            "part EMBER changes; use --perception-cost-ms to fold a realistic inference cost "
            "back in."
        )

    report.add(
        Section(
            title="1. End-to-End Latency and Determinism",
            description=(
                "Perception-to-Execution latency, single-command round trip, sustained "
                "telemetry rate and watchdog reaction, measured through the identical "
                "Coordinator/pipeline/driver stack on both runtimes."
            ),
            rows=rows,
            comparisons=comparisons,
            key_values=key_values,
            footnotes=footnotes,
        )
    )


def run_micro_suite(report: Report, args: argparse.Namespace) -> None:
    iterations = max(2000, args.iterations * 10)
    warmup = max(500, args.warmup * 10)

    serialization = transport.serialization_rows(iterations, warmup)
    reassembly = transport.stream_reassembly_rows(max(1000, iterations // 5), warmup // 5)

    def find(rows: list[BenchRow], name: str, arm_prefix: str) -> BenchRow | None:
        for row in rows:
            if row.name == name and row.arm.startswith(arm_prefix):
                return row
        return None

    comparisons: list[Comparison] = []
    for name in (
        "Command encode (framed)",
        "Command decode (framed+verify)",
        "Telemetry encode (no text)",
        "Telemetry decode (no text)",
        "Telemetry encode (48B text)",
        "Telemetry decode (48B text)",
    ):
        baseline_row = find(serialization, name, f"{transport.PYTHON_ARM} [{args.codec}]")
        hybrid_row = find(serialization, name, transport.BINARY_ARM)
        comparison = _pair(f"{name} [vs {args.codec}]", baseline_row, hybrid_row)
        if comparison is not None:
            comparisons.append(comparison)

    report.add(
        Section(
            title="2a. Data Transport and Serialization",
            description=(
                "Encode/decode of the exact messages that cross the bridge, at the framed "
                "level - Python json/pickle with a length prefix, versus EMBER's binary "
                "packing with a Fletcher-16 checksum. The C++ side of these same operations "
                "is measured by edge/benchmarks/bench_bridge.cpp."
            ),
            rows=serialization + reassembly,
            comparisons=comparisons,
            key_values=transport.wire_size_table(),
            footnotes=[
                "Timed in batches, not per call: a 21-byte struct pack completes faster than "
                "the platform clock can resolve, so per-call tails from this section are "
                "clock artifacts and are not quoted as determinism evidence.",
                "json+fletcher16 is the like-for-like row against the binary arm - plain json "
                "performs no integrity checking at all.",
            ],
        )
    )

    event_rows = (
        transport.asyncio_queue_rows(iterations, warmup)
        + transport.thread_queue_rows(iterations, warmup)
        + transport.cross_thread_handoff_rows(
            min(2000, max(500, args.iterations)), min(200, args.warmup)
        )
    )
    contention_rows = transport.queue_contention_rows(
        producer_counts=(1, 2, 4, args.threads) if args.threads not in (1, 2, 4) else (1, 2, 4, 8),
        per_producer=max(2000, args.iterations * 4),
    )

    report.add(
        Section(
            title="2b. Event Handling and Lock Contention",
            description=(
                "Python's asyncio and threading queues against EMBER's ThreadSafeQueue and "
                "EventBus. The contention rows repeat the same push measurement at 1/2/4/8 "
                "producers; the growth in P99 across those rows is the lock-contention "
                "figure, obtained without instrumenting (and thereby slowing) the queue."
            ),
            rows=event_rows + contention_rows,
            footnotes=[
                "EMBER's counterpart numbers come from `bench_bridge --markdown=...` "
                "(edge/benchmarks/bench_bridge.cpp), which uses the same producer counts, the "
                "same per-producer volume and the same percentile definition.",
                "asyncio.Queue is not thread-safe and queue.Queue is; both appear because the "
                "legacy runtime used the former while the boundary EMBER replaced needs the "
                "latter's guarantees.",
            ],
        )
    )


def run_resource_suite(report: Report, args: argparse.Namespace, ember_ok: bool) -> None:
    workload = WorkloadConfig(perception_cost_s=args.perception_cost_ms / 1000.0)
    key_values: dict[str, object] = {
        "probe backend": probe_backend_name(),
        "interpreter": interpreter_allocation_note(),
    }
    footnotes = [
        "The hybrid arm is sampled across both processes. Offloading work to a C++ runtime "
        "relocates its memory rather than eliminating it, and a Python-only measurement "
        "would report that relocation as a saving.",
        "CPU percent is CPU-seconds per wall-second and legitimately exceeds 100 for a "
        "multi-threaded process; EMBER runs accept, reader, writer, runtime and per-subsystem "
        "worker threads.",
    ]

    def profile(rig: Rig, label: str) -> None:
        commands = max(500, args.iterations)
        # Raw commands alone allocate almost nothing on the Python side, so a
        # profile built only from them reports zero collections and says
        # nothing about GC. The pipeline is where the interpreter actually
        # churns - a Message per hop, payload dicts merged at every step - so
        # both are run under one sampling window and the totals cover the mix
        # a real deployment produces.
        tasks = max(20, commands // 20)
        rig.wait_for_telemetry(timeout_s=2.0)

        with ResourceSampler(rig.process_pids(), interval_s=0.05) as sampler, GcProbe() as gc_probe:
            start = time.perf_counter()
            for i in range(commands):
                rig.send_command(
                    CommandOp.MOVE_FORWARD if i % 2 == 0 else CommandOp.TURN, 0.5
                )
            command_elapsed = time.perf_counter() - start

            pipeline_start = time.perf_counter()
            for i in range(tasks):
                assert_pipeline_succeeded(run_perception_to_execution(rig.coordinator, i))
            pipeline_elapsed = time.perf_counter() - pipeline_start
            elapsed = time.perf_counter() - start
        resources = sampler.stop()
        gc_report = gc_probe.report

        key_values[f"{label} / commands issued"] = f"{commands:,}"
        key_values[f"{label} / pipeline tasks"] = f"{tasks:,}"
        key_values[f"{label} / wall time"] = f"{elapsed:.3f} s"
        key_values[f"{label} / sustained throughput"] = (
            f"{commands / command_elapsed:,.0f} cmds/s, {tasks / pipeline_elapsed:,.1f} tasks/s"
        )
        for process in resources.processes:
            if not process.available:
                key_values[f"{label} / {process.label}"] = "no probe available"
                continue
            key_values[f"{label} / {process.label} peak RSS"] = format_bytes(
                process.rss_peak_bytes
            )
            key_values[f"{label} / {process.label} RSS growth"] = format_bytes(
                process.rss_growth_bytes
            )
            key_values[f"{label} / {process.label} CPU"] = (
                f"{process.cpu_seconds:.3f} s ({process.cpu_percent:.1f}%)"
            )
        key_values[f"{label} / total peak RSS"] = format_bytes(resources.total_rss_peak_bytes)
        key_values[f"{label} / total CPU"] = f"{resources.total_cpu_seconds:.3f} s"
        key_values[f"{label} / GC collections"] = (
            f"{gc_report.total_collections} "
            f"(gen0={gc_report.collections_gen0}, gen1={gc_report.collections_gen1}, "
            f"gen2={gc_report.collections_gen2})"
        )
        key_values[f"{label} / GC total pause"] = f"{gc_report.total_pause_ns / 1e6:.3f} ms"
        key_values[f"{label} / GC max pause"] = f"{gc_report.max_pause_ns / 1e6:.3f} ms"
        key_values[f"{label} / GC pause per 1k ops"] = (
            f"{gc_report.total_pause_ns / 1e6 / ((commands + tasks) / 1000):.3f} ms"
        )
        for note in resources.notes:
            footnotes.append(note)

    baseline = BaselineRig(
        workload=workload,
        codec_name=args.codec,
        transport=args.transport,
        actuation_us=args.actuation_us,
    )
    with baseline:
        profile(baseline, "Pure Python")

    if ember_ok:
        hybrid = HybridRig(
            workload=workload,
            actuation_us=args.actuation_us,
            runtime_hz=args.runtime_hz,
            timer_resolution_ms=args.timer_resolution_ms,
            command_timeout_s=args.command_timeout_s,
        )
        with hybrid:
            # The pipeline half of the profile calls get_distance_to_front(),
            # which reads EMBER's last Motion sample - without a telemetry
            # stream it reads 0.0 and _get_object takes its failure path, so
            # the profile would be of a pipeline that did almost nothing. The
            # baseline arm gets the equivalent from seed_telemetry() at start.
            hybrid.start_telemetry_stream(E2E_TELEMETRY_HZ)
            try:
                profile(hybrid, "Hybrid")
            finally:
                hybrid.stop_telemetry_stream()
    else:
        footnotes.append("Hybrid arm skipped: no compiled EMBER runtime available.")

    key_values.update(
        {f"codec memory / {k}": v for k, v in transport.gc_pressure_summary().items()}
    )
    footnotes.append(
        "Codec memory rows isolate encoding cost from the runtime arms: retained bytes are "
        "measured with tracemalloc while 256 encoded messages are held at once - the same "
        "depth BridgeConfig::max_buffered_telemetry_frames allows - so the figure reads as "
        "what a full outbound buffer costs in each encoding."
    )
    footnotes.append(
        "Zero garbage collections is a result, not a gap in the measurement: CPython reclaims "
        "these objects by reference counting as they go out of scope, and the generational "
        "counter is decremented on deallocation, so message churn never climbs toward the "
        "gen0 threshold. GC pauses on this workload come from objects that survive, not from "
        "encoding volume - which is why the retained-bytes column is the one that moves."
    )

    report.add(
        Section(
            title="3. Resource Consumption",
            description=(
                "CPU time, resident memory and garbage-collection behaviour while sustaining "
                "a command load, sampled at 20Hz across every process each arm uses."
            ),
            key_values=key_values,
            footnotes=footnotes,
        )
    )


def run_diagnostics_suite(report: Report, args: argparse.Namespace, ember_ok: bool) -> None:
    workload = WorkloadConfig(perception_cost_s=args.perception_cost_ms / 1000.0)
    key_values: dict[str, object] = {}
    footnotes: list[str] = []

    commands = max(200, args.iterations // 2)

    baseline = BaselineRig(
        workload=workload,
        codec_name=args.codec,
        transport=args.transport,
        telemetry_hz=args.telemetry_hz,
    )
    with baseline:
        baseline.wait_for_telemetry(timeout_s=2.0)
        for i in range(commands):
            baseline.send_command(CommandOp.MOVE_FORWARD if i % 2 else CommandOp.TURN, 0.5)
        time.sleep(0.3)
        snapshot = baseline.diagnostics.snapshot()
        for key, value in snapshot.as_key_values().items():
            key_values[f"[baseline] {key}"] = value

    if ember_ok:
        hybrid = HybridRig(
            workload=workload,
            actuation_us=args.actuation_us,
            runtime_hz=args.runtime_hz,
            timer_resolution_ms=args.timer_resolution_ms,
            command_timeout_s=args.command_timeout_s,
        )
        with hybrid:
            hybrid.reset_remote_stats()
            hybrid.start_telemetry_stream(args.telemetry_hz)
            hybrid.wait_for_telemetry(timeout_s=5.0)
            for i in range(commands):
                hybrid.send_command(CommandOp.MOVE_FORWARD if i % 2 else CommandOp.TURN, 0.5)
            time.sleep(0.3)
            hybrid.stop_telemetry_stream()
            hybrid.refresh_diagnostics()
            snapshot = hybrid.diagnostics.snapshot()
            for key, value in snapshot.as_key_values().items():
                key_values[f"[hybrid] {key}"] = value

        footnotes.append(
            "ember.* counters are read out of the EMBER process's own atomics via its STATS "
            "control command, not inferred from the Python side - so a dropped telemetry "
            "frame is counted where it was actually dropped."
        )
    else:
        footnotes.append("Hybrid diagnostics skipped: no compiled EMBER runtime available.")

    footnotes.append(
        "Lock rows report contended acquisitions only. The probe takes a non-blocking "
        "acquire first and reads the clock solely when that fails, so an uncontended lock "
        "costs no timing overhead and the probe does not manufacture the contention it "
        "measures."
    )

    report.add(
        Section(
            title="4. Runtime Diagnostics",
            description=(
                "Live observability counters after a command burst with telemetry streaming: "
                "task throughput, dropped telemetry, IPC queue depth and lock contention, "
                "read from both sides of the bridge."
            ),
            key_values=key_values,
            footnotes=footnotes,
        )
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compare_runtimes",
        description="Comparative benchmarks: pure-Python AgentCore vs AgentCore + EMBER.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--suite",
        action="append",
        choices=(*SUITES, "all"),
        help="suite to run; repeatable. Defaults to all.",
    )
    parser.add_argument("--iterations", type=int, default=1000, help="timed iterations per case")
    parser.add_argument("--warmup", type=int, default=200, help="untimed warm-up iterations")
    parser.add_argument("--threads", type=int, default=4, help="threads for the load test")
    parser.add_argument(
        "--telemetry-hz", type=float, default=500.0, help="telemetry rate for the loop benchmark"
    )
    parser.add_argument(
        "--telemetry-duration", type=float, default=2.0, help="telemetry sampling window, seconds"
    )
    parser.add_argument(
        "--actuation-us",
        type=int,
        default=0,
        help="simulated actuation hold per command; 0 measures the runtime, not the hardware",
    )
    parser.add_argument(
        "--perception-cost-ms",
        type=float,
        default=0.0,
        help="simulated per-stage perception cost, added identically to both arms",
    )
    parser.add_argument("--runtime-hz", type=int, default=200, help="EMBER Runtime tick rate")
    parser.add_argument(
        "--timer-resolution-ms",
        type=int,
        default=1,
        help="timer quantum the EMBER host process requests (Windows); 0 leaves the "
        "system default (~15.6ms), which caps the runtime tick rate at ~64Hz",
    )
    parser.add_argument(
        "--command-timeout-s", type=float, default=3.0, help="bridge client command timeout"
    )
    parser.add_argument(
        "--codec",
        choices=("json", "json+fletcher16", "pickle"),
        default="json",
        help="serialization used by the pure-Python arm",
    )
    parser.add_argument(
        "--transport",
        choices=("in_process", "loopback"),
        default="in_process",
        help="pure-Python transport: in_process is the true legacy architecture; "
        "loopback matches the hybrid arm's socket",
    )
    parser.add_argument("--markdown", type=Path, help="write the report as Markdown to this path")
    parser.add_argument("--json", type=Path, help="write the raw results as JSON to this path")
    parser.add_argument(
        "--no-ember", action="store_true", help="skip the hybrid arm even if EMBER is available"
    )
    parser.add_argument(
        "--rebuild-ember", action="store_true", help="force a rebuild of the EMBER harness binary"
    )
    parser.add_argument("--quiet", action="store_true", help="suppress the console report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    suites = args.suite or ["all"]
    if "all" in suites:
        suites = list(SUITES)

    if args.rebuild_ember:
        from benchmarks.bench_core.ember_harness import find_or_build_server_binary

        find_or_build_server_binary(force_rebuild=True)

    ember_ok = False
    ember_detail = "disabled by --no-ember"
    if not args.no_ember:
        ember_ok, ember_detail = ember_available()

    report = Report(
        title="AgentCore/BAI Runtime Comparison: Pure Python vs Hybrid Python + EMBER",
        subtitle=(
            f"suites={','.join(suites)} | iterations={args.iterations} | "
            f"threads={args.threads} | baseline={args.codec}/{args.transport}"
        ),
    )
    report.environment["ember runtime"] = ember_detail if ember_ok else f"unavailable ({ember_detail})"

    started = time.perf_counter()
    if "e2e" in suites:
        run_e2e_suite(report, args, ember_ok)
    if "micro" in suites:
        run_micro_suite(report, args)
    if "resources" in suites:
        run_resource_suite(report, args, ember_ok)
    if "diagnostics" in suites:
        run_diagnostics_suite(report, args, ember_ok)
    report.environment["total benchmark wall time"] = f"{time.perf_counter() - started:.1f} s"

    if not args.quiet:
        print_report(report)

    if args.markdown:
        path = report.write_markdown(args.markdown)
        print(f"\nMarkdown report written to: {path}")
    if args.json:
        path = report.write_json(args.json)
        print(f"JSON results written to: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
