"""Real-time diagnostics for the combined AgentCore + EMBER runtime.

This is the Python-side observability layer the benchmark suite reports from,
and it is designed to be usable outside the benchmarks: a deployment can hold
a `RuntimeDiagnostics` and poll `snapshot()` from a health endpoint.

Four kinds of instrument, matching the four things worth watching on this
runtime:

  Counter          monotonic totals (tasks completed, packets dropped)
  RateMeter        counters plus the wall time to derive throughput (Hz)
  Gauge            current value plus its high-water mark (IPC queue depth)
  ContentionProbe  a lock wrapper that measures acquisition wait

`ContentionProbe` is the one that needs justifying. Lock contention cannot be
read out of `threading.Lock`, so measuring it requires either sampling
(misses short waits entirely) or wrapping. This wraps: it does a non-blocking
`acquire(False)` first, and only times the wait when that fails. An
uncontended acquisition therefore costs one extra non-blocking acquire and no
clock reads at all, which keeps the probe from manufacturing the contention
it is meant to measure.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from benchmarks.bench_core.stats import LatencyStats, compute_stats


class Counter:
    """Monotonic counter. Guarded by a lock rather than relying on the GIL:
    `+=` on an int attribute is a read-modify-write that a thread switch can
    interleave, and an undercount in a dropped-packet counter would be
    indistinguishable from the thing working correctly."""

    __slots__ = ("_value", "_lock", "name")

    def __init__(self, name: str):
        self.name = name
        self._value = 0
        self._lock = threading.Lock()

    def increment(self, amount: int = 1) -> None:
        with self._lock:
            self._value += amount

    @property
    def value(self) -> int:
        with self._lock:
            return self._value

    def reset(self) -> None:
        with self._lock:
            self._value = 0


class RateMeter:
    """A counter that also knows how long it has been counting, so it can
    report a rate in Hz. Used for telemetry-loop and task-throughput figures,
    where the raw total is far less informative than the sustained rate."""

    __slots__ = ("_count", "_lock", "_started", "name")

    def __init__(self, name: str):
        self.name = name
        self._count = 0
        self._lock = threading.Lock()
        self._started = time.perf_counter()

    def mark(self, amount: int = 1) -> None:
        with self._lock:
            self._count += amount

    def reset(self) -> None:
        with self._lock:
            self._count = 0
            self._started = time.perf_counter()

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    @property
    def elapsed_s(self) -> float:
        with self._lock:
            return time.perf_counter() - self._started

    @property
    def hz(self) -> float:
        with self._lock:
            elapsed = time.perf_counter() - self._started
            return self._count / elapsed if elapsed > 0 else 0.0


class Gauge:
    """Current value plus its high-water mark.

    The high-water mark is the interesting half for a queue-depth gauge: a
    queue that momentarily hit its cap and shed packets reads as empty again
    by the time anyone polls it, so `peak` is what actually reveals
    backpressure after the fact.
    """

    __slots__ = ("_value", "_peak", "_lock", "name")

    def __init__(self, name: str):
        self.name = name
        self._value = 0
        self._peak = 0
        self._lock = threading.Lock()

    def set(self, value: int) -> None:
        with self._lock:
            self._value = value
            if value > self._peak:
                self._peak = value

    def add(self, delta: int) -> None:
        with self._lock:
            self._value += delta
            if self._value > self._peak:
                self._peak = self._value

    @property
    def value(self) -> int:
        with self._lock:
            return self._value

    @property
    def peak(self) -> int:
        with self._lock:
            return self._peak

    def reset(self) -> None:
        with self._lock:
            self._value = 0
            self._peak = 0


class ContentionProbe:
    """Wraps a lock and measures how long acquisitions have to wait.

    See the module docstring for why the fast path is a non-blocking acquire:
    the uncontended case must not pay for two clock reads, or the probe's own
    cost would show up as contention in the numbers.
    """

    def __init__(self, name: str, lock: threading.Lock | None = None):
        self.name = name
        self._lock = lock if lock is not None else threading.Lock()
        self._acquisitions = Counter(f"{name}.acquisitions")
        self._contended = Counter(f"{name}.contended")
        self._waits_ns: list[float] = []
        self._waits_lock = threading.Lock()

    @contextmanager
    def guard(self) -> Iterator[None]:
        self._acquisitions.increment()
        if self._lock.acquire(False):
            try:
                yield
            finally:
                self._lock.release()
            return

        # Contended: only now is a clock read worth its cost.
        self._contended.increment()
        start = time.perf_counter_ns()
        self._lock.acquire()
        wait_ns = float(time.perf_counter_ns() - start)
        with self._waits_lock:
            self._waits_ns.append(wait_ns)
        try:
            yield
        finally:
            self._lock.release()

    @property
    def raw_lock(self) -> threading.Lock:
        """The wrapped lock, for code paths that must acquire it without
        going through the probe (a `Condition`, say)."""
        return self._lock

    @property
    def acquisitions(self) -> int:
        return self._acquisitions.value

    @property
    def contended(self) -> int:
        return self._contended.value

    @property
    def contention_rate(self) -> float:
        total = self._acquisitions.value
        return self._contended.value / total if total else 0.0

    def wait_stats(self) -> LatencyStats:
        with self._waits_lock:
            samples = list(self._waits_ns)
        return compute_stats(samples)

    def reset(self) -> None:
        self._acquisitions.reset()
        self._contended.reset()
        with self._waits_lock:
            self._waits_ns.clear()


@dataclass
class DiagnosticsSnapshot:
    """A point-in-time read of every instrument. Plain data, so it can be
    serialized into the JSON export or rendered into a table without the
    renderer needing to know about the live objects."""

    label: str
    counters: dict[str, int] = field(default_factory=dict)
    rates: dict[str, float] = field(default_factory=dict)
    gauges: dict[str, tuple[int, int]] = field(default_factory=dict)  # name -> (current, peak)
    contention: dict[str, dict[str, float]] = field(default_factory=dict)
    remote: dict[str, int] = field(default_factory=dict)  # counters read from the EMBER process

    def as_key_values(self) -> dict[str, object]:
        """Flattens into the {metric: value} shape reporting.Section wants."""
        out: dict[str, object] = {}
        for name, value in sorted(self.counters.items()):
            out[f"counter.{name}"] = f"{value:,}"
        for name, value in sorted(self.rates.items()):
            out[f"rate.{name}"] = f"{value:,.1f} Hz"
        for name, (current, peak) in sorted(self.gauges.items()):
            out[f"gauge.{name}"] = f"{current:,} (peak {peak:,})"
        for name, values in sorted(self.contention.items()):
            out[f"lock.{name}"] = (
                f"{values['contended']:,.0f}/{values['acquisitions']:,.0f} contended "
                f"({values['contention_rate'] * 100:.2f}%), "
                f"mean wait {values['mean_wait_us']:.3f}us, "
                f"p99 wait {values['p99_wait_us']:.3f}us"
            )
        for name, value in sorted(self.remote.items()):
            out[f"ember.{name}"] = f"{value:,}"
        return out


class RuntimeDiagnostics:
    """Registry of every instrument for one runtime arm.

    Instruments are created on first use (`counter("x")` returns the same
    object every time), so instrumentation can be added at a call site without
    threading a registration through the constructor — which matters because
    the call sites are spread across a client, a driver, an agent and a relay.
    """

    def __init__(self, label: str):
        self.label = label
        self._lock = threading.Lock()
        self._counters: dict[str, Counter] = {}
        self._rates: dict[str, RateMeter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._contention: dict[str, ContentionProbe] = {}
        self._remote: dict[str, int] = {}

    def counter(self, name: str) -> Counter:
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(name)
            return self._counters[name]

    def rate(self, name: str) -> RateMeter:
        with self._lock:
            if name not in self._rates:
                self._rates[name] = RateMeter(name)
            return self._rates[name]

    def gauge(self, name: str) -> Gauge:
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(name)
            return self._gauges[name]

    def contention_probe(self, name: str, lock: threading.Lock | None = None) -> ContentionProbe:
        with self._lock:
            if name not in self._contention:
                self._contention[name] = ContentionProbe(name, lock)
            return self._contention[name]

    def set_remote_counters(self, counters: dict[str, int]) -> None:
        """Records counters read out of the EMBER process (its STATS line).

        Kept in their own namespace rather than merged into `_counters`: they
        are produced by a different process with a different clock and reset
        lifecycle, and conflating them with Python-side counters would make an
        `commands_received` figure ambiguous about which side counted it.
        """
        with self._lock:
            self._remote = dict(counters)

    def snapshot(self) -> DiagnosticsSnapshot:
        with self._lock:
            counters = dict(self._counters)
            rates = dict(self._rates)
            gauges = dict(self._gauges)
            contention = dict(self._contention)
            remote = dict(self._remote)

        snapshot = DiagnosticsSnapshot(label=self.label, remote=remote)
        snapshot.counters = {name: c.value for name, c in counters.items()}
        snapshot.rates = {name: r.hz for name, r in rates.items()}
        snapshot.gauges = {name: (g.value, g.peak) for name, g in gauges.items()}
        for name, probe in contention.items():
            stats = probe.wait_stats()
            snapshot.contention[name] = {
                "acquisitions": float(probe.acquisitions),
                "contended": float(probe.contended),
                "contention_rate": probe.contention_rate,
                "mean_wait_us": stats.mean_ns / 1000.0,
                "p99_wait_us": stats.p99_ns / 1000.0,
                "max_wait_us": stats.max_ns / 1000.0,
            }
        return snapshot

    def reset(self) -> None:
        with self._lock:
            instruments = (
                list(self._counters.values())
                + list(self._rates.values())
                + list(self._gauges.values())
                + list(self._contention.values())
            )
            self._remote = {}
        for instrument in instruments:
            instrument.reset()


# ---------------------------------------------------------------------------
# Shared metric names
#
# String keys shared by the baseline client, the hybrid client and the
# reporting layer. Constants rather than literals so a typo at one call site
# produces a NameError instead of a metric that silently reads zero forever.
# ---------------------------------------------------------------------------

COMMANDS_SENT = "commands_sent"
COMMANDS_ACKED = "commands_acked"
COMMANDS_REJECTED = "commands_rejected"
COMMANDS_TIMED_OUT = "commands_timed_out"
TELEMETRY_RECEIVED = "telemetry_received"
TELEMETRY_DECODE_FAILURES = "telemetry_decode_failures"
FRAMES_CORRUPT = "frames_corrupt"
BYTES_SENT = "bytes_sent"
BYTES_RECEIVED = "bytes_received"
RECONNECTS = "reconnects"

TASK_THROUGHPUT = "task_throughput"
TELEMETRY_LOOP = "telemetry_loop"

INBOUND_QUEUE_DEPTH = "inbound_queue_depth"
PENDING_COMMANDS = "pending_commands"

TELEMETRY_CACHE_LOCK = "telemetry_cache"
DISPATCH_LOCK = "coordinator_dispatch"
