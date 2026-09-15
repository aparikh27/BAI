"""Latency statistics for the comparative benchmark suite.

Deliberately a direct twin of `edge/benchmarks/bench_framework.hpp`: the same
linear-interpolated ("R7") percentile definition, the same sample-standard-
deviation denominator (n-1), the same nanosecond unit. A Python row and a C++
row in the same report have to be comparable, and two different percentile
conventions would put a silent few-percent skew between the two arms of every
comparison in this suite.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

# Windows' clock granularity means a single fast operation frequently measures
# as exactly 0ns. That is a resolution artifact, not a real measurement, so very
# fast operations are timed in batches (see `time_batched`) rather than reported
# as instantaneous.


def percentile(sorted_samples: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile over an already-sorted sample set.

    Matches `ember::bench::percentile` in bench_framework.hpp, which in turn
    matches numpy's default. Callers must sort first — this is called inside
    aggregation loops where re-sorting per percentile would dominate the cost.
    """
    if not sorted_samples:
        return 0.0
    if len(sorted_samples) == 1:
        return float(sorted_samples[0])

    rank = (pct / 100.0) * (len(sorted_samples) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(sorted_samples[low])

    frac = rank - low
    return float(sorted_samples[low] + (sorted_samples[high] - sorted_samples[low]) * frac)


@dataclass
class LatencyStats:
    """Aggregated latency distribution, in nanoseconds throughout.

    `wcet_spikes` counts samples exceeding `spike_threshold_ns`; the two
    together are the suite's determinism signal. A distribution can have an
    excellent mean and still be unusable for real-time control if it spikes,
    which is exactly the property offloading to EMBER is supposed to improve,
    so it is reported as a first-class field rather than left to be inferred
    from max alone.
    """

    count: int = 0
    min_ns: float = 0.0
    max_ns: float = 0.0
    mean_ns: float = 0.0
    p50_ns: float = 0.0
    p95_ns: float = 0.0
    p99_ns: float = 0.0
    p999_ns: float = 0.0
    stddev_ns: float = 0.0
    spike_threshold_ns: float = 0.0
    wcet_spikes: int = 0

    @property
    def jitter_ns(self) -> float:
        """Peak-to-peak spread. The figure that bounds a control loop's worst
        case, as opposed to stddev, which describes its typical one."""
        return self.max_ns - self.min_ns

    @property
    def determinism_index(self) -> float:
        """P99 / P50. 1.0 is perfectly deterministic; the larger it gets, the
        further the tail is from the typical case. Unitless, so it compares
        directly across two runtimes whose absolute latencies differ by an
        order of magnitude."""
        if self.p50_ns <= 0.0:
            return 0.0
        return self.p99_ns / self.p50_ns

    @property
    def spike_rate(self) -> float:
        if self.count == 0:
            return 0.0
        return self.wcet_spikes / self.count


def compute_stats(samples_ns: Iterable[float], spike_factor: float = 10.0) -> LatencyStats:
    """Aggregates raw per-operation latencies (nanoseconds) into LatencyStats.

    `spike_factor` sets the WCET spike threshold as a multiple of the median.
    A multiple of the median rather than a fixed nanosecond budget is what
    makes the count meaningful across both arms: a 10x-median outlier means
    the same thing whether the median is 3us (EMBER) or 300us (pure Python),
    whereas any absolute threshold would flatter one arm by construction.
    """
    ordered = sorted(float(s) for s in samples_ns)
    stats = LatencyStats(count=len(ordered))
    if not ordered:
        return stats

    stats.min_ns = ordered[0]
    stats.max_ns = ordered[-1]
    stats.mean_ns = statistics.fmean(ordered)
    stats.stddev_ns = statistics.stdev(ordered) if len(ordered) > 1 else 0.0
    stats.p50_ns = percentile(ordered, 50.0)
    stats.p95_ns = percentile(ordered, 95.0)
    stats.p99_ns = percentile(ordered, 99.0)
    stats.p999_ns = percentile(ordered, 99.9)

    stats.spike_threshold_ns = stats.p50_ns * spike_factor
    if stats.spike_threshold_ns > 0.0:
        # Samples are sorted, so the spikes are a suffix — but a linear count
        # is clearer than a bisect here and runs once per benchmark row, not
        # per sample.
        stats.wcet_spikes = sum(1 for s in ordered if s > stats.spike_threshold_ns)

    return stats


@dataclass
class Comparison:
    """One baseline-vs-hybrid pairing, with the ratios a report actually
    quotes. Kept as a dataclass rather than computed in the renderer so the
    JSON export and the Markdown table can't drift apart."""

    name: str
    baseline: LatencyStats
    hybrid: LatencyStats
    unit_note: str = ""

    def _ratio(self, baseline_value: float, hybrid_value: float) -> float:
        """Speedup as baseline/hybrid: >1 means the hybrid runtime is faster.

        Returns 0.0 rather than raising or reporting infinity when the hybrid
        measurement is zero — a zero there means the clock could not resolve
        the operation, and printing "inf x faster" from a resolution artifact
        would be the single most misleading number this suite could emit.
        """
        if hybrid_value <= 0.0:
            return 0.0
        return baseline_value / hybrid_value

    @property
    def mean_speedup(self) -> float:
        return self._ratio(self.baseline.mean_ns, self.hybrid.mean_ns)

    @property
    def p50_speedup(self) -> float:
        return self._ratio(self.baseline.p50_ns, self.hybrid.p50_ns)

    @property
    def p99_speedup(self) -> float:
        return self._ratio(self.baseline.p99_ns, self.hybrid.p99_ns)

    @property
    def jitter_reduction(self) -> float:
        return self._ratio(self.baseline.jitter_ns, self.hybrid.jitter_ns)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "unit_note": self.unit_note,
            "baseline": vars(self.baseline) | {"jitter_ns": self.baseline.jitter_ns},
            "hybrid": vars(self.hybrid) | {"jitter_ns": self.hybrid.jitter_ns},
            "mean_speedup": self.mean_speedup,
            "p50_speedup": self.p50_speedup,
            "p99_speedup": self.p99_speedup,
            "jitter_reduction": self.jitter_reduction,
        }


@dataclass
class BenchRow:
    """One reported measurement. Mirrors `ember::bench::BenchRow` so the C++
    and Python tables share a column layout."""

    suite: str
    name: str
    arm: str = ""  # "Pure Python" / "Hybrid (Python+EMBER)" / "" for arm-agnostic rows
    latency: LatencyStats = field(default_factory=LatencyStats)
    throughput_value: float | None = None
    throughput_unit: str = ""
    notes: str = ""


def time_each_call(
    iterations: int,
    warmup: int,
    fn: Callable[[], object],
    spike_factor: float = 10.0,
) -> LatencyStats:
    """Times `iterations` individual calls, after `warmup` untimed ones.

    Warm-up is not optional bookkeeping here: the first calls into any Python
    path pay for bytecode specialization, lazily-built caches and cold branch
    predictors, and on the hybrid arm additionally for the TCP connection's
    first round trips. Folding those into the sample set would show up as a
    fat tail attributable to nothing in the code under test.
    """
    for _ in range(warmup):
        fn()

    clock = time.perf_counter_ns
    samples: list[float] = []
    samples_append = samples.append
    for _ in range(iterations):
        start = clock()
        fn()
        samples_append(clock() - start)

    return compute_stats(samples, spike_factor=spike_factor)


def time_batched(
    iterations: int,
    warmup: int,
    fn: Callable[[], object],
    batch_size: int = 100,
    spike_factor: float = 10.0,
) -> LatencyStats:
    """Times `iterations` calls in batches of `batch_size`, reporting the
    per-call average of each batch as one sample.

    For operations faster than the clock's granularity (pure-Python struct
    packing on Windows, where perf_counter_ns resolves to ~100ns), per-call
    timing produces a bimodal 0ns/100ns distribution that is an artifact of
    the clock rather than a property of the code. Batching trades the tail
    detail — a batch average hides an individual spike — for a mean and P50
    that are actually meaningful, so it is used only for the micro-benchmarks
    where the tail is not the question being asked.
    """
    for _ in range(warmup):
        fn()

    clock = time.perf_counter_ns
    batches = max(1, iterations // batch_size)
    samples: list[float] = []
    for _ in range(batches):
        start = clock()
        for _ in range(batch_size):
            fn()
        samples.append((clock() - start) / batch_size)

    return compute_stats(samples, spike_factor=spike_factor)


def throughput_from_stats(stats: LatencyStats) -> float:
    """Operations per second implied by the mean per-operation latency."""
    if stats.mean_ns <= 0.0:
        return 0.0
    return 1e9 / stats.mean_ns
