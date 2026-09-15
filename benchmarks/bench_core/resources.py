"""CPU, memory and garbage-collection accounting for the benchmark suite.

`psutil` is used when installed, but is not a dependency of this project (it
is in neither pyproject.toml nor requirements.txt), so every probe here has a
stdlib fallback: ctypes against psapi/kernel32 on Windows, /proc on Linux.
A resource comparison that silently reports nothing on the machine the
project actually develops on would be worse than useless.

The hybrid arm is deliberately measured across *both* processes. Moving work
into a C++ runtime does not make it free — it moves it into a second OS
process, and a report that measured only the Python interpreter would show a
memory "saving" that is really just a memory relocation. `ResourceSampler`
therefore takes a list of PIDs and reports each one plus the total.
"""

from __future__ import annotations

import ctypes
import gc
import os
import platform
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Sequence

try:  # optional, and absent on this project's default environment
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - depends on the host environment
    psutil = None  # type: ignore

_IS_WINDOWS = platform.system() == "Windows"


# ---------------------------------------------------------------------------
# Per-process probes
# ---------------------------------------------------------------------------


class _WindowsProcessProbe:
    """RSS and CPU time for an arbitrary PID via psapi/kernel32.

    Used when psutil is unavailable. `time.process_time()` would cover CPU for
    *this* process only, and the hybrid arm's whole point is that a second
    process is doing the work, so the Win32 calls are worth the ctypes
    boilerplate.
    """

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    PROCESS_VM_READ = 0x0010

    class _ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_uint32),
            ("PageFaultCount", ctypes.c_uint32),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    class _Filetime(ctypes.Structure):
        _fields_ = [("dwLowDateTime", ctypes.c_uint32), ("dwHighDateTime", ctypes.c_uint32)]

    def __init__(self, pid: int):
        self._pid = pid
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._psapi = ctypes.WinDLL("psapi", use_last_error=True)
        # PROCESS_QUERY_LIMITED_INFORMATION is enough for both calls and, unlike
        # PROCESS_QUERY_INFORMATION, is grantable across integrity levels — so
        # this keeps working if the harness ever launches EMBER differently.
        access = self.PROCESS_QUERY_LIMITED_INFORMATION | self.PROCESS_VM_READ
        self._handle = self._kernel32.OpenProcess(access, False, pid)
        if not self._handle:
            # Retry without VM_READ: GetProcessMemoryInfo accepts a
            # limited-information handle on Vista+, and dropping VM_READ is
            # what makes the open succeed for a process we don't own.
            self._handle = self._kernel32.OpenProcess(
                self.PROCESS_QUERY_LIMITED_INFORMATION, False, pid
            )

    @property
    def available(self) -> bool:
        return bool(self._handle)

    def rss_bytes(self) -> int | None:
        if not self._handle:
            return None
        counters = self._ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        ok = self._psapi.GetProcessMemoryInfo(
            self._handle, ctypes.byref(counters), ctypes.sizeof(counters)
        )
        if not ok:
            return None
        return int(counters.WorkingSetSize)

    def cpu_seconds(self) -> float | None:
        if not self._handle:
            return None
        creation = self._Filetime()
        exit_time = self._Filetime()
        kernel = self._Filetime()
        user = self._Filetime()
        ok = self._kernel32.GetProcessTimes(
            self._handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel),
            ctypes.byref(user),
        )
        if not ok:
            return None

        def to_seconds(ft: "_WindowsProcessProbe._Filetime") -> float:
            # FILETIME counts 100-nanosecond intervals.
            return ((ft.dwHighDateTime << 32) | ft.dwLowDateTime) / 1e7

        return to_seconds(kernel) + to_seconds(user)

    def close(self) -> None:
        if self._handle:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


class _PosixProcessProbe:
    """RSS and CPU time for a PID from /proc. Linux-shaped; on a POSIX system
    without /proc both readers return None and the sampler degrades to
    reporting no resource data rather than raising."""

    def __init__(self, pid: int):
        self._pid = pid
        self._clock_ticks = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        self._page_size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096

    @property
    def available(self) -> bool:
        return os.path.exists(f"/proc/{self._pid}/statm")

    def rss_bytes(self) -> int | None:
        try:
            with open(f"/proc/{self._pid}/statm", "r", encoding="ascii") as handle:
                fields = handle.read().split()
            return int(fields[1]) * self._page_size
        except (OSError, IndexError, ValueError):
            return None

    def cpu_seconds(self) -> float | None:
        try:
            with open(f"/proc/{self._pid}/stat", "r", encoding="ascii") as handle:
                content = handle.read()
            # The comm field can contain spaces and parentheses, so fields are
            # counted from after the closing paren, not from the line start.
            fields = content[content.rindex(")") + 2 :].split()
            utime, stime = int(fields[11]), int(fields[12])
            return (utime + stime) / self._clock_ticks
        except (OSError, IndexError, ValueError):
            return None

    def close(self) -> None:
        return None


class _PsutilProcessProbe:
    def __init__(self, pid: int):
        self._proc = psutil.Process(pid)

    @property
    def available(self) -> bool:
        try:
            return self._proc.is_running()
        except Exception:
            return False

    def rss_bytes(self) -> int | None:
        try:
            return int(self._proc.memory_info().rss)
        except Exception:
            return None

    def cpu_seconds(self) -> float | None:
        try:
            times = self._proc.cpu_times()
            return float(times.user + times.system)
        except Exception:
            return None

    def close(self) -> None:
        return None


def make_process_probe(pid: int):
    """Best available probe for `pid`, or None if this platform offers none."""
    if psutil is not None:
        try:
            probe = _PsutilProcessProbe(pid)
            if probe.available:
                return probe
        except Exception:
            pass

    if _IS_WINDOWS:
        probe = _WindowsProcessProbe(pid)
        return probe if probe.available else None

    probe = _PosixProcessProbe(pid)
    return probe if probe.available else None


# ---------------------------------------------------------------------------
# Garbage collection
# ---------------------------------------------------------------------------


@dataclass
class GcReport:
    collections_gen0: int = 0
    collections_gen1: int = 0
    collections_gen2: int = 0
    total_pause_ns: float = 0.0
    max_pause_ns: float = 0.0
    pause_count: int = 0

    @property
    def total_collections(self) -> int:
        return self.collections_gen0 + self.collections_gen1 + self.collections_gen2

    @property
    def mean_pause_ns(self) -> float:
        return self.total_pause_ns / self.pause_count if self.pause_count else 0.0


class GcProbe:
    """Measures garbage-collection pauses over a region of code.

    Uses `gc.callbacks`, which fires on the thread performing the collection,
    around the collection itself — so the start/stop delta is the actual stop-
    the-world pause, not an estimate derived from counter deltas. GC pause is
    the mechanism behind a specific failure this bridge exists to survive (a
    Python pause longer than EMBER's 250ms watchdog deadline), so it is worth
    measuring directly rather than inferring.

    Nested collections are not possible — CPython's collector is not
    reentrant — so a single start timestamp is sufficient state.
    """

    def __init__(self) -> None:
        self._report = GcReport()
        self._start_ns: int | None = None
        self._installed = False

    def _callback(self, phase: str, info: dict) -> None:
        if phase == "start":
            self._start_ns = time.perf_counter_ns()
            return

        generation = info.get("generation", 0)
        if generation == 0:
            self._report.collections_gen0 += 1
        elif generation == 1:
            self._report.collections_gen1 += 1
        else:
            self._report.collections_gen2 += 1

        if self._start_ns is not None:
            pause = float(time.perf_counter_ns() - self._start_ns)
            self._report.total_pause_ns += pause
            self._report.max_pause_ns = max(self._report.max_pause_ns, pause)
            self._report.pause_count += 1
            self._start_ns = None

    def __enter__(self) -> "GcProbe":
        self._report = GcReport()
        self._start_ns = None
        gc.callbacks.append(self._callback)
        self._installed = True
        return self

    def __exit__(self, *exc_info) -> None:
        if self._installed:
            try:
                gc.callbacks.remove(self._callback)
            except ValueError:
                pass
            self._installed = False

    @property
    def report(self) -> GcReport:
        return self._report


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


@dataclass
class ProcessResourceSummary:
    label: str
    pid: int
    samples: int = 0
    rss_start_bytes: int = 0
    rss_peak_bytes: int = 0
    rss_end_bytes: int = 0
    cpu_seconds: float = 0.0
    wall_seconds: float = 0.0
    available: bool = True

    @property
    def cpu_percent(self) -> float:
        """CPU seconds consumed per wall second, as a percentage. Can exceed
        100 for a multi-threaded process — EMBER runs an accept, reader,
        writer, runtime and per-subsystem worker thread, so values above 100
        are expected and meaningful rather than a bug."""
        if self.wall_seconds <= 0.0:
            return 0.0
        return 100.0 * self.cpu_seconds / self.wall_seconds

    @property
    def rss_growth_bytes(self) -> int:
        return self.rss_end_bytes - self.rss_start_bytes


@dataclass
class ResourceReport:
    processes: list[ProcessResourceSummary] = field(default_factory=list)
    gc: GcReport = field(default_factory=GcReport)
    wall_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def total_rss_peak_bytes(self) -> int:
        return sum(p.rss_peak_bytes for p in self.processes)

    @property
    def total_cpu_seconds(self) -> float:
        return sum(p.cpu_seconds for p in self.processes)


class ResourceSampler:
    """Samples RSS/CPU for one or more processes on a background thread.

    Sampling on a thread rather than only at the boundaries is what makes the
    *peak* meaningful: an allocation spike that has been freed by the time the
    workload finishes is invisible to a start/end pair, and a peak RSS that
    briefly doubles is exactly the kind of thing worth catching on an embedded
    target.

    The sampler's own overhead is bounded by `interval_s` (50ms default) and
    it takes no locks the workload can contend on, so it does not perturb the
    latency measurements running alongside it.
    """

    def __init__(
        self,
        pids: Sequence[tuple[str, int]],
        interval_s: float = 0.05,
    ):
        self._targets = list(pids)
        self._interval_s = interval_s
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._summaries: dict[int, ProcessResourceSummary] = {}
        self._probes: dict[int, object] = {}
        self._cpu_start: dict[int, float] = {}
        self._wall_start = 0.0
        self._notes: list[str] = []

    def __enter__(self) -> "ResourceSampler":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

    def start(self) -> None:
        self._wall_start = time.perf_counter()
        for label, pid in self._targets:
            probe = make_process_probe(pid)
            summary = ProcessResourceSummary(label=label, pid=pid)
            if probe is None:
                summary.available = False
                self._notes.append(
                    f"no resource probe available for {label} (pid {pid}) on this platform"
                )
            else:
                self._probes[pid] = probe
                rss = probe.rss_bytes() or 0
                summary.rss_start_bytes = rss
                summary.rss_peak_bytes = rss
                summary.rss_end_bytes = rss
                self._cpu_start[pid] = probe.cpu_seconds() or 0.0
            self._summaries[pid] = summary

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="resource-sampler", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._sample_once()

    def _sample_once(self) -> None:
        for pid, probe in self._probes.items():
            summary = self._summaries[pid]
            rss = probe.rss_bytes()  # type: ignore[attr-defined]
            if rss is None:
                continue
            summary.samples += 1
            summary.rss_end_bytes = rss
            summary.rss_peak_bytes = max(summary.rss_peak_bytes, rss)

    def stop(self) -> ResourceReport:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
            self._thread = None

        # One final sample so the "end" figures reflect the state at stop(),
        # not up to `interval_s` before it.
        self._sample_once()

        wall = time.perf_counter() - self._wall_start
        for pid, probe in self._probes.items():
            summary = self._summaries[pid]
            cpu_now = probe.cpu_seconds()  # type: ignore[attr-defined]
            if cpu_now is not None:
                summary.cpu_seconds = max(0.0, cpu_now - self._cpu_start.get(pid, 0.0))
            summary.wall_seconds = wall
            probe.close()  # type: ignore[attr-defined]
        self._probes.clear()

        return ResourceReport(
            processes=list(self._summaries.values()),
            wall_seconds=wall,
            notes=list(self._notes),
        )


def format_bytes(value: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(value) < 1024.0 or unit == "GiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} GiB"


def probe_backend_name() -> str:
    """Which implementation the resource numbers came from — recorded in the
    report so a reader can tell psutil-quality data from the ctypes fallback."""
    if psutil is not None:
        return f"psutil {getattr(psutil, '__version__', '?')}"
    if _IS_WINDOWS:
        return "ctypes (psapi/kernel32)"
    if os.path.exists("/proc/self/statm"):
        return "procfs"
    return "unavailable"


def interpreter_allocation_note() -> str:
    """Python-side allocator context worth recording alongside RSS: the
    interpreter's own arena behaviour is a large part of why the pure-Python
    arm's footprint moves the way it does."""
    return (
        f"gc thresholds={gc.get_threshold()}, "
        f"gc enabled={gc.isenabled()}, "
        f"allocator={'pymalloc' if sys.implementation.name == 'cpython' else 'n/a'}"
    )
