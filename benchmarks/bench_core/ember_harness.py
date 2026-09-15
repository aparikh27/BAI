"""Locates, builds and drives the compiled EMBER benchmark harness server
(`edge/tests/integration/bench_harness_server.cpp`).

Deliberately parallel to `agents/tests/integration/ember_test_server.py`
rather than importing it: that module builds and drives a *different* binary
(`ember_pipeline_test_server`) with a different argument form and a different
stdout vocabulary. The build strategy is the same three-step search, and the
reasoning behind `-static` and behind the mtime-keyed cache is the same — see
that module's docstring, which documents it at length — but sharing the code
would mean parameterizing it over both binaries' argument shapes for no
benefit beyond avoiding forty lines.

If no C++ toolchain and no prebuilt binary are available,
`find_or_build_server_binary()` returns None and callers degrade to running
the baseline arm alone, exactly as the integration suite skips rather than
fails.
"""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EDGE_DIR = REPO_ROOT / "edge"

_CACHE_DIR = Path(tempfile.gettempdir()) / "ember_bench_harness_cache"

_GXX_SOURCES = [
    "run/log.cpp",
    "run/time.cpp",
    "run/runtime.cpp",
    "messages/coordinator.cpp",
    "messages/publisher.cpp",
    "messages/subscriber.cpp",
    "schedule/task.cpp",
    "schedule/scheduler.cpp",
    "bridge/tcp_socket.cpp",
    "bridge/ember_bridge_adapter.cpp",
    "tests/integration/bench_harness_server.cpp",
]

BINARY_STEM = "ember_bench_harness_server"


def find_free_port() -> int:
    """Binds an ephemeral port and releases it immediately.

    The gap between release and the C++ server's own bind() is a real (small)
    race, accepted for the same reason the integration suite accepts it: a
    collision produces a loud "never reached READY" failure on one run, not a
    silently wrong measurement.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _candidate_cmake_binaries():
    names = (f"{BINARY_STEM}.exe", BINARY_STEM)
    for build_dir in ("build", "cmake-build-debug", "cmake-build-release", "out/build"):
        for sub in ("", "Debug", "Release", "RelWithDebInfo"):
            base = EDGE_DIR / build_dir / sub if sub else EDGE_DIR / build_dir
            for name in names:
                yield base / name


def _direct_gxx_build(force: bool = False) -> Path | None:
    gxx = shutil.which("g++")
    if gxx is None:
        return None

    binary_name = f"{BINARY_STEM}.exe" if os.name == "nt" else BINARY_STEM
    out_path = _CACHE_DIR / binary_name
    source_paths = [EDGE_DIR / s for s in _GXX_SOURCES]

    if out_path.exists() and not force:
        newest = max(p.stat().st_mtime for p in source_paths if p.exists())
        if out_path.stat().st_mtime >= newest:
            return out_path

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # -O2, not the integration suite's -O1: these numbers are meant to
    # characterize EMBER as it would ship, and a debug-ish optimization level
    # would understate it by a factor that has nothing to do with the design.
    # -static for the MinGW runtime-DLL reason documented in
    # agents/tests/integration/ember_test_server.py.
    cmd = [
        gxx,
        "-std=c++20",
        "-O2",
        "-static",
        "-I",
        str(EDGE_DIR),
        *[str(p) for p in source_paths],
        "-o",
        str(out_path),
    ]
    # -lwinmm: bench_harness_server calls timeBeginPeriod to request a 1ms
    # timer quantum (see its header comment on --timer-resolution-ms).
    cmd += ["-lws2_32", "-lwinmm", "-lpthread"] if os.name == "nt" else ["-lpthread"]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to build ember_bench_harness_server via g++:\n"
            f"command: {' '.join(cmd)}\n\nstderr:\n{result.stderr}"
        )
    return out_path


def find_or_build_server_binary(force_rebuild: bool = False) -> Path | None:
    env_path = os.environ.get("EMBER_BENCH_SERVER_BIN")
    if env_path:
        candidate = Path(env_path)
        return candidate if candidate.is_file() else None

    if not force_rebuild:
        for candidate in _candidate_cmake_binaries():
            if candidate.is_file():
                return candidate

    return _direct_gxx_build(force=force_rebuild)


# Scientific notation is matched deliberately: the harness prints floats with
# %.9g, so a float32 near its maximum comes back as "3.40282347e+38". A pattern
# that stopped at the mantissa would silently parse that as 3.40282347 -- a
# wrong value rather than a parse failure, which is the worst way for a
# measurement harness to fail.
_STATS_TOKEN = re.compile(r"(\w+)=([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:inf|nan))")


def parse_key_values(line: str) -> dict[str, float | int]:
    """Parses a `KEY=value KEY=value` status line into a dict of numbers.

    Only numeric values are extracted; the leading verb (STATS, SCHED, ...)
    carries no `=` and is skipped by the pattern.

    Integral literals become `int`, not `float`. That is not cosmetic: a
    steady-clock timestamp in nanoseconds is around 1.7e18, which needs 61
    bits of mantissa and a float64 has 53 - so parsing one as a float loses
    the low digits and an exact-equality assertion against it fails for
    reasons that have nothing to do with the wire format.
    """
    parsed: dict[str, float | int] = {}
    for key, value in _STATS_TOKEN.findall(line):
        parsed[key] = int(value) if value.lstrip("+-").isdigit() else float(value)
    return parsed


class EmberBenchServer:
    """Subprocess wrapper for the compiled harness.

    stdout is drained by a dedicated thread into a line history. That thread
    is not optional: the harness writes unbuffered, and a full OS pipe buffer
    would block a C++ thread mid-benchmark and produce a latency spike caused
    entirely by this process not reading fast enough.
    """

    def __init__(
        self,
        binary_path: Path,
        port: int,
        watchdog_ms: int = 250,
        actuation_us: int = 0,
        runtime_hz: int = 200,
        probe_period_us: int = 5000,
        send_timeout_ms: int = 50,
        recv_timeout_ms: int = 200,
        max_telemetry_buffer: int = 256,
        timer_resolution_ms: int = 1,
        log_commands: bool = False,
    ):
        self.port = port
        args = [
            str(binary_path),
            f"--port={port}",
            f"--watchdog-ms={watchdog_ms}",
            f"--actuation-us={actuation_us}",
            f"--runtime-hz={runtime_hz}",
            f"--probe-period-us={probe_period_us}",
            f"--send-timeout-ms={send_timeout_ms}",
            f"--recv-timeout-ms={recv_timeout_ms}",
            f"--max-telemetry-buffer={max_telemetry_buffer}",
            f"--timer-resolution-ms={timer_resolution_ms}",
        ]
        if log_commands:
            args.append("--log-commands")

        self._proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._lines: list[str] = []
        self._lines_lock = threading.Lock()
        self._reader = threading.Thread(target=self._drain_stdout, name="ember-bench-stdout", daemon=True)
        self._reader.start()

    # ---- stdout ------------------------------------------------------

    def _drain_stdout(self) -> None:
        assert self._proc.stdout is not None
        for raw in self._proc.stdout:
            with self._lines_lock:
                self._lines.append(raw.rstrip("\n"))

    def all_lines(self) -> list[str]:
        with self._lines_lock:
            return list(self._lines)

    def line_count(self) -> int:
        with self._lines_lock:
            return len(self._lines)

    def wait_for_line(self, predicate, timeout: float = 5.0, since: int = 0) -> str | None:
        """Waits for a line satisfying `predicate`, searching from index
        `since` onward.

        `since` is what makes this usable for request/response control
        commands: without it, a second `STATS` would immediately match the
        first one's reply still sitting in the history. Callers capture
        `line_count()` before sending and pass it here.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._lines_lock:
                for line in self._lines[since:]:
                    if predicate(line):
                        return line
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.005)

    def wait_ready(self, timeout: float = 15.0) -> bool:
        return self.wait_for_line(lambda line: line.startswith("READY"), timeout=timeout) is not None

    # ---- stdin control ------------------------------------------------

    def send_control(self, line: str) -> None:
        if self._proc.poll() is not None:
            raise RuntimeError("ember bench server process has exited")
        assert self._proc.stdin is not None
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()

    def _request(self, control: str, prefix: str, timeout: float = 10.0) -> str | None:
        mark = self.line_count()
        self.send_control(control)
        return self.wait_for_line(lambda line: line.startswith(prefix), timeout=timeout, since=mark)

    def stats(self, timeout: float = 10.0) -> dict[str, float | int]:
        line = self._request("STATS", "STATS ", timeout=timeout)
        return parse_key_values(line) if line else {}

    def reset_stats(self) -> None:
        self._request("RESET_STATS", "STATS_RESET")

    def scheduler_stats(self, timeout: float = 10.0) -> dict[str, float | int]:
        line = self._request("SCHED_STATS", "SCHED ", timeout=timeout)
        return parse_key_values(line) if line else {}

    def set_command_logging(self, enabled: bool) -> None:
        self._request(f"LOG_COMMANDS {'ON' if enabled else 'OFF'}", "LOG_COMMANDS ")

    def set_actuation_us(self, value: int) -> None:
        self._request(f"ACTUATION_US {value}", "ACTUATION_US ")

    def start_telemetry(self, hz: float, value: float = 1.0) -> None:
        self._request(f"TELEMETRY_START {hz} {value}", "TELEMETRY_STARTED")

    def stop_telemetry(self) -> None:
        self._request("TELEMETRY_STOP", "TELEMETRY_STOPPED")

    def telemetry_burst(
        self, count: int, interval_us: int = 0, timeout: float = 60.0
    ) -> dict[str, float | int]:
        line = self._request(f"TELEMETRY_BURST {count} {interval_us}", "BURST_DONE", timeout=timeout)
        return parse_key_values(line) if line else {}

    def inject_battery_low(self, percentage: int, voltage: float) -> None:
        self._request(f"INJECT_BATTERY_LOW {percentage} {voltage}", "INJECTED ")

    def inject_fault(self, code: int, component: str, description: str) -> None:
        self._request(f"INJECT_FAULT {code} {component} {description}", "INJECTED ")

    def inject_thermal(self, component: str, temperature: float) -> None:
        self._request(f"INJECT_THERMAL {component} {temperature}", "INJECTED ")

    # ---- codec parity probes ------------------------------------------

    def pack_telemetry(
        self,
        subsystem: int,
        sequence: int,
        timestamp_ns: int,
        value_a: float,
        value_b: float,
        code: int,
        text: str,
    ) -> bytes | None:
        """Returns the framed packet EMBER's own encoder produces, so a test
        can compare it byte-for-byte against `ember_wire.pack_frame`."""
        line = self._request(
            f"PACK_TELEMETRY {subsystem} {sequence} {timestamp_ns} {value_a!r} {value_b!r} {code} {text}",
            "PACKED ",
        )
        return bytes.fromhex(line.split(" ", 1)[1]) if line else None

    def pack_ack(self, sequence: int, result: int, request_id_hash: int) -> bytes | None:
        line = self._request(f"PACK_ACK {sequence} {result} {request_id_hash}", "PACKED ")
        return bytes.fromhex(line.split(" ", 1)[1]) if line else None

    def unpack_command(self, packet: bytes) -> dict[str, float | int] | None:
        """Hands EMBER a Python-encoded packet and returns the field values it
        decoded, or None if it rejected the frame."""
        mark = self.line_count()
        self.send_control(f"UNPACK_COMMAND {packet.hex()}")
        line = self.wait_for_line(
            lambda text: text.startswith("UNPACKED") or text.startswith("UNPACK_FAILED"),
            timeout=10.0,
            since=mark,
        )
        if line is None or line.startswith("UNPACK_FAILED"):
            return None
        return parse_key_values(line)

    # ---- lifecycle ----------------------------------------------------

    @property
    def pid(self) -> int:
        return self._proc.pid

    def is_alive(self) -> bool:
        return self._proc.poll() is None

    def kill(self) -> None:
        """Terminates without a clean shutdown, for tests that need the peer
        to vanish rather than close politely."""
        if self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait(timeout=5.0)

    def stop(self, timeout: float = 10.0) -> None:
        if self._proc.poll() is not None:
            return
        try:
            self.send_control("QUIT")
            self._proc.wait(timeout=timeout)
        except Exception:
            self._proc.kill()
            self._proc.wait(timeout=timeout)
