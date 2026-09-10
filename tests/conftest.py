"""Fixtures for the hybrid-pipeline suite.

Every fixture that needs the compiled EMBER runtime skips rather than fails
when no C++ toolchain is available, mirroring how `agents/tests/conftest.py`
stubs whisper/cv2/ultralytics/llama_cpp instead of failing the whole suite.
A machine without a compiler should still be able to run everything that does
not genuinely need one — and a good deal of this file's coverage (wire-format
round trips, framer resync, the baseline runtime) does not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Importable whether pytest is invoked from the repository root or from
# tests/, without depending on rootdir inference.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.bench_core.ember_harness import (  # noqa: E402
    EmberBenchServer,
    find_free_port,
    find_or_build_server_binary,
)
from benchmarks.bench_core.rigs import BaselineRig, HybridRig  # noqa: E402
from benchmarks.bench_core.workload import WorkloadConfig  # noqa: E402

_SKIP_REASON = (
    "No C++ toolchain (g++) and no prebuilt ember_bench_harness_server found. "
    "Set EMBER_BENCH_SERVER_BIN to a prebuilt binary, build edge/'s "
    "ember_bench_harness_server CMake target, or install a C++20 compiler to run "
    "the hybrid-runtime tests."
)


@pytest.fixture(scope="session")
def ember_bench_binary():
    """Located or built once per session; each test still gets its own
    subprocess and port from the fixtures below, so no test can observe
    another's counters or leave a connection behind."""
    try:
        binary = find_or_build_server_binary()
    except RuntimeError as exc:
        pytest.skip(f"EMBER benchmark harness failed to build: {exc}")
    if binary is None:
        pytest.skip(_SKIP_REASON)
    return binary


@pytest.fixture
def ember_server(ember_bench_binary):
    """A bare EMBER harness subprocess with no client attached.

    Command logging is on: these are correctness tests, where seeing what the
    C++ side actually received matters more than the printf cost the benchmark
    suite avoids.
    """
    server = EmberBenchServer(
        ember_bench_binary,
        port=find_free_port(),
        watchdog_ms=300,
        actuation_us=0,
        log_commands=True,
    )
    try:
        assert server.wait_ready(timeout=20.0), (
            "EMBER harness never printed READY:\n" + "\n".join(server.all_lines())
        )
        yield server
    finally:
        server.stop()


@pytest.fixture
def hybrid_rig(ember_bench_binary):
    """The full hybrid stack: EMBER subprocess, bridge client, robot driver,
    Coordinator with the four-agent pipeline."""
    rig = HybridRig(
        workload=WorkloadConfig(),
        binary_path=ember_bench_binary,
        actuation_us=0,
        watchdog_ms=300,
        command_timeout_s=3.0,
        log_commands=True,
    )
    rig.start()
    try:
        yield rig
    finally:
        rig.stop()


@pytest.fixture
def baseline_rig():
    """The pure-Python arm. Needs no toolchain, so tests that assert the two
    runtimes behave equivalently can still run one half of the comparison on a
    machine without a compiler."""
    rig = BaselineRig(workload=WorkloadConfig(), codec_name="json", transport="in_process")
    rig.start()
    try:
        yield rig
    finally:
        rig.stop()
