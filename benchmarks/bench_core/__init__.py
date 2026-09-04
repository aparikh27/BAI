"""Shared measurement, reporting and test-rig machinery for the comparative
benchmark suite.

Module map:

    stats.py       latency statistics (Python twin of edge/benchmarks/bench_framework.hpp)
    reporting.py   ASCII console tables + Markdown export
    resources.py   CPU / RSS / garbage-collection sampling
    diagnostics.py live runtime counters (throughput, drops, queue depth, lock contention)
    baseline.py    the pure-Python legacy runtime the hybrid one is measured against
    ember_harness.py  locate/build/drive the compiled EMBER benchmark server
    workload.py    the shared perception-to-execution workload both arms run
    rigs.py        BaselineRig / HybridRig — one interface over both arms
    transport.py   serialization and event-handling micro-benchmarks
"""
