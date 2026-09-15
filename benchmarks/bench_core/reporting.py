"""Console (ASCII) and Markdown rendering for benchmark results.

Two renderers over one row model, so a table can never say one thing on the
terminal and another in the committed report. Output is plain ASCII rather
than box-drawing characters: this suite is run on Windows consoles whose
default code page mangles anything outside cp1252, and a report that renders
as mojibake is worse than one that renders as pipes and dashes.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from benchmarks.bench_core.stats import BenchRow, Comparison

# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------


def format_us(ns: float) -> str:
    """Nanoseconds as microseconds, the suite's display unit throughout."""
    return f"{ns / 1000.0:.3f}"


def format_scaled(value: float) -> str:
    """Thousands/millions/billions with a K/M/G suffix, matching
    `ember::bench::format_scaled` so throughput columns read the same in both
    languages' reports."""
    if value >= 1e9:
        return f"{value / 1e9:.2f} G"
    if value >= 1e6:
        return f"{value / 1e6:.2f} M"
    if value >= 1e3:
        return f"{value / 1e3:.2f} K"
    return f"{value:.2f} "


def format_speedup(ratio: float) -> str:
    """A ratio of 0.0 means "not measurable" (see Comparison._ratio), and is
    rendered as such rather than as 0.00x, which would read as an infinite
    slowdown."""
    if ratio <= 0.0:
        return "n/a"
    return f"{ratio:.2f}x"


def format_count(value: float) -> str:
    return f"{value:,.0f}"


# ---------------------------------------------------------------------------
# Generic ASCII table
# ---------------------------------------------------------------------------


def render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    align_right: Sequence[bool] | None = None,
) -> str:
    """Renders a fixed-width ASCII table sized to its widest cell."""
    if align_right is None:
        align_right = [False] + [True] * (len(headers) - 1)

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def render_row(cells: Sequence[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            parts.append(cell.rjust(widths[i]) if align_right[i] else cell.ljust(widths[i]))
        return "| " + " | ".join(parts) + " |"

    separator = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    lines = [separator, render_row(headers), separator]
    lines.extend(render_row(row) for row in rows)
    lines.append(separator)
    return "\n".join(lines)


def render_markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    align_right: Sequence[bool] | None = None,
) -> str:
    if align_right is None:
        align_right = [False] + [True] * (len(headers) - 1)

    alignment = ["---:" if right else "---" for right in align_right]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(alignment) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

_LATENCY_HEADERS = (
    "Benchmark",
    "Arm",
    "N",
    "Min us",
    "P50 us",
    "Mean us",
    "P95 us",
    "P99 us",
    "Max us",
    "StdDev us",
    "P99/P50",
    "Spikes",
    "Throughput",
)
_LATENCY_ALIGN = (False, False, True, True, True, True, True, True, True, True, True, True, True)


def _latency_cells(row: BenchRow) -> list[str]:
    s = row.latency
    throughput = "-"
    if row.throughput_value is not None and row.throughput_value > 0:
        throughput = f"{format_scaled(row.throughput_value)}{row.throughput_unit}"
    return [
        row.name,
        row.arm or "-",
        format_count(s.count),
        format_us(s.min_ns),
        format_us(s.p50_ns),
        format_us(s.mean_ns),
        format_us(s.p95_ns),
        format_us(s.p99_ns),
        format_us(s.max_ns),
        format_us(s.stddev_ns),
        f"{s.determinism_index:.2f}",
        format_count(s.wcet_spikes),
        throughput,
    ]


_COMPARISON_HEADERS = (
    "Measurement",
    "Baseline P50 us",
    "Hybrid P50 us",
    "P50 gain",
    "Baseline P99 us",
    "Hybrid P99 us",
    "P99 gain",
    "Baseline jitter us",
    "Hybrid jitter us",
    "Jitter gain",
)
_COMPARISON_ALIGN = (False, True, True, True, True, True, True, True, True, True)


def _comparison_cells(cmp: Comparison) -> list[str]:
    return [
        cmp.name,
        format_us(cmp.baseline.p50_ns),
        format_us(cmp.hybrid.p50_ns),
        format_speedup(cmp.p50_speedup),
        format_us(cmp.baseline.p99_ns),
        format_us(cmp.hybrid.p99_ns),
        format_speedup(cmp.p99_speedup),
        format_us(cmp.baseline.jitter_ns),
        format_us(cmp.hybrid.jitter_ns),
        format_speedup(cmp.jitter_reduction),
    ]


@dataclass
class Section:
    """One titled block of a report. `rows`, `comparisons` and `key_values`
    are all optional and render in that order, so a section can be a plain
    latency table, a head-to-head comparison, a diagnostics dump, or any
    combination without needing a separate section type for each."""

    title: str
    description: str = ""
    rows: list[BenchRow] = field(default_factory=list)
    comparisons: list[Comparison] = field(default_factory=list)
    key_values: dict[str, object] = field(default_factory=dict)
    footnotes: list[str] = field(default_factory=list)

    def render_console(self) -> str:
        parts = [f"\n=== {self.title} ===\n"]
        if self.description:
            parts.append(self.description.strip() + "\n")

        if self.comparisons:
            parts.append(
                render_table(
                    _COMPARISON_HEADERS,
                    [_comparison_cells(c) for c in self.comparisons],
                    _COMPARISON_ALIGN,
                )
            )
            parts.append("")

        if self.rows:
            parts.append(
                render_table(
                    _LATENCY_HEADERS, [_latency_cells(r) for r in self.rows], _LATENCY_ALIGN
                )
            )
            parts.append("")

        if self.key_values:
            kv_rows = [[str(k), str(v)] for k, v in self.key_values.items()]
            parts.append(render_table(("Metric", "Value"), kv_rows, (False, True)))
            parts.append("")

        for note in self.footnotes:
            parts.append(f"  note: {note}")

        return "\n".join(parts)

    def render_markdown(self) -> str:
        parts = [f"## {self.title}", ""]
        if self.description:
            parts.extend([self.description.strip(), ""])

        if self.comparisons:
            parts.append(
                render_markdown_table(
                    _COMPARISON_HEADERS,
                    [_comparison_cells(c) for c in self.comparisons],
                    _COMPARISON_ALIGN,
                )
            )
            parts.append("")

        if self.rows:
            parts.append(
                render_markdown_table(
                    _LATENCY_HEADERS, [_latency_cells(r) for r in self.rows], _LATENCY_ALIGN
                )
            )
            parts.append("")

        if self.key_values:
            parts.append(
                render_markdown_table(
                    ("Metric", "Value"),
                    [[str(k), str(v)] for k, v in self.key_values.items()],
                    (False, True),
                )
            )
            parts.append("")

        for note in self.footnotes:
            parts.append(f"> {note}")
        if self.footnotes:
            parts.append("")

        return "\n".join(parts)

    def as_dict(self) -> dict:
        return {
            "title": self.title,
            "description": self.description,
            "rows": [
                {
                    "suite": r.suite,
                    "name": r.name,
                    "arm": r.arm,
                    "latency": vars(r.latency) | {"jitter_ns": r.latency.jitter_ns},
                    "throughput_value": r.throughput_value,
                    "throughput_unit": r.throughput_unit,
                    "notes": r.notes,
                }
                for r in self.rows
            ],
            "comparisons": [c.as_dict() for c in self.comparisons],
            "key_values": {k: str(v) for k, v in self.key_values.items()},
            "footnotes": list(self.footnotes),
        }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def environment_metadata() -> dict[str, str]:
    """Captured into every report. Benchmark numbers are meaningless without
    the machine and interpreter that produced them, and a committed report
    outlives the shell session that knew them."""
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": str(__import__("os").cpu_count()),
        "python": f"{platform.python_implementation()} {platform.python_version()}",
        "python_executable": sys.executable,
    }


class Report:
    """Accumulates sections and renders them to console, Markdown or JSON."""

    def __init__(self, title: str, subtitle: str = ""):
        self.title = title
        self.subtitle = subtitle
        self.sections: list[Section] = []
        self.environment = environment_metadata()

    def add(self, section: Section) -> Section:
        self.sections.append(section)
        return section

    def render_console(self) -> str:
        header = [
            "",
            "=" * 78,
            f" {self.title}",
        ]
        if self.subtitle:
            header.append(f" {self.subtitle}")
        header.append("=" * 78)
        header.append(
            "  " + " | ".join(f"{k}: {v}" for k, v in self.environment.items() if k != "python_executable")
        )
        body = [s.render_console() for s in self.sections]
        return "\n".join(header) + "\n" + "\n".join(body)

    def render_markdown(self) -> str:
        parts = [f"# {self.title}", ""]
        if self.subtitle:
            parts.extend([f"_{self.subtitle}_", ""])

        parts.append("## Environment")
        parts.append("")
        parts.append(
            render_markdown_table(
                ("Field", "Value"),
                [[k, v] for k, v in self.environment.items()],
                (False, False),
            )
        )
        parts.append("")
        parts.extend(s.render_markdown() for s in self.sections)
        return "\n".join(parts)

    def as_dict(self) -> dict:
        return {
            "title": self.title,
            "subtitle": self.subtitle,
            "environment": self.environment,
            "sections": [s.as_dict() for s in self.sections],
        }

    def write_markdown(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.render_markdown(), encoding="utf-8")
        return target

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        return target


def print_report(report: Report, stream=None) -> None:
    """Writes the console rendering, tolerating consoles that cannot encode
    every character in the report (a non-UTF-8 Windows code page, most often)
    rather than dying with a UnicodeEncodeError after the whole benchmark run
    has already completed."""
    out = stream if stream is not None else sys.stdout
    text = report.render_console()
    try:
        out.write(text + "\n")
    except UnicodeEncodeError:
        encoding = getattr(out, "encoding", None) or "ascii"
        out.write(text.encode(encoding, errors="replace").decode(encoding) + "\n")
    out.flush()
