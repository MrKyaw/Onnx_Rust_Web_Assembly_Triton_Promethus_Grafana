"""Dependency-free Prometheus metrics registry."""

from __future__ import annotations

import re
from typing import Mapping

_NAME_RE = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
_DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


class MetricsError(ValueError):
    """Raised for invalid metric names or observations."""


class Metrics:
    """Counters and histograms with Prometheus text exposition."""

    def __init__(self, *, buckets: tuple[float, ...] = _DEFAULT_BUCKETS) -> None:
        if not buckets or tuple(sorted(buckets)) != buckets:
            raise MetricsError("buckets must be a non-empty sorted tuple")
        if any(bucket <= 0 for bucket in buckets):
            raise MetricsError("buckets must be > 0")
        self.buckets = buckets
        self._counters: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = {}

    def increment(self, name: str, value: float = 1.0) -> None:
        self._validate_name(name)
        if value < 0:
            raise MetricsError("counter increments must be >= 0")
        self._counters[name] = self._counters.get(name, 0.0) + value

    def observe(self, name: str, value: float) -> None:
        self._validate_name(name)
        if value < 0:
            raise MetricsError("observations must be >= 0")
        self._histograms.setdefault(name, []).append(float(value))

    def counter(self, name: str) -> float:
        return self._counters.get(name, 0.0)

    def observations(self, name: str) -> tuple[float, ...]:
        return tuple(self._histograms.get(name, ()))

    def render(self) -> str:
        """Render Prometheus text format (deterministic ordering)."""
        lines: list[str] = []
        for name in sorted(self._counters):
            lines.append(f"# HELP {name} Counter maintained by triton_gateway.")
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {self._format(self._counters[name])}")
        for name in sorted(self._histograms):
            values = self._histograms[name]
            lines.append(f"# HELP {name} Histogram maintained by triton_gateway.")
            lines.append(f"# TYPE {name} histogram")
            for bucket in self.buckets:
                count = sum(1 for value in values if value <= bucket)
                lines.append(f'{name}_bucket{{le="{self._format(bucket)}"}} {count}')
            lines.append(f'{name}_bucket{{le="+Inf"}} {len(values)}')
            lines.append(f"{name}_count {len(values)}")
            lines.append(f"{name}_sum {self._format(sum(values))}")
        return "\n".join(lines) + ("\n" if lines else "")

    @staticmethod
    def _validate_name(name: str) -> None:
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise MetricsError(f"invalid metric name: {name!r}")

    @staticmethod
    def _format(value: float) -> str:
        return f"{value:g}"
