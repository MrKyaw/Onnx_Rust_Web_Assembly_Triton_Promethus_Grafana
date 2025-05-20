"""Application service that maps validated requests to backend calls."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .backends import Backend, BackendError, DemoBackend
from .config import GatewayConfig
from .metrics import Metrics
from .validation import InferenceRequest, ValidationError


@dataclass(frozen=True)
class PredictionResult:
    """HTTP-ready result. ``predict`` never raises."""

    success: bool
    output: tuple[float, ...] = ()
    error: str | None = None
    status_code: int = 200
    elapsed_ms: float = 0.0
    batches: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output": list(self.output),
            "error": self.error,
            "status_code": self.status_code,
            "elapsed_ms": self.elapsed_ms,
            "batches": self.batches,
        }


class GatewayService:
    """Coordinates validation, batching, inference, metrics, and readiness."""

    def __init__(
        self,
        config: GatewayConfig | None = None,
        backend: Backend | None = None,
        metrics: Metrics | None = None,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.config = config or GatewayConfig()
        self.backend = backend or DemoBackend()
        self.metrics = metrics or Metrics()
        self._clock = clock

    def predict(self, payload: Mapping[str, Any]) -> PredictionResult:
        """Validate and execute an inference request."""
        started = self._clock()
        self.metrics.increment("inference_requests_total")
        try:
            request = InferenceRequest.from_payload(
                payload, max_elements=self.config.max_request_elements
            )
        except ValidationError as exc:
            return self._failure(exc.status_code, str(exc), started)

        batches = request.batches(self.config.max_batch_size)
        self.metrics.observe("inference_batch_size", float(len(batches)))
        output: list[float] = []
        try:
            for batch in batches:
                output.extend(self.backend.infer(batch))
        except BackendError as exc:
            return self._failure(503, str(exc), started, batches=len(batches))
        except Exception as exc:  # defensive: a backend bug must not kill the API
            return self._failure(500, f"inference failed: {exc}", started, batches=len(batches))

        elapsed = self._elapsed(started)
        self.metrics.observe("inference_latency_seconds", elapsed / 1000.0)
        return PredictionResult(
            success=True,
            output=tuple(output),
            status_code=200,
            elapsed_ms=elapsed,
            batches=len(batches),
        )

    def _failure(
        self,
        status_code: int,
        error: str,
        started: float,
        *,
        batches: int = 0,
    ) -> PredictionResult:
        self.metrics.increment("inference_errors_total")
        return PredictionResult(
            success=False,
            error=error,
            status_code=status_code,
            elapsed_ms=self._elapsed(started),
            batches=batches,
        )

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": self.config.model_name,
            "triton_enabled": self.config.triton_enabled,
        }

    def ready(self) -> tuple[bool, dict[str, Any]]:
        try:
            healthy = bool(self.backend.health())
        except Exception:
            healthy = False
        return healthy, {
            "status": "ready" if healthy else "unavailable",
            "model": self.config.model_name,
        }

    def model_info(self) -> dict[str, Any]:
        return {
            "model": self.config.model_name,
            "triton_url": self.config.triton_url,
            "max_batch_size": self.config.max_batch_size,
            "max_request_elements": self.config.max_request_elements,
            "timeout_s": self.config.timeout_s,
        }

    def metrics_text(self) -> str:
        return self.metrics.render()

    def _elapsed(self, started: float) -> float:
        return round((self._clock() - started) * 1000.0, 3)
