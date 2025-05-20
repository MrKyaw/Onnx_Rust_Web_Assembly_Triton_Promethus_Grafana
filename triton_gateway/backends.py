"""Inference backends: deterministic demo backend and Triton HTTP client."""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from .validation import InferenceRequest


class BackendError(RuntimeError):
    """Raised when inference or a readiness probe fails."""


class Backend(Protocol):
    def infer(self, request: InferenceRequest) -> list[float]: ...

    def health(self) -> bool: ...


@dataclass
class DemoBackend:
    """Deterministic, dependency-free backend used for demos and tests.

    It average-pools the flattened input into ``output_size`` buckets and
    applies a numerically stable softmax, which produces a probability-like
    vector without any model weights.
    """

    output_size: int = 5

    def __post_init__(self) -> None:
        if self.output_size < 1:
            raise ValueError("output_size must be >= 1")

    def infer(self, request: InferenceRequest) -> list[float]:
        if request.numel == 0:
            raise BackendError("cannot infer an empty tensor")
        step = max(1, math.ceil(request.numel / self.output_size))
        pooled = [
            sum(request.input_data[start : start + step]) / len(request.input_data[start : start + step])
            for start in range(0, request.numel, step)
        ][: self.output_size]
        peak = max(pooled)
        exps = [math.exp(value - peak) for value in pooled]
        total = sum(exps)
        return [value / total for value in exps]

    def health(self) -> bool:
        return True


@dataclass
class TritonHTTPBackend:
    """Minimal Triton v2 HTTP client built on the standard library."""

    url: str
    model_name: str
    timeout_s: float = 5.0
    input_name: str = "input"
    output_name: str = "output"

    def __post_init__(self) -> None:
        if not self.url:
            raise ValueError("url must not be empty")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")

    def infer(self, request: InferenceRequest) -> list[float]:
        body = {
            "inputs": [
                {
                    "name": self.input_name,
                    "shape": list(request.input_shape),
                    "datatype": "FP32",
                    "data": list(request.input_data),
                }
            ],
            "outputs": [{"name": self.output_name}],
        }
        payload = self._request(
            f"/v2/models/{self.model_name}/infer", body, method="POST"
        )
        try:
            data = payload["outputs"][0]["data"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError("Triton response is missing outputs[0].data") from exc
        if not isinstance(data, list) or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in data
        ):
            raise BackendError("Triton output data must be a numeric list")
        return [float(value) for value in data]

    def health(self) -> bool:
        try:
            self._request(f"/v2/health/ready", None, method="GET")
        except BackendError:
            return False
        return True

    def _request(self, path: str, body: dict[str, Any] | None, *, method: str) -> Any:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            f"{self.url.rstrip('/')}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise BackendError(f"Triton request failed: {exc}") from exc

        if method == "GET":
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendError("Triton returned invalid JSON") from exc
