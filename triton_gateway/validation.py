"""Request validation and deterministic batching for the inference gateway."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


class ValidationError(ValueError):
    """Client error with an HTTP status code."""

    status_code = 400


class PayloadTooLarge(ValidationError):
    """Raised when a request exceeds the configured element budget."""

    status_code = 413


@dataclass(frozen=True)
class InferenceRequest:
    """Validated, immutable inference input."""

    input_data: tuple[float, ...]
    input_shape: tuple[int, ...]

    @property
    def numel(self) -> int:
        return len(self.input_data)

    def batches(self, max_batch_size: int) -> tuple["InferenceRequest", ...]:
        """Split along the first dimension, preserving row-major order."""
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        first = self.input_shape[0]
        if len(self.input_shape) == 1 or first <= max_batch_size:
            return (self,)

        stride = self.numel // first
        return tuple(
            InferenceRequest(
                input_data=self.input_data[start * stride : (start + batch) * stride],
                input_shape=(batch,) + self.input_shape[1:],
            )
            for start in range(0, first, max_batch_size)
            for batch in (min(max_batch_size, first - start),)
        )

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, max_elements: int = 1_000_000
    ) -> "InferenceRequest":
        """Validate a JSON payload.

        Raises:
            ValidationError: malformed shape/data or non-numeric values.
            PayloadTooLarge: the request exceeds ``max_elements``.
        """
        if not isinstance(payload, Mapping):
            raise ValidationError("payload must be a JSON object")

        raw_shape = payload.get("input_shape")
        raw_data = payload.get("input_data")
        if not isinstance(raw_shape, list) or not raw_shape:
            raise ValidationError("input_shape must be a non-empty list")
        if not isinstance(raw_data, list):
            raise ValidationError("input_data must be a list")

        shape: list[int] = []
        for dim in raw_shape:
            if isinstance(dim, bool) or not isinstance(dim, int):
                raise ValidationError("input_shape values must be integers")
            if dim < 1:
                raise ValidationError("input_shape values must be >= 1")
            shape.append(dim)

        expected = math.prod(shape)
        if expected > max_elements:
            raise PayloadTooLarge(
                f"request has {expected} elements (limit {max_elements})"
            )
        if len(raw_data) != expected:
            raise ValidationError(
                f"input_data length {len(raw_data)} does not match shape product {expected}"
            )

        data: list[float] = []
        for value in raw_data:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValidationError("input_data values must be numbers")
            number = float(value)
            if not math.isfinite(number):
                raise ValidationError("input_data values must be finite")
            data.append(number)

        return cls(tuple(data), tuple(shape))


def flatten_batches(
    request: InferenceRequest, outputs: Iterable[Iterable[float]]
) -> list[float]:
    """Concatenate per-batch outputs in order."""
    result: list[float] = []
    for batch in outputs:
        result.extend(float(value) for value in batch)
    return result
