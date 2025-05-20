"""Configuration and Triton model-config parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping


class ConfigError(ValueError):
    """Raised when configuration is missing or malformed."""


@dataclass(frozen=True)
class TensorSpec:
    """One Triton input or output tensor declaration."""

    name: str
    dtype: str
    dims: tuple[int, ...]

    @property
    def element_count(self) -> int:
        count = 1
        for dim in self.dims:
            count *= dim
        return count


@dataclass(frozen=True)
class ModelSpec:
    """Subset of a Triton ``config.pbtxt`` needed by the gateway."""

    name: str
    platform: str
    max_batch_size: int
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]


@dataclass(frozen=True)
class GatewayConfig:
    """Runtime configuration for the HTTP gateway."""

    triton_url: str = "http://localhost:8001"
    model_name: str = "onnx_wasm_model"
    timeout_s: float = 5.0
    max_batch_size: int = 8
    max_request_elements: int = 1_000_000
    allow_origins: tuple[str, ...] = ("*",)
    triton_enabled: bool = False

    def __post_init__(self) -> None:
        if not self.triton_url.startswith(("http://", "https://")):
            raise ConfigError("triton_url must start with http:// or https://")
        if not self.model_name.strip():
            raise ConfigError("model_name must not be empty")
        if self.timeout_s <= 0:
            raise ConfigError("timeout_s must be > 0")
        if self.max_batch_size < 1:
            raise ConfigError("max_batch_size must be >= 1")
        if self.max_request_elements < 1:
            raise ConfigError("max_request_elements must be >= 1")
        if not self.allow_origins:
            raise ConfigError("allow_origins must not be empty")

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "GatewayConfig":
        """Build config from environment variables with safe defaults."""

        def number(name: str, default: float, cast: type) -> float:
            raw = env.get(name)
            if raw is None or raw == "":
                return default
            try:
                value = cast(raw)
            except ValueError as exc:
                raise ConfigError(f"{name} must be a number") from exc
            return value

        origins = tuple(
            origin.strip()
            for origin in env.get("ALLOW_ORIGINS", "*").split(",")
            if origin.strip()
        )
        return cls(
            triton_url=env.get("TRITON_URL", "http://localhost:8001").rstrip("/"),
            model_name=env.get("TRITON_MODEL", "onnx_wasm_model"),
            timeout_s=float(number("TRITON_TIMEOUT", 5.0, float)),
            max_batch_size=int(number("MAX_BATCH_SIZE", 8, int)),
            max_request_elements=int(number("MAX_REQUEST_ELEMENTS", 1_000_000, int)),
            allow_origins=origins,
            triton_enabled=env.get("TRITON_ENABLED", "false").lower() == "true",
        )


_NAME_RE = re.compile(r'name:\s*"([^"]+)"')
_PLATFORM_RE = re.compile(r'platform:\s*"([^"]+)"')
_BATCH_RE = re.compile(r"max_batch_size:\s*(\d+)")
_BLOCK_START_RE = re.compile(r"(input|output)\s*\[")
_DTYPE_RE = re.compile(r"data_type:\s*TYPE_([A-Z0-9_]+)")
_DIMS_RE = re.compile(r"dims:\s*\[([^\]]*)\]")


def _iter_blocks(text: str):
    """Yield ``(kind, body)`` for bracketed input/output blocks.

    Triton configs nest ``dims: [...]`` inside the block, so a naive regex
    would stop at the first closing bracket. This scanner balances brackets.
    """
    for match in _BLOCK_START_RE.finditer(text):
        kind = match.group(1)
        start = match.end()
        depth = 1
        for index in range(start, len(text)):
            if text[index] == "[":
                depth += 1
            elif text[index] == "]":
                depth -= 1
                if depth == 0:
                    yield kind, text[start:index]
                    break
        else:
            raise ConfigError(f"unclosed {kind} block")


def parse_model_config(text: str) -> ModelSpec:
    """Parse the supported subset of a Triton ``config.pbtxt``.

    Raises:
        ConfigError: when required fields are missing or malformed.
    """
    if not text or not text.strip():
        raise ConfigError("model config is empty")

    name_match = _NAME_RE.search(text)
    if name_match is None:
        raise ConfigError("model config is missing name")
    name = name_match.group(1)

    platform_match = _PLATFORM_RE.search(text)
    platform = platform_match.group(1) if platform_match else "unknown"

    batch_match = _BATCH_RE.search(text)
    max_batch = int(batch_match.group(1)) if batch_match else 0

    inputs: list[TensorSpec] = []
    outputs: list[TensorSpec] = []
    for kind, body in _iter_blocks(text):
        spec_name = _NAME_RE.search(body)
        dtype = _DTYPE_RE.search(body)
        dims = _DIMS_RE.search(body)
        if spec_name is None or dims is None:
            raise ConfigError(f"{kind} block is missing name or dims")
        parsed_dims = tuple(int(part.strip()) for part in dims.group(1).split(",") if part.strip())
        if not parsed_dims or any(dim < 1 for dim in parsed_dims):
            raise ConfigError(f"{kind} block has invalid dims")
        target = inputs if kind == "input" else outputs
        target.append(
            TensorSpec(
                name=spec_name.group(1),
                dtype=dtype.group(1) if dtype else "FP32",
                dims=parsed_dims,
            )
        )

    if not inputs or not outputs:
        raise ConfigError("model config must declare at least one input and output")

    return ModelSpec(name, platform, max_batch, tuple(inputs), tuple(outputs))
