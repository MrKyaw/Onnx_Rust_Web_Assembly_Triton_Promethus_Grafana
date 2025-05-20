"""Dependency-free core for an ONNX → Triton inference gateway.

The original repository demonstrated ONNX inference in Rust/WebAssembly plus
a FastAPI/Triton/Prometheus/Grafana stack. This package isolates the parts
that can be reasoned about and tested without a GPU or container runtime:
configuration, model-config parsing, request validation, batching, backend
contracts, Prometheus exposition, and readiness.
"""

from .backends import Backend, BackendError, DemoBackend, TritonHTTPBackend
from .config import ConfigError, GatewayConfig, ModelSpec, TensorSpec, parse_model_config
from .metrics import Metrics, MetricsError
from .service import GatewayService, PredictionResult
from .validation import (
    InferenceRequest,
    PayloadTooLarge,
    ValidationError,
    flatten_batches,
)

__version__ = "1.0.0"

__all__ = [
    "Backend",
    "BackendError",
    "ConfigError",
    "DemoBackend",
    "GatewayConfig",
    "GatewayService",
    "InferenceRequest",
    "Metrics",
    "MetricsError",
    "ModelSpec",
    "PayloadTooLarge",
    "PredictionResult",
    "TensorSpec",
    "TritonHTTPBackend",
    "ValidationError",
    "flatten_batches",
    "parse_model_config",
]
