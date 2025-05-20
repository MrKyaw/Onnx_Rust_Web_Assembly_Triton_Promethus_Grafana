"""FastAPI adapter for the tested :mod:`triton_gateway` core.

Run locally:
    uvicorn app.main:app --reload --port 8000

The service uses the deterministic demo backend unless ``TRITON_ENABLED=true``.
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from triton_gateway import (
    DemoBackend,
    GatewayConfig,
    GatewayService,
    TritonHTTPBackend,
)


@lru_cache(maxsize=1)
def get_service() -> GatewayService:
    """Build the gateway once per process from environment configuration."""
    config = GatewayConfig.from_env(os.environ)
    if config.triton_enabled:
        backend = TritonHTTPBackend(
            config.triton_url, config.model_name, config.timeout_s
        )
    else:
        backend = DemoBackend(output_size=int(os.getenv("DEMO_OUTPUT_SIZE", "5")))
    return GatewayService(config, backend)


config = GatewayConfig.from_env(os.environ)
app = FastAPI(
    title="ONNX Rust WASM + Triton Gateway",
    version="1.0.0",
    description="Validated inference gateway with Prometheus metrics and readiness probes.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(config.allow_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/")
def read_root() -> dict[str, object]:
    return {
        "service": "onnx-rust-wasm-triton-gateway",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/health")
def health() -> dict[str, object]:
    return get_service().health()


@app.get("/ready")
def ready() -> JSONResponse:
    healthy, body = get_service().ready()
    return JSONResponse(status_code=200 if healthy else 503, content=body)


@app.get("/model")
def model() -> dict[str, object]:
    return get_service().model_info()


@app.post("/predict")
def predict(payload: dict[str, object]) -> JSONResponse:
    result = get_service().predict(payload)
    return JSONResponse(status_code=result.status_code, content=result.to_dict())


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> PlainTextResponse:
    return PlainTextResponse(
        get_service().metrics_text(), media_type="text/plain; version=0.0.4"
    )
