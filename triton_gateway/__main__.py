"""CLI demo for the ONNX/Triton gateway core.

Usage:
    python -m triton_gateway --demo
    python -m triton_gateway --demo --json
    python -m triton_gateway --model-config model_repository/onnx_wasm_model/config.pbtxt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .backends import DemoBackend
from .config import ConfigError, parse_model_config
from .service import GatewayService


def demo_payload() -> dict[str, object]:
    """A small deterministic tensor (1x3x4x4)."""
    return {
        "input_data": [round(index / 10.0, 2) for index in range(48)],
        "input_shape": [1, 3, 4, 4],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="triton_gateway",
        description="Dependency-free ONNX/Triton inference gateway core.",
    )
    parser.add_argument("--demo", action="store_true", help="run the built-in tensor")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument(
        "--model-config", type=Path, help="parse a Triton config.pbtxt and print its spec"
    )
    parser.add_argument(
        "--output-size", type=int, default=5, help="demo backend output width (default: 5)"
    )
    parser.add_argument("--version", action="version", version=f"triton_gateway {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.model_config:
        try:
            spec = parse_model_config(args.model_config.read_text(encoding="utf-8"))
        except (OSError, ConfigError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        payload = {
            "name": spec.name,
            "platform": spec.platform,
            "max_batch_size": spec.max_batch_size,
            "inputs": [
                {"name": t.name, "dtype": t.dtype, "dims": list(t.dims)}
                for t in spec.inputs
            ],
            "outputs": [
                {"name": t.name, "dtype": t.dtype, "dims": list(t.dims)}
                for t in spec.outputs
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if not args.demo:
        print("error: provide --demo or --model-config", file=sys.stderr)
        return 2

    try:
        service = GatewayService(backend=DemoBackend(output_size=args.output_size))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    result = service.predict(demo_payload())
    report = {
        "result": result.to_dict(),
        "health": service.health(),
        "ready": service.ready()[1],
        "metrics": service.metrics_text(),
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"status: {result.status_code} success={result.success}")
        print(f"output: {[round(value, 4) for value in result.output]}")
        print(f"elapsed_ms: {result.elapsed_ms} batches: {result.batches}")
        print("--- metrics ---")
        print(service.metrics_text(), end="")
    return 0 if result.success else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
