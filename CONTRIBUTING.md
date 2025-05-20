# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
pytest --cov=triton_gateway --cov-branch --cov-report=term-missing
cargo test
```

## Quality bar

- Python 3.10–3.12 pass; Rust stable passes.
- 100% line and branch coverage for `triton_gateway`.
- `wasm-pack build --target web --release` succeeds for changes under `src/`.
- No new runtime dependencies in the core without a written rationale.
- Public functions have type hints and docstrings.
- Status-code semantics stay stable.

## Pull requests

1. Branch from `main`.
2. Keep commits focused; explain the *why*.
3. Run `make check`.
4. Describe the change, tests, and operational impact.

## Model changes

If you change `model_repository/*/config.pbtxt`, update the README and verify the gateway's `max_batch_size` and input dims. If you change the ONNX model, validate both the WASM path and the Triton path.

## Reporting issues

Include the request payload shape, status code, gateway logs, `config.pbtxt`, and whether Triton was enabled.
