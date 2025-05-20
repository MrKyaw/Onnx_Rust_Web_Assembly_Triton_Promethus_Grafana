# Testing

## Run everything

```bash
python -m pip install -e ".[dev]"
pytest --cov=triton_gateway --cov-branch --cov-report=term-missing
cargo test
wasm-pack build --target web --release   # optional, slow on a cold cache
```

## Verified results

**Python 3.12.8**

```text
83 passed  (77 core + 6 FastAPI contract)

Name                           Stmts   Miss Branch BrPart  Cover
----------------------------------------------------------------
triton_gateway/__init__.py         7      0      0      0   100%
triton_gateway/__main__.py        49      0      6      0   100%
triton_gateway/backends.py        69      0     12      0   100%
triton_gateway/config.py         105      0     38      0   100%
triton_gateway/metrics.py         53      0     16      0   100%
triton_gateway/service.py         61      0      2      0   100%
triton_gateway/validation.py      59      0     28      0   100%
----------------------------------------------------------------
TOTAL                            403      0    102      0   100%
```

**Rust**

```text
running 4 tests
test tests::computes_element_count ... ok
test tests::rejects_empty_shape ... ok
test tests::rejects_overflow ... ok
test tests::rejects_zero_dimension ... ok
test result: ok. 4 passed; 0 failed
```

**WASM**

```text
wasm-pack build --target web --release
[INFO]: ✨   Done in 4m 49s
[INFO]: 📦   Your wasm pkg is ready to publish at .../pkg
```

## Coverage matrix

| Module | What is verified |
|---|---|
| `config` | Full and minimal `config.pbtxt`; missing name/inputs/outputs; invalid dims; unclosed blocks; every `GatewayConfig` invariant; env defaults, custom values, bad numbers, origin filtering. |
| `validation` | Non-object payloads; missing/empty/wrong-typed shape; boolean dims; zero dims; length mismatch; NaN/Inf; oversized requests (413); even and uneven batching; invalid batch size; flattening. |
| `backends` | Demo softmax and edge sizes; empty tensor; Triton request shape; success; missing/malformed outputs; non-numeric data; readiness true/false; transport failure; invalid JSON; invalid UTF-8. |
| `metrics` | Empty render; counters; histogram buckets; `+Inf`, `_sum`, `_count`; invalid bucket sets; invalid names; negative values. |
| `service` | Success; validation 400; payload 413; backend 503; unexpected 500; batching; health; readiness true/false/exception; model info; metrics text; timing. |
| CLI | Demo text/JSON; model-config parsing; missing/invalid config; no action; invalid output size; failure exit code; `--version`. |
| HTTP adapter | Root, health, model, readiness, prediction, validation, metrics, content type. |

## CI

`.github/workflows/ci.yml`:

1. Python 3.10/3.11/3.12: install `.[dev]`, run `pytest` with the 100% coverage gate.
2. Rust stable: `cargo test` with a cargo cache.
3. Linux WASM job: install `wasm-pack`, run `wasm-pack build --target web --release`.

## Not covered by CI

- A live Triton server. The backend protocol is tested with a fake `urlopen`; end-to-end validation requires `docker compose up` and a model in `model_repository/`.
- Grafana dashboard rendering. Provisioning is documented but not screenshot-tested.

## Failure injection examples

```python
from triton_gateway import GatewayService, TritonHTTPBackend

service = GatewayService(
    backend=TritonHTTPBackend("http://localhost:8001", "onnx_wasm_model")
)
result = service.predict({"input_data": [1.0], "input_shape": [1]})
print(result.status_code, result.error)   # 503 when Triton is down
```
