# Design decisions

## 1. Validate before batching

**Context:** Batching is an optimization; validation is a safety boundary.

**Decision:** Parse and validate the entire request first, then split into batches.

**Consequence:** A malformed request cannot occupy a Triton batch slot or trigger a partial inference.

## 2. Stdlib-only core

**Context:** The original service depended on `numpy`, `tritonclient`, `prometheus-fastapi-instrumentator`, and `onnxruntime` for a thin proxy.

**Decision:** Implement config, validation, batching, HTTP, and metrics with `urllib`, `json`, `math`, and `dataclasses`.

**Consequence:** The core installs anywhere and tests run in milliseconds. FastAPI is an optional adapter.

## 3. Explicit error mapping

**Context:** Returning HTTP 200 with `{"success": false}` breaks clients, retries, and dashboards.

**Decision:** Map validation → 400, budget → 413, backend → 503, unexpected → 500.

**Consequence:** Callers can rely on status codes; the body still carries a machine-readable error.

## 4. First-dimension batching

**Context:** Triton's `max_batch_size` applies to the leading dimension.

**Decision:** Split `[N, ...]` into `[min(N, max_batch), ...]` chunks; never batch a 1-D tensor.

**Consequence:** The gateway respects model configuration and keeps row-major data contiguous.

## 5. Balanced-bracket config parser

**Context:** A non-greedy regex stops at the first `]`, which is inside `dims: [...]`.

**Decision:** Scan from `input [` / `output [` while tracking bracket depth.

**Consequence:** Real Triton configs parse correctly; unclosed blocks fail loudly.

## 6. Hand-rolled Prometheus exposition

**Context:** `prometheus_client` is convenient but adds a dependency and global registry.

**Decision:** A tiny registry renders counters and histograms with sorted names, validated names, finite buckets, `+Inf`, `_sum`, and `_count`.

**Consequence:** Output is deterministic and diffable; no global state.

## 7. Readiness is a probe, not a flag

**Context:** A gateway that reports healthy while Triton is down causes cascading failures.

**Decision:** `/ready` calls `Backend.health()` and returns 503 when it fails.

**Consequence:** Load balancers and Kubernetes remove unhealthy replicas.

## 8. Remove dead dependencies from the WASM build

**Context:** The original crate pulled Liquid templates (which pulled `pest`/`psm`) and `wee_alloc` without using them; the WASM build failed on `psm`.

**Decision:** Remove unused Liquid crates and `wee_alloc`; keep `tract-onnx`, `wasm-bindgen`, and `console_error_panic_hook`.

**Consequence:** The release build succeeds and the bundle is smaller.

## 9. Shape validation in Rust before tensor creation

**Context:** `ArrayD::from_shape_vec` produces an opaque error and can overflow.

**Decision:** A native-testable `shape_product()` rejects empty shapes, zero dimensions, and overflow.

**Consequence:** Browser callers get a clear message, and the helper has four unit tests.

## 10. Fix container topology

**Context:** The original compose mounted `./app` over `/app`, ran `main:app` instead of `app.main:app`, scraped `localhost` from inside containers, and hard-coded a GPU reservation.

**Decision:** Copy code into the image, use `app.main:app`, scrape service DNS names, make the Triton URL configurable, and keep GPU optional.

**Consequence:** `docker compose up` is coherent on CPU-only machines; Triton still benefits from a GPU if present.
