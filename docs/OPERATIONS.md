# Operations

## Endpoints

| Method | Path | Behaviour |
|---|---|---|
| GET | `/` | Service identity and links. |
| GET | `/health` | Liveness; always 200 while the process is up. |
| GET | `/ready` | 200 when the configured backend reports ready, otherwise 503. |
| GET | `/model` | Effective model, URL, batch size, element budget, timeout. |
| POST | `/predict` | Validated inference. Status codes: 200/400/413/500/503. |
| GET | `/metrics` | Prometheus text exposition (`text/plain; version=0.0.4`). |

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `TRITON_URL` | `http://localhost:8001` | Triton HTTP base URL (no trailing slash needed). |
| `TRITON_MODEL` | `onnx_wasm_model` | Model name in the repository. |
| `TRITON_TIMEOUT` | `5` | Per-request timeout in seconds. |
| `MAX_BATCH_SIZE` | `8` | Maximum first-dimension batch sent to Triton. |
| `MAX_REQUEST_ELEMENTS` | `1000000` | Reject larger requests with 413. |
| `ALLOW_ORIGINS` | `*` | Comma-separated CORS origins. |
| `TRITON_ENABLED` | `false` | `true` uses Triton; `false` uses the deterministic demo backend. |
| `DEMO_OUTPUT_SIZE` | `5` | Output width for the demo backend. |

Configuration is validated at startup; invalid values fail fast.

## Metrics

| Metric | Type | Meaning |
|---|---|---|
| `inference_requests_total` | counter | Requests received (including invalid). |
| `inference_errors_total` | counter | Requests that returned non-2xx. |
| `inference_latency_seconds` | histogram | End-to-end service time. |
| `inference_batch_size` | histogram | Number of batches per request. |

Example queries:

```promql
rate(inference_requests_total[5m])
sum(rate(inference_errors_total[5m])) / sum(rate(inference_requests_total[5m]))
histogram_quantile(0.95, sum(rate(inference_latency_seconds_bucket[5m])) by (le))
```

## Prometheus

`prometheus.yml` scrapes:

- `app:8000/metrics` (gateway)
- `triton:8002/metrics` (Triton)

The original file used `localhost`, which is wrong inside a container network. Service DNS names are now used.

## Grafana

1. Open `http://localhost:3000` (default `admin` / `admin`).
2. Add a Prometheus data source: `http://prometheus:9090`.
3. Panel suggestions:
   - Request rate: `rate(inference_requests_total[1m])`
   - Error ratio: `sum(rate(inference_errors_total[1m])) / sum(rate(inference_requests_total[1m]))`
   - p95 latency: `histogram_quantile(0.95, sum(rate(inference_latency_seconds_bucket[1m])) by (le))`
   - Batch size: `histogram_quantile(0.5, sum(rate(inference_batch_size_bucket[5m])) by (le))`

## Health checks

```bash
curl -fsS localhost:8000/health
curl -fsS localhost:8000/ready
curl -fsS localhost:8000/metrics | head
```

Kubernetes probes:

```yaml
livenessProbe:
  httpGet: { path: /health, port: 8000 }
readinessProbe:
  httpGet: { path: /ready, port: 8000 }
  periodSeconds: 5
```

## Docker notes

- `docker compose up --build` starts gateway, Triton, Prometheus, Grafana.
- Triton pulls a large NVIDIA image. On CPU-only hosts it can still run, but inference will be slow.
- The gateway builds the WASM package in a separate stage; the runtime image contains only Python and the compiled `pkg/`.
- Set `TRITON_ENABLED=false` to run the gateway alone with the demo backend:

```bash
docker build -t gateway .
docker run --rm -p 8000:8000 -e TRITON_ENABLED=false gateway
```

## Capacity guidance

- Keep `MAX_BATCH_SIZE` aligned with the model's `max_batch_size` in `config.pbtxt`.
- Size `MAX_REQUEST_ELEMENTS` to protect memory; a 224×224×3 float tensor is ~600 KB as JSON.
- Use a reverse proxy for TLS and request-size limits; the app also enforces its own budget.
- Monitor p95 latency and error ratio before increasing batch size.

## Incident playbook

| Symptom | Check | Action |
|---|---|---|
| `/ready` 503 | Triton container logs, `curl :8002/v2/health/ready` | Restart Triton; confirm model repository path. |
| 413 spike | Client payload sizes | Raise `MAX_REQUEST_ELEMENTS` only with memory headroom. |
| 500 spike | Gateway logs | Backend bug or unexpected response; inspect `error` field. |
| High latency | Grafana p95, Triton queue metrics | Lower batch size or add Triton instances. |
| No metrics | Prometheus targets page | Verify scrape DNS and `/metrics` path. |
