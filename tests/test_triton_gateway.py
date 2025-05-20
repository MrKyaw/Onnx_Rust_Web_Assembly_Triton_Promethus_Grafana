"""Tests for the dependency-free ONNX/Triton gateway core."""

from __future__ import annotations

import io
import json
import math
import urllib.error

import pytest

from triton_gateway import (
    BackendError,
    ConfigError,
    DemoBackend,
    GatewayConfig,
    GatewayService,
    InferenceRequest,
    Metrics,
    MetricsError,
    PayloadTooLarge,
    PredictionResult,
    TensorSpec,
    TritonHTTPBackend,
    ValidationError,
    flatten_batches,
    parse_model_config,
)
from triton_gateway.__main__ import build_parser, demo_payload, main
from triton_gateway.config import ModelSpec


MODEL_CONFIG = '''
name: "onnx_wasm_model"
platform: "onnxruntime_onnx"
max_batch_size: 8
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [3, 224, 224]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [1000]
  }
]
'''


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def read(self) -> bytes:
        if isinstance(self.payload, bytes):
            return self.payload
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


class TestModelConfig:
    def test_parse_full(self) -> None:
        spec = parse_model_config(MODEL_CONFIG)
        assert spec.name == "onnx_wasm_model"
        assert spec.platform == "onnxruntime_onnx"
        assert spec.max_batch_size == 8
        assert spec.inputs == (TensorSpec("input", "FP32", (3, 224, 224)),)
        assert spec.outputs == (TensorSpec("output", "FP32", (1000,)),)
        assert spec.inputs[0].element_count == 3 * 224 * 224

    def test_defaults_when_optional_fields_missing(self) -> None:
        spec = parse_model_config(
            'name: "m"\ninput [{ name: "x" dims: [2] }]\noutput [{ name: "y" dims: [1] }]'
        )
        assert spec.platform == "unknown"
        assert spec.max_batch_size == 0
        assert spec.inputs[0].dtype == "FP32"

    @pytest.mark.parametrize(
        "text,message",
        [
            ("", "empty"),
            ("   ", "empty"),
            ('platform: "x"', "name"),
            ('name: "m"\ninput [{ name: "x" dims: [2] }]', "at least one"),
            ('name: "m"\ninput [{ dims: [2] }]\noutput [{ name: "y" dims: [1] }]', "missing name"),
            ('name: "m"\ninput [{ name: "x" }]\noutput [{ name: "y" dims: [1] }]', "missing name"),
            ('name: "m"\ninput [{ name: "x" dims: [] }]\noutput [{ name: "y" dims: [1] }]', "invalid dims"),
            ('name: "m"\ninput [{ name: "x" dims: [0] }]\noutput [{ name: "y" dims: [1] }]', "invalid dims"),
        ],
    )
    def test_parse_errors(self, text: str, message: str) -> None:
        with pytest.raises(ConfigError, match=message):
            parse_model_config(text)

    def test_tensor_element_count(self) -> None:
        assert TensorSpec("x", "FP32", (2, 3)).element_count == 6

    def test_unclosed_block(self) -> None:
        with pytest.raises(ConfigError, match="unclosed input"):
            parse_model_config('name: "m"\ninput [ name: "x" dims: [2] ')


class TestGatewayConfig:
    def test_defaults(self) -> None:
        config = GatewayConfig()
        assert config.model_name == "onnx_wasm_model"
        assert config.allow_origins == ("*",)
        assert config.triton_enabled is False

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"triton_url": "localhost:8001"},
            {"model_name": " "},
            {"timeout_s": 0},
            {"max_batch_size": 0},
            {"max_request_elements": 0},
            {"allow_origins": ()},
        ],
    )
    def test_rejects_invalid(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ConfigError):
            GatewayConfig(**kwargs)  # type: ignore[arg-type]

    def test_from_env_defaults(self) -> None:
        config = GatewayConfig.from_env({})
        assert config.triton_url == "http://localhost:8001"
        assert config.timeout_s == 5.0

    def test_from_env_custom(self) -> None:
        config = GatewayConfig.from_env(
            {
                "TRITON_URL": "http://triton:8001/",
                "TRITON_MODEL": "m",
                "TRITON_TIMEOUT": "2.5",
                "MAX_BATCH_SIZE": "4",
                "MAX_REQUEST_ELEMENTS": "100",
                "ALLOW_ORIGINS": "https://a.test, https://b.test ,",
                "TRITON_ENABLED": "TRUE",
            }
        )
        assert config.triton_url == "http://triton:8001"
        assert config.model_name == "m"
        assert config.timeout_s == 2.5
        assert config.max_batch_size == 4
        assert config.max_request_elements == 100
        assert config.allow_origins == ("https://a.test", "https://b.test")
        assert config.triton_enabled is True

    def test_from_env_bad_number(self) -> None:
        with pytest.raises(ConfigError, match="MAX_BATCH_SIZE"):
            GatewayConfig.from_env({"MAX_BATCH_SIZE": "many"})


class TestValidation:
    def test_valid_request(self) -> None:
        request = InferenceRequest.from_payload(
            {"input_data": [1, 2.5, 3, 4], "input_shape": [2, 2]}
        )
        assert request.input_data == (1.0, 2.5, 3.0, 4.0)
        assert request.input_shape == (2, 2)
        assert request.numel == 4

    @pytest.mark.parametrize(
        "payload,message",
        [
            ([], "JSON object"),
            ({}, "input_shape"),
            ({"input_shape": [], "input_data": []}, "input_shape"),
            ({"input_shape": [1], "input_data": "x"}, "input_data"),
            ({"input_shape": [True], "input_data": [1]}, "integers"),
            ({"input_shape": ["1"], "input_data": [1]}, "integers"),
            ({"input_shape": [0], "input_data": []}, ">= 1"),
            ({"input_shape": [2], "input_data": [1]}, "does not match"),
            ({"input_shape": [1], "input_data": [True]}, "numbers"),
            ({"input_shape": [1], "input_data": ["1"]}, "numbers"),
            ({"input_shape": [1], "input_data": [math.nan]}, "finite"),
            ({"input_shape": [1], "input_data": [math.inf]}, "finite"),
        ],
    )
    def test_rejects_bad_payloads(self, payload: object, message: str) -> None:
        with pytest.raises(ValidationError, match=message):
            InferenceRequest.from_payload(payload)  # type: ignore[arg-type]

    def test_payload_too_large(self) -> None:
        with pytest.raises(PayloadTooLarge) as exc:
            InferenceRequest.from_payload(
                {"input_shape": [10], "input_data": list(range(10))}, max_elements=9
            )
        assert exc.value.status_code == 413

    def test_batches_whole_request(self) -> None:
        request = InferenceRequest((1.0, 2.0), (2,))
        assert request.batches(8) == (request,)
        flat = InferenceRequest((1.0, 2.0, 3.0), (3, 1))
        assert flat.batches(8) == (flat,)

    def test_batches_split_first_dimension(self) -> None:
        request = InferenceRequest(tuple(range(8)), (4, 2))
        batches = request.batches(2)
        assert len(batches) == 2
        assert batches[0].input_shape == (2, 2)
        assert batches[0].input_data == (0.0, 1.0, 2.0, 3.0)
        assert batches[1].input_data == (4.0, 5.0, 6.0, 7.0)

    def test_batches_uneven_split(self) -> None:
        request = InferenceRequest(tuple(range(6)), (3, 2))
        batches = request.batches(2)
        assert [batch.input_shape[0] for batch in batches] == [2, 1]

    def test_batches_invalid_size(self) -> None:
        with pytest.raises(ValueError, match="max_batch_size"):
            InferenceRequest((1.0,), (1,)).batches(0)

    def test_flatten_batches(self) -> None:
        assert flatten_batches(None, [[1, 2], [3.5]]) == [1.0, 2.0, 3.5]  # type: ignore[arg-type]


class TestDemoBackend:
    def test_softmax_output(self) -> None:
        backend = DemoBackend(output_size=2)
        result = backend.infer(InferenceRequest((1.0, 3.0, 5.0, 7.0), (4,)))
        assert len(result) == 2
        assert math.isclose(sum(result), 1.0)
        assert result[1] > result[0]
        assert backend.health() is True

    def test_small_input_and_invalid_size(self) -> None:
        with pytest.raises(ValueError, match="output_size"):
            DemoBackend(output_size=0)
        result = DemoBackend(output_size=5).infer(InferenceRequest((1.0,), (1,)))
        assert result == [1.0]

    def test_empty_tensor(self) -> None:
        with pytest.raises(BackendError, match="empty"):
            DemoBackend().infer(InferenceRequest((), ()))


class TestTritonBackend:
    def build(self) -> TritonHTTPBackend:
        return TritonHTTPBackend("http://triton:8001", "m")

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="url"):
            TritonHTTPBackend("", "m")
        with pytest.raises(ValueError, match="timeout"):
            TritonHTTPBackend("http://x", "m", timeout_s=0)

    def test_infer_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_urlopen(request: object, timeout: float) -> FakeResponse:
            captured["url"] = request.full_url  # type: ignore[attr-defined]
            captured["body"] = json.loads(request.data)  # type: ignore[attr-defined]
            return FakeResponse({"outputs": [{"data": [0.1, 0.9]}]})

        monkeypatch.setattr("triton_gateway.backends.urllib.request.urlopen", fake_urlopen)
        result = self.build().infer(InferenceRequest((1.0, 2.0), (1, 2)))
        assert result == [0.1, 0.9]
        assert captured["url"] == "http://triton:8001/v2/models/m/infer"
        assert captured["body"]["inputs"][0]["datatype"] == "FP32"

    @pytest.mark.parametrize(
        "payload,message",
        [
            ({}, "missing outputs"),
            ({"outputs": []}, "missing outputs"),
            ({"outputs": [{}]}, "missing outputs"),
            ({"outputs": [{"data": "x"}]}, "numeric list"),
            ({"outputs": [{"data": [1, True]}]}, "numeric list"),
        ],
    )
    def test_infer_bad_response(
        self, monkeypatch: pytest.MonkeyPatch, payload: object, message: str
    ) -> None:
        monkeypatch.setattr(
            "triton_gateway.backends.urllib.request.urlopen",
            lambda request, timeout: FakeResponse(payload),
        )
        with pytest.raises(BackendError, match=message):
            self.build().infer(InferenceRequest((1.0,), (1,)))

    def test_health(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "triton_gateway.backends.urllib.request.urlopen",
            lambda request, timeout: FakeResponse({}),
        )
        assert self.build().health() is True

    def test_health_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fail(request: object, timeout: float) -> FakeResponse:
            raise urllib.error.URLError("down")

        monkeypatch.setattr("triton_gateway.backends.urllib.request.urlopen", fail)
        assert self.build().health() is False

    def test_invalid_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "triton_gateway.backends.urllib.request.urlopen",
            lambda request, timeout: FakeResponse(b"not-json"),
        )
        with pytest.raises(BackendError, match="invalid JSON"):
            self.build().infer(InferenceRequest((1.0,), (1,)))

    def test_invalid_utf8(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "triton_gateway.backends.urllib.request.urlopen",
            lambda request, timeout: FakeResponse(b"\xff"),
        )
        with pytest.raises(BackendError, match="invalid JSON"):
            self.build().infer(InferenceRequest((1.0,), (1,)))


class TestMetrics:
    def test_render_empty(self) -> None:
        assert Metrics().render() == ""

    def test_counters_and_histograms(self) -> None:
        metrics = Metrics(buckets=(0.1, 1.0))
        metrics.increment("requests_total")
        metrics.increment("requests_total", 2)
        metrics.observe("latency_seconds", 0.05)
        metrics.observe("latency_seconds", 2.0)
        assert metrics.counter("requests_total") == 3
        assert metrics.counter("missing") == 0
        assert metrics.observations("latency_seconds") == (0.05, 2.0)
        assert metrics.observations("missing") == ()
        text = metrics.render()
        assert "requests_total 3" in text
        assert 'latency_seconds_bucket{le="0.1"} 1' in text
        assert 'latency_seconds_bucket{le="1"} 1' in text
        assert 'latency_seconds_bucket{le="+Inf"} 2' in text
        assert "latency_seconds_sum 2.05" in text
        assert text.endswith("\n")

    def test_validation(self) -> None:
        with pytest.raises(MetricsError):
            Metrics(buckets=())
        with pytest.raises(MetricsError):
            Metrics(buckets=(1.0, 0.1))
        with pytest.raises(MetricsError):
            Metrics(buckets=(0,))
        metrics = Metrics()
        with pytest.raises(MetricsError):
            metrics.increment("bad-name")
        with pytest.raises(MetricsError):
            metrics.increment("ok", -1)
        with pytest.raises(MetricsError):
            metrics.observe("ok", -1)


class TestService:
    def build(self, backend: object | None = None, clock: object | None = None) -> GatewayService:
        ticks = iter([1.0, 1.25, 1.5, 2.0, 2.25])
        return GatewayService(
            GatewayConfig(max_batch_size=2, max_request_elements=16),
            backend=backend,  # type: ignore[arg-type]
            clock=clock or (lambda: next(ticks)),  # type: ignore[arg-type]
        )

    def test_success_and_batching(self) -> None:
        service = self.build()
        result = service.predict(
            {"input_data": list(range(8)), "input_shape": [4, 2]}
        )
        assert result.success is True
        assert result.status_code == 200
        assert result.batches == 2
        assert result.output
        assert result.elapsed_ms == 250.0
        assert service.metrics.counter("inference_requests_total") == 1
        assert service.metrics.observations("inference_batch_size") == (2.0,)

    def test_validation_error(self) -> None:
        service = self.build()
        result = service.predict({"input_shape": [1]})
        assert result.success is False
        assert result.status_code == 400
        assert result.error is not None
        assert service.metrics.counter("inference_errors_total") == 1
        assert result.to_dict()["output"] == []

    def test_too_large(self) -> None:
        result = self.build().predict(
            {"input_data": list(range(17)), "input_shape": [17]}
        )
        assert result.status_code == 413

    def test_backend_error(self) -> None:
        class Failing:
            def infer(self, request: InferenceRequest) -> list[float]:
                raise BackendError("triton down")

            def health(self) -> bool:
                return False

        result = self.build(Failing()).predict(
            {"input_data": [1.0], "input_shape": [1]}
        )
        assert result.status_code == 503
        assert result.error == "triton down"

    def test_unexpected_backend_error(self) -> None:
        class Exploding:
            def infer(self, request: InferenceRequest) -> list[float]:
                raise RuntimeError("bug")

            def health(self) -> bool:
                raise RuntimeError("bug")

        service = self.build(Exploding())
        result = service.predict({"input_data": [1.0], "input_shape": [1]})
        assert result.status_code == 500
        assert result.error == "inference failed: bug"
        ready, body = service.ready()
        assert ready is False
        assert body["status"] == "unavailable"

    def test_health_and_model_info(self) -> None:
        service = self.build()
        assert service.health()["status"] == "ok"
        assert service.ready() == (True, {"status": "ready", "model": "onnx_wasm_model"})
        info = service.model_info()
        assert info["max_batch_size"] == 2
        assert "inference_requests_total" not in service.metrics_text()

    def test_default_backend_and_metrics(self) -> None:
        service = GatewayService(clock=lambda: 1.0)
        assert isinstance(service.backend, DemoBackend)
        assert isinstance(service.metrics, Metrics)
        assert service.predict({"input_data": [1.0], "input_shape": [1]}).elapsed_ms == 0.0

    def test_prediction_result_to_dict(self) -> None:
        result = PredictionResult(True, (0.5,), None, 200, 1.5, 1)
        assert result.to_dict() == {
            "success": True,
            "output": [0.5],
            "error": None,
            "status_code": 200,
            "elapsed_ms": 1.5,
            "batches": 1,
        }


class TestCli:
    def test_demo_text(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--demo"]) == 0
        out = capsys.readouterr().out
        assert "status: 200" in out
        assert "inference_requests_total 1" in out

    def test_demo_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--demo", "--json"]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["result"]["success"] is True
        assert report["ready"]["status"] == "ready"

    def test_model_config(self, tmp_path: object, capsys: pytest.CaptureFixture[str]) -> None:
        path = tmp_path / "config.pbtxt"  # type: ignore[operator]
        path.write_text(MODEL_CONFIG, encoding="utf-8")
        assert main(["--model-config", str(path)]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["name"] == "onnx_wasm_model"
        assert payload["inputs"][0]["dims"] == [3, 224, 224]

    def test_missing_config(self, tmp_path: object, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--model-config", str(tmp_path / "missing")]) == 2  # type: ignore[operator]
        assert "error:" in capsys.readouterr().err

    def test_invalid_config(self, tmp_path: object, capsys: pytest.CaptureFixture[str]) -> None:
        path = tmp_path / "bad.pbtxt"  # type: ignore[operator]
        path.write_text("", encoding="utf-8")
        assert main(["--model-config", str(path)]) == 2
        assert "empty" in capsys.readouterr().err

    def test_no_action(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main([]) == 2
        assert "provide" in capsys.readouterr().err

    def test_invalid_output_size(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--demo", "--output-size", "0"]) == 2
        assert "output_size" in capsys.readouterr().err

    def test_failure_exit_code(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(
            "triton_gateway.__main__.GatewayService.predict",
            lambda self, payload: PredictionResult(False, error="boom", status_code=500),
        )
        assert main(["--demo"]) == 1
        assert "status: 500" in capsys.readouterr().out

    def test_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0
        assert "triton_gateway" in capsys.readouterr().out

    def test_demo_payload_shape(self) -> None:
        payload = demo_payload()
        assert payload["input_shape"] == [1, 3, 4, 4]
        assert len(payload["input_data"]) == 48  # type: ignore[arg-type]

    def test_parser(self) -> None:
        parser = build_parser()
        assert parser.prog == "triton_gateway"
