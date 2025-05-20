
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from tritonclient.http import InferenceServerClient, InferInput
from prometheus_fastapi_instrumentator import Instrumentator
import logging
from pydantic import BaseModel

app = FastAPI()

# Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

# Triton client
triton_client = InferenceServerClient(url="localhost:8000")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InferenceRequest(BaseModel):
    input_data: list[float]
    input_shape: list[int]

@app.get("/")
def read_root():
    return {"message": "ONNX Rust WASM with FastAPI"}

@app.post("/predict")
async def predict(request: InferenceRequest):
    try:
        # Convert input to numpy array
        input_array = np.array(request.input_data, dtype=np.float32)
        input_array = input_array.reshape(request.input_shape)
        
        # Prepare Triton inputs
        inputs = [InferInput("input", input_array.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_array)
        
        # Send to Triton
        result = triton_client.infer(
            model_name="onnx_wasm_model",
            inputs=inputs
        )
        
        # Get output
        output = result.as_numpy("output")
        
        return {
            "success": True,
            "output": output.tolist()
        }
    except Exception as e:
        logging.error(f"Inference error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}