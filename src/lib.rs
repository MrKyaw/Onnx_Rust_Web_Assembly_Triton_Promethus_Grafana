use tract_onnx::prelude::*;
use wasm_bindgen::prelude::*;

// Use wee_alloc as the global allocator for WASM
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

// Initialize panic hook for better error messages
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct OnnxModel {
    model: tract_onnx::prelude::SimplePlan<
        tract_onnx::prelude::TypedFact,
        Box<dyn tract_onnx::prelude::TypedOp>,
        tract_onnx::prelude::Graph<tract_onnx::prelude::TypedFact, Box<dyn tract_onnx::prelude::TypedOp>>,
    >,
}

#[wasm_bindgen]
impl OnnxModel {
    #[wasm_bindgen(constructor)]
    pub fn new(model_data: &[u8]) -> Result<OnnxModel, JsValue> {
        let model = tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(model_data))
            .map_err(|e| JsValue::from_str(&format!("Model loading failed: {}", e)))?
            .into_optimized()
            .map_err(|e| JsValue::from_str(&format!("Optimization failed: {}", e)))?
            .into_runnable()
            .map_err(|e| JsValue::from_str(&format!("Plan creation failed: {}", e)))?;

        Ok(OnnxModel { model })
    }

    #[wasm_bindgen]
    pub fn predict(&self, input_data: &[f32], input_shape: &[usize]) -> Result<Vec<f32>, JsValue> {
        let input_tensor = tract_ndarray::ArrayD::from_shape_vec(
            input_shape,
            input_data.to_vec(),
        ).map_err(|e| JsValue::from_str(&format!("Tensor creation failed: {}", e)))?;

        let output = self.model.run(tvec![input_tensor.into()])
            .map_err(|e| JsValue::from_str(&format!("Inference failed: {}", e)))?;

        let output_array = output[0]
            .to_array_view::<f32>()
            .map_err(|e| JsValue::from_str(&format!("Output conversion failed: {}", e)))?;

        Ok(output_array.iter().cloned().collect())
    }
}