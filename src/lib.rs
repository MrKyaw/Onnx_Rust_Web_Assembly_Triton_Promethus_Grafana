use tract_onnx::prelude::*;
use wasm_bindgen::prelude::*;

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
        let expected = shape_product(input_shape).map_err(|e| JsValue::from_str(&e))?;
        if expected != input_data.len() {
            return Err(JsValue::from_str(&format!(
                "input_data length {} does not match shape product {}",
                input_data.len(),
                expected
            )));
        }

        let input_tensor = tract_onnx::prelude::Tensor::from_shape(input_shape, input_data)
            .map_err(|e| JsValue::from_str(&format!("Tensor creation failed: {}", e)))?;

        let output = self.model.run(tvec![input_tensor.into()])
            .map_err(|e| JsValue::from_str(&format!("Inference failed: {}", e)))?;

        let output_array = output[0]
            .to_array_view::<f32>()
            .map_err(|e| JsValue::from_str(&format!("Output conversion failed: {}", e)))?;

        Ok(output_array.iter().cloned().collect())
    }
}

/// Validate a tensor shape and return its element count.
///
/// Kept outside the wasm-bindgen surface so it can be unit-tested natively.
fn shape_product(shape: &[usize]) -> Result<usize, String> {
    if shape.is_empty() {
        return Err("shape must not be empty".to_string());
    }
    shape.iter().try_fold(1usize, |acc, &dim| {
        if dim == 0 {
            return Err("shape dimensions must be > 0".to_string());
        }
        acc.checked_mul(dim)
            .ok_or_else(|| "shape product overflows usize".to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::shape_product;

    #[test]
    fn computes_element_count() {
        assert_eq!(shape_product(&[1, 3, 224, 224]), Ok(150528));
        assert_eq!(shape_product(&[7]), Ok(7));
    }

    #[test]
    fn rejects_empty_shape() {
        assert_eq!(
            shape_product(&[]),
            Err("shape must not be empty".to_string())
        );
    }

    #[test]
    fn rejects_zero_dimension() {
        assert!(shape_product(&[1, 0, 3]).is_err());
    }

    #[test]
    fn rejects_overflow() {
        assert!(shape_product(&[usize::MAX, 2]).is_err());
    }
}