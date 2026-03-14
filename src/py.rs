//! Minimal Python bindings for vulkan_tensor_matching

use pyo3::prelude::*;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Python wrapper for ImageData
#[pyclass(name = "ImageData")]
#[derive(Clone)]
pub struct PyImageData {
    #[pyo3(get)]
    pub data: Vec<f32>,
    #[pyo3(get)]
    pub width: u32,
    #[pyo3(get)]
    pub height: u32,
}

#[pymethods]
impl PyImageData {
    #[new]
    #[pyo3(signature = (data, width, height))]
    fn new(data: Vec<f32>, width: u32, height: u32) -> PyResult<Self> {
        if data.len() != (width * height) as usize {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Data length must equal width * height",
            ));
        }
        Ok(Self { data, width, height })
    }

    /// Load image from file (grayscale)
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        use crate::image::loader::ImageData as RustImageData;
        let img = RustImageData::from_file(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self {
            data: img.data,
            width: img.width,
            height: img.height,
        })
    }
}

/// Python wrapper for VulkanTensorMatcher
#[pyclass(name = "VulkanTensorMatcher")]
pub struct PyVulkanTensorMatcher {
    inner: crate::image::tensor_matcher::VulkanTensorMatcher,
}

#[pymethods]
impl PyVulkanTensorMatcher {
    #[new]
    fn new() -> PyResult<Self> {
        let inner = crate::image::tensor_matcher::VulkanTensorMatcher::new()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Match template against target image
    /// 
    /// Args:
    ///     target: Target image
    ///     template: Template to find
    ///     threshold: Minimum correlation (0.0-1.0)
    ///     max_matches: Maximum matches to return
    /// 
    /// Returns:
    ///     List of TemplateMatch objects
    fn match_template(
        &self,
        target: &PyImageData,
        template: &PyImageData,
        threshold: f32,
        max_matches: usize,
    ) -> PyResult<Vec<PyTemplateMatch>> {
        let target_img = crate::image::loader::ImageData::new(
            target.data.clone(),
            target.width,
            target.height,
            1,
        );
        let template_img = crate::image::loader::ImageData::new(
            template.data.clone(),
            template.width,
            template.height,
            1,
        );

        let matches = self
            .inner
            .match_template(&target_img, &template_img, threshold, max_matches)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(matches
            .iter()
            .map(|m| PyTemplateMatch {
                x: m.x,
                y: m.y,
                correlation: m.correlation,
                rotation_angle: m.rotation_angle,
                confidence: m.confidence,
            })
            .collect())
    }
}

/// Template match result
#[pyclass(name = "TemplateMatch")]
#[derive(Clone)]
pub struct PyTemplateMatch {
    #[pyo3(get)]
    pub x: u32,
    #[pyo3(get)]
    pub y: u32,
    #[pyo3(get)]
    pub correlation: f32,
    #[pyo3(get)]
    pub rotation_angle: f32,
    #[pyo3(get)]
    pub confidence: f32,
}

#[pymethods]
impl PyTemplateMatch {
    fn __repr__(&self) -> String {
        format!(
            "TemplateMatch(x={}, y={}, correlation={:.3}, rotation={:.1}°)",
            self.x,
            self.y,
            self.correlation,
            self.rotation_angle.to_degrees()
        )
    }
}

/// Python module for vulkan_tensor_matching
#[pymodule]
fn vulkan_tensor_matching(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("VERSION", VERSION)?;
    m.add_class::<PyImageData>()?;
    m.add_class::<PyVulkanTensorMatcher>()?;
    m.add_class::<PyTemplateMatch>()?;
    Ok(())
}
