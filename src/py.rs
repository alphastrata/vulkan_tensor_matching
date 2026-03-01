//! Python bindings for vulkan_tensor_matching library
//!
//! This module provides Python-compatible wrappers for the Rust library functions.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Re-export the actual library types
use crate::image::{
    loader::ImageData,
    matcher::{TemplateMatch, VulkanTensorMatcher},
    tensor_matcher::TensorTemplateMatch,
};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Library author
pub const AUTHOR: &str = "jer, <alphastrata@gmail.com>";

/// Initialize Vulkan environment for Python (macOS/MoltenVK specific)
#[cfg(target_os = "macos")]
fn init_vulkan_env() {
    // On macOS, ensure MoltenVK can be found
    // This is called before any Vulkan operations
    use std::env;
    
    // Set VK_ICD_FILENAMES if not already set
    if env::var("VK_ICD_FILENAMES").is_err() {
        // Common MoltenVK ICD locations on macOS
        let icd_paths = [
            "/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json",
            "/usr/local/etc/vulkan/icd.d/MoltenVK_icd.json",
        ];
        for path in icd_paths {
            if std::path::Path::new(path).exists() {
                unsafe { env::set_var("VK_ICD_FILENAMES", path) };
                break;
            }
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn init_vulkan_env() {
    // No special setup needed on other platforms
}

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
    #[pyo3(get)]
    pub channels: u32,
}

#[pymethods]
impl PyImageData {
    #[new]
    #[pyo3(signature = (data, width, height, channels=1))]
    fn new(data: Vec<f32>, width: u32, height: u32, channels: u32) -> PyResult<Self> {
        let expected_len = (width * height * channels) as usize;
        if data.len() != expected_len {
            return Err(PyValueError::new_err(format!(
                "Data length {} does not match dimensions {}x{}x{}={}",
                data.len(),
                width,
                height,
                channels,
                expected_len
            )));
        }
        Ok(Self {
            data,
            width,
            height,
            channels,
        })
    }

    /// Load an image from file path
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        ImageData::from_file(path)
            .map(|img| PyImageData {
                data: img.data,
                width: img.width,
                height: img.height,
                channels: img.channels,
            })
            .map_err(|e| PyValueError::new_err(format!("Failed to load image: {}", e)))
    }

    fn __repr__(&self) -> String {
        format!(
            "ImageData(width={}, height={}, channels={}, data_len={})",
            self.width, self.height, self.channels, self.data.len()
        )
    }
}

/// Template match result
#[pyclass(name = "TemplateMatch")]
#[derive(Clone, Serialize, Deserialize)]
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
            "TemplateMatch(x={}, y={}, correlation={:.4}, rotation={:.2}°, confidence={:.4})",
            self.x, self.y, self.correlation, self.rotation_angle, self.confidence
        )
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("x", self.x)?;
        dict.set_item("y", self.y)?;
        dict.set_item("correlation", self.correlation)?;
        dict.set_item("rotation_angle", self.rotation_angle)?;
        dict.set_item("confidence", self.confidence)?;
        Ok(dict.into())
    }
}

/// Vulkan Tensor Matcher - GPU-accelerated template matching
#[pyclass(name = "VulkanTensorMatcher")]
pub struct PyVulkanTensorMatcher {
    matcher: Option<VulkanTensorMatcher>,
}

#[pymethods]
impl PyVulkanTensorMatcher {
    #[new]
    fn new() -> PyResult<Self> {
        let matcher = VulkanTensorMatcher::new()
            .map_err(|e| PyValueError::new_err(format!("Failed to initialize Vulkan matcher: {}", e)))?;
        Ok(Self {
            matcher: Some(matcher),
        })
    }

    /// Perform template matching
    ///
    /// Args:
    ///     target_image: The target image to search in
    ///     template_image: The template to match
    ///     correlation_threshold: Minimum correlation threshold (0.0-1.0)
    ///     max_matches: Maximum number of matches to return
    ///
    /// Returns:
    ///     List of TemplateMatch objects
    fn match_template(
        &self,
        target_image: &PyImageData,
        template_image: &PyImageData,
        correlation_threshold: f32,
        max_matches: usize,
    ) -> PyResult<Vec<PyTemplateMatch>> {
        let matcher = self
            .matcher
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Matcher not initialized"))?;

        let target = ImageData {
            data: target_image.data.clone(),
            width: target_image.width,
            height: target_image.height,
            channels: target_image.channels,
        };

        let template = ImageData {
            data: template_image.data.clone(),
            width: template_image.width,
            height: template_image.height,
            channels: template_image.channels,
        };

        let matches = matcher
            .match_template(&target, &template, correlation_threshold, max_matches)
            .map_err(|e| PyValueError::new_err(format!("Template matching failed: {}", e)))?;

        Ok(matches
            .into_iter()
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

/// Match template method enum
#[pyclass(name = "MatchTemplateMethod", eq, eq_int)]
#[derive(Clone, Copy, PartialEq)]
pub enum PyMatchTemplateMethod {
    SumOfSquaredErrors,
    SumOfSquaredErrorsNormalized,
    CrossCorrelation,
    CrossCorrelationNormalized,
}

#[pymethods]
impl PyMatchTemplateMethod {
    #[staticmethod]
    fn sum_of_squared_errors() -> Self {
        PyMatchTemplateMethod::SumOfSquaredErrors
    }

    #[staticmethod]
    fn sum_of_squared_errors_normalized() -> Self {
        PyMatchTemplateMethod::SumOfSquaredErrorsNormalized
    }

    #[staticmethod]
    fn cross_correlation() -> Self {
        PyMatchTemplateMethod::CrossCorrelation
    }

    #[staticmethod]
    fn cross_correlation_normalized() -> Self {
        PyMatchTemplateMethod::CrossCorrelationNormalized
    }
}

/// CPU-based template matching (for comparison/testing)
#[pyfunction]
fn match_template_cpu(
    image: &PyImageData,
    template: &PyImageData,
    method: PyMatchTemplateMethod,
) -> PyResult<PyImageData> {
    use crate::image::loader::MatchTemplateMethod;

    let method = match method {
        PyMatchTemplateMethod::SumOfSquaredErrors => MatchTemplateMethod::SumOfSquaredErrors,
        PyMatchTemplateMethod::SumOfSquaredErrorsNormalized => {
            MatchTemplateMethod::SumOfSquaredErrorsNormalized
        }
        PyMatchTemplateMethod::CrossCorrelation => MatchTemplateMethod::CrossCorrelation,
        PyMatchTemplateMethod::CrossCorrelationNormalized => {
            MatchTemplateMethod::CrossCorrelationNormalized
        }
    };

    let img = ImageData {
        data: image.data.clone(),
        width: image.width,
        height: image.height,
        channels: image.channels,
    };

    let tmpl = ImageData {
        data: template.data.clone(),
        width: template.width,
        height: template.height,
        channels: template.channels,
    };

    let result = img.match_template(&tmpl, method);

    Ok(PyImageData {
        data: result.data,
        width: result.width,
        height: result.height,
        channels: result.channels,
    })
}

/// Find extreme values (min/max) in an image
#[pyfunction]
fn find_extremes(image: &PyImageData) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let img = ImageData {
            data: image.data.clone(),
            width: image.width,
            height: image.height,
            channels: image.channels,
        };

        let ((max_val, max_pos), (min_val, min_pos)) = img.find_extremes();

        let dict = PyDict::new(py);
        
        // Create max dict
        let max_dict = PyDict::new(py);
        max_dict.set_item("value", max_val)?;
        max_dict.set_item("x", max_pos.0)?;
        max_dict.set_item("y", max_pos.1)?;
        dict.set_item("max", max_dict)?;
        
        // Create min dict
        let min_dict = PyDict::new(py);
        min_dict.set_item("value", min_val)?;
        min_dict.set_item("x", min_pos.0)?;
        min_dict.set_item("y", min_pos.1)?;
        dict.set_item("min", min_dict)?;

        Ok(dict.into())
    })
}

/// Compress an image by averaging pixels in blocks
#[pyfunction]
fn compress_image(image: &PyImageData, factor: u32) -> PyResult<PyImageData> {
    let img = ImageData {
        data: image.data.clone(),
        width: image.width,
        height: image.height,
        channels: image.channels,
    };

    let result = img.compress(factor);

    Ok(PyImageData {
        data: result.data,
        width: result.width,
        height: result.height,
        channels: result.channels,
    })
}

/// Python module for rust_python_lib
#[pymodule]
fn rust_python_lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("VERSION", VERSION)?;
    m.add("AUTHOR", AUTHOR)?;
    m.add_class::<PyImageData>()?;
    m.add_class::<PyTemplateMatch>()?;
    m.add_class::<PyVulkanTensorMatcher>()?;
    m.add_class::<PyMatchTemplateMethod>()?;
    m.add_function(wrap_pyfunction!(match_template_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(find_extremes, m)?)?;
    m.add_function(wrap_pyfunction!(compress_image, m)?)?;
    Ok(())
}
