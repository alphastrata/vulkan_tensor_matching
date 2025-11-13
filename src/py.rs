//! Python bindings for vulkan_tensor_matching library
//!
//! This module provides Python-compatible wrappers for the Rust library functions.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};

// Re-export the actual library types
use crate::image::loader::ImageData;
use crate::image::tensor_matcher::VulkanTensorMatcher;

/// Library version
pub const VERSION: &str = "0.1.5-DEBUG-1";

/// Library author
pub const AUTHOR: &str = "jer, <alphastrata@gmail.com>";

/// Initialize Vulkan environment for Python (macOS/MoltenVK specific)
#[cfg(target_os = "macos")]
fn ensure_vulkan_ready() -> PyResult<()> {
    // macOS logic same as before...
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn ensure_vulkan_ready() -> PyResult<()> {
    Ok(())
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
            self.width,
            self.height,
            self.channels,
            self.data.len()
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

/// Vulkan Tensor Matcher - Rotation-invariant GPU-accelerated matching
#[pyclass(name = "VulkanTensorMatcher")]
pub struct PyVulkanTensorMatcher {
    matcher: Option<VulkanTensorMatcher>,
}

#[pymethods]
impl PyVulkanTensorMatcher {
    #[new]
    fn new() -> PyResult<Self> {
        let matcher = VulkanTensorMatcher::new().map_err(|e| {
            PyValueError::new_err(format!("Failed to initialize Vulkan Tensor matcher: {}", e))
        })?;
        Ok(Self {
            matcher: Some(matcher),
        })
    }

    fn __del__(&mut self) {
        // Explicitly drop the matcher to ensure Vulkan cleanup happens
        // while the Python interpreter is still fully functional
        self.matcher = None;
    }

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
            .map_err(|e| {
                PyValueError::new_err(format!("Tensor template matching failed: {}", e))
            })?;

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

/// Python module for vulkan_tensor_matching
#[pymodule]
fn vulkan_tensor_matching(m: &Bound<'_, PyModule>) -> PyResult<()> {
    ensure_vulkan_ready()?;
    m.add("VERSION", VERSION)?;
    m.add("AUTHOR", AUTHOR)?;
    m.add_class::<PyImageData>()?;
    m.add_class::<PyTemplateMatch>()?;
    m.add_class::<PyVulkanTensorMatcher>()?;
    Ok(())
}
