//! Python bindings for vulkan_tensor_matching library
//!
//! This module provides Python-compatible wrappers for the Rust library functions.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Library author
pub const AUTHOR: &str = "jer, <alphastrata@gmail.com>";

/// Result of processing numbers
#[pyclass(name = "ProcessResult")]
#[derive(Clone)]
pub struct PyProcessResult {
    #[pyo3(get)]
    pub sum: f64,
    #[pyo3(get)]
    pub average: f64,
    #[pyo3(get)]
    pub min: f64,
    #[pyo3(get)]
    pub max: f64,
    #[pyo3(get)]
    pub count: usize,
}

#[pymethods]
impl PyProcessResult {
    fn __repr__(&self) -> String {
        format!(
            "ProcessResult(sum={}, average={}, min={}, max={}, count={})",
            self.sum, self.average, self.min, self.max, self.count
        )
    }
}

/// A person with name, age, and email
#[pyclass(name = "Person")]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyPerson {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub age: u32,
    #[pyo3(get, set)]
    pub email: Option<String>,
}

#[pymethods]
impl PyPerson {
    #[new]
    #[pyo3(signature = (name, age, email=None))]
    fn new(name: String, age: u32, email: Option<String>) -> PyResult<Self> {
        if name.trim().is_empty() {
            return Err(PyValueError::new_err("Name cannot be empty"));
        }
        Ok(Self { name, age, email })
    }

    fn is_adult(&self) -> bool {
        self.age >= 18
    }

    fn greet(&self) -> String {
        format!(
            "Hello, my name is {} and I am {} years old.",
            self.name, self.age
        )
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(self).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, json_str: &str) -> PyResult<Self> {
        serde_json::from_str(json_str).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "Person(name='{}', age={}, email={:?})",
            self.name, self.age, self.email
        )
    }
}

/// A data point with x, y coordinates and a label
#[pyclass(name = "DataPoint")]
#[derive(Clone)]
pub struct PyDataPoint {
    #[pyo3(get, set)]
    pub x: f64,
    #[pyo3(get, set)]
    pub y: f64,
    #[pyo3(get, set)]
    pub label: String,
}

#[pymethods]
impl PyDataPoint {
    #[new]
    fn new(x: f64, y: f64, label: String) -> Self {
        Self { x, y, label }
    }

    fn distance_from_origin(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    fn scale(&mut self, factor: f64) {
        self.x *= factor;
        self.y *= factor;
    }

    fn __repr__(&self) -> String {
        format!("DataPoint(x={}, y={}, label='{}')", self.x, self.y, self.label)
    }
}

/// Process a list of numbers and return statistics
#[pyfunction]
#[pyo3(signature = (numbers))]
fn process_numbers(numbers: Vec<f64>) -> PyResult<PyProcessResult> {
    if numbers.is_empty() {
        return Err(PyValueError::new_err("Cannot process empty list"));
    }

    let sum: f64 = numbers.iter().sum();
    let count = numbers.len();
    let average = sum / count as f64;
    let min = numbers
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let max = numbers
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    Ok(PyProcessResult {
        sum,
        average,
        min,
        max,
        count,
    })
}

/// Concatenate strings with an optional separator
#[pyfunction]
#[pyo3(signature = (strings, separator=None))]
fn concatenate_strings(strings: Vec<String>, separator: Option<String>) -> String {
    let sep = separator.unwrap_or_else(|| ", ".to_string());
    strings.join(&sep)
}

/// Create a new Person
#[pyfunction]
#[pyo3(signature = (name, age, email=None))]
fn create_person(name: String, age: u32, email: Option<String>) -> PyResult<PyPerson> {
    if name.trim().is_empty() {
        return Err(PyValueError::new_err("Name cannot be empty"));
    }
    Ok(PyPerson { name, age, email })
}

/// Analyze a list of data points
#[pyfunction]
fn analyze_data(py: Python<'_>, points: Vec<PyDataPoint>) -> PyResult<PyObject> {
    let dict = PyDict::new(py);

    let total_points = points.len();
    let sum_x: f64 = points.iter().map(|p| p.x).sum();
    let sum_y: f64 = points.iter().map(|p| p.y).sum();
    let average_x = sum_x / total_points as f64;
    let average_y = sum_y / total_points as f64;

    // Count labels
    let mut label_counts: HashMap<String, usize> = HashMap::new();
    for point in &points {
        *label_counts.entry(point.label.clone()).or_insert(0) += 1;
    }

    // Convert label_counts to Python dict
    let label_dict = PyDict::new(py);
    for (key, value) in label_counts {
        label_dict.set_item(key, value)?;
    }

    dict.set_item("total_points", total_points)?;
    dict.set_item("average_x", average_x)?;
    dict.set_item("average_y", average_y)?;
    dict.set_item("label_counts", label_dict)?;

    Ok(dict.into())
}

/// Process mixed data types
#[pyfunction]
fn process_mixed_data(py: Python<'_>, items: Vec<PyObject>) -> PyResult<PyObject> {
    let dict = PyDict::new(py);

    for (i, item) in items.iter().enumerate() {
        let key = format!("item_{}", i);
        let value = item.bind(py);
        let value_str = if let Ok(b) = value.downcast::<pyo3::types::PyBool>() {
            format!("bool:{}", b.is_true())
        } else if let Ok(s) = value.downcast::<pyo3::types::PyString>() {
            format!("str:{}", s.to_string_lossy())
        } else if let Ok(n) = value.downcast::<pyo3::types::PyInt>() {
            format!("int:{}", n.extract::<i64>()?)
        } else if let Ok(n) = value.downcast::<pyo3::types::PyFloat>() {
            format!("float:{}", n.extract::<f64>()?)
        } else {
            "unknown".to_string()
        };
        dict.set_item(key, value_str)?;
    }

    Ok(dict.into())
}

/// Generate Fibonacci sequence
#[pyfunction]
fn fibonacci(n: usize) -> Vec<u64> {
    if n == 0 {
        return vec![];
    }

    let mut seq = Vec::with_capacity(n);
    let (mut a, mut b) = (0u64, 1u64);

    for _ in 0..n {
        seq.push(a);
        let next = a + b;
        a = b;
        b = next;
    }

    seq
}

/// Python module for rust_python_lib
#[pymodule]
fn rust_python_lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("VERSION", VERSION)?;
    m.add("AUTHOR", AUTHOR)?;
    m.add_class::<PyProcessResult>()?;
    m.add_class::<PyPerson>()?;
    m.add_class::<PyDataPoint>()?;
    m.add_function(wrap_pyfunction!(process_numbers, m)?)?;
    m.add_function(wrap_pyfunction!(concatenate_strings, m)?)?;
    m.add_function(wrap_pyfunction!(create_person, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_data, m)?)?;
    m.add_function(wrap_pyfunction!(process_mixed_data, m)?)?;
    m.add_function(wrap_pyfunction!(fibonacci, m)?)?;
    Ok(())
}
