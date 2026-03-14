//! Vulkan Tensor Matching Library
//!
//! Implements high-performance template matching using Vulkan compute shaders
//! for GPU-accelerated image processing.

pub mod error;
pub mod image;
pub mod vulkan;

#[cfg(feature = "python")]
pub mod py;

pub use error::{Result, TensorMatchingError};
pub use image::{
    image_data_to_rgb_image, loader::ImageData, loader::TestShape,
    tensor_matcher::TensorTemplateMatch, tensor_matcher::VulkanTensorMatcher,
};
