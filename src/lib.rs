//! Vulkan Tensor Matching Library
//!
//! Implements high-performance rotation-invariant template matching using Vulkan compute shaders
//! and tensor mathematics for GPU-accelerated image processing.
//!
//! Based on: "Tensorial template matching for fast cross-correlation with rotations"
//! Martinez-Sanchez et al., arXiv:2408.02398v1 [cs.CV]

pub mod error;
pub mod image;
pub mod tensor;
pub mod vulkan;

#[cfg(feature = "python")]
pub mod py;

pub use error::{Result, TensorMatchingError};
pub use image::{
    image_data_to_rgb_image, loader::ImageData, loader::TestShape,
    tensor_matcher::TensorTemplateMatch, tensor_matcher::VulkanTensorMatcher,
};
