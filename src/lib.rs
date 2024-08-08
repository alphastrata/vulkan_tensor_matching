//! Vulkan Tensor Matching Library
//!
//! Implements high-performance rotation-invariant template matching using Vulkan compute shaders
//! and tensor mathematics for GPU-accelerated image processing.
//!
//! Based on: "Tensorial template matching for fast cross-correlation with rotations"
//! Martinez-Sanchez et al., arXiv:2408.02398v1 [cs.CV]
//!
//! ## Key Features from the Paper
//!
//! ### Tensorial Template Matching (Section 3, p. 4-8)
//! - **Algorithm 1**: Tensor template generation by integration over all rotations (p. 5)
//! - **Algorithm 2**: Tensorial field computation using correlation tensors (p. 6)
//! - **Section 3.1**: Optimal rotation determination using SS-HOPM (p. 6-7)
//! - **Section 3.2**: Instance positions using Frobenius norm (p. 7)
//!
//! ### Key Advantages Claimed in the Paper
//!
//! 1. **Computational Complexity Independence** (p. 3, 7):
//!    - Traditional TM: O((360/ε)³) for 3D images with angular accuracy ε
//!    - TTM: O(1) - independent of angular accuracy
//!
//! 2. **Reduced Correlation Count** (p. 6):
//!    - Traditional TM: Thousands of correlations required for rotation sampling
//!    - TTM: Only 35 correlations for degree-4 symmetric tensors in 3D
//!
//! 3. **Complete Rotation Information** (p. 4-5):
//!    - Integrates template information over all rotations into tensor field
//!    - No need to explicitly sample rotation space during matching
//!
//! ## Implementation Structure
//!
//! The library implements the three-stage pipeline described in the paper:
//!
//! 1. **Tensor Generation Stage** (`tensor_matcher::VulkanTensorMatcher::generate_template_tensor_field`)
//!    - Implements Algorithm 1: Tensor template generation (p. 5)
//!    - Integrates template over all rotations to create tensor field T(t)
//!
//! 2. **Tensorial Correlation Stage** (`tensor_matcher::VulkanTensorMatcher::compute_tensorial_correlation`)
//!    - Implements Algorithm 2: Tensorial field computation (p. 6)
//!    - Computes correlation tensor field Cn(x) = w(x)(f ⋆ T(t))(x)
//!
//! 3. **Peak Detection Stage** (`tensor_matcher::VulkanTensorMatcher::detect_peaks`)
//!    - Implements Section 3.1: Optimal rotation determination (p. 6-7)
//!    - Implements Section 3.2: Instance positions using Frobenius norm (p. 7)
//!    - Uses SS-HOPM for eigenvalue decomposition to find optimal rotations

pub mod error;
pub mod image;
pub mod tensor;
pub mod vulkan;

pub use error::{Result, TensorMatchingError};
pub use image::{
    annotate_image_with_matches, image_data_to_rgb_image, loader::ImageData, loader::MatchTemplateMethod, loader::TestShape,
    matcher::TemplateMatch, matcher::VulkanTensorMatcher, tensor_matcher::TensorTemplateMatch,
};