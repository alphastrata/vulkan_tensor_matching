use thiserror::Error;
use ash::{vk};

#[derive(Error, Debug)]
pub enum TensorMatchingError {
    #[error("Vulkan error: {0}")]
    VulkanError(#[from] ash::vk::Result),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Image error: {0}")]
    ImageError(#[from] image::ImageError),

    #[error("GPU allocator error: {0}")]
    GpuAllocatorError(#[from] gpu_allocator::AllocationError),

    #[error("No suitable GPU found")]
    NoGpuFound,

    #[error("Vulkan entry load error: {0}")]
    VulkanEntryLoadError(String),

    #[error("Unsupported Vulkan version")]
    UnsupportedVulkanVersion,

    #[error("Validation layers not available")]
    ValidationLayersNotAvailable,

    #[error("Shader compilation error: {0}")]
    ShaderCompilationError(String),

    #[error("FromBytesWithNul error: {0}")]
    FromBytesWithNulError(#[from] std::ffi::FromBytesWithNulError),

    #[error("Pipeline creation error: {0:?}")]
    PipelineCreationError((Vec<vk::Pipeline>, vk::Result)),

    #[error("Other error: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, TensorMatchingError>;