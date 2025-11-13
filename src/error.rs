use ash::vk;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TensorMatchingError {
    #[error("Vulkan error: {0}")]
    VulkanError(#[from] ash::vk::Result),

    #[error("Vulkan loading error: {0}")]
    VulkanLoadingError(#[from] ash::LoadingError),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Image error: {0}")]
    ImageError(#[from] image::ImageError),

    #[error("GPU allocation error: {0}")]
    AllocationError(#[from] gpu_allocator::AllocationError),

    #[error("Null byte error: {0}")]
    NullByteError(#[from] std::ffi::FromBytesWithNulError),

    #[error("Pipeline creation error: {0:?}")]
    PipelineCreationError((Vec<vk::Pipeline>, vk::Result)),

    #[error("Other error: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, TensorMatchingError>;
