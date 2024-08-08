use ash::{vk, Device};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc, Allocation, AllocationCreateDesc, AllocationScheme};
use gpu_allocator::{AllocationSizes, MemoryLocation};
use crate::error::{Result, TensorMatchingError};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

/// Key for identifying buffer pools by size and usage
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct BufferPoolKey {
    size: u64,
    usage: vk::BufferUsageFlags,
    memory_location: MemoryLocation,
}

pub struct VulkanBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Option<Allocation>,
    pub size: u64,
    pub usage: vk::BufferUsageFlags,
}

#[derive(Clone)]
pub struct VulkanMemoryManager {
    device: Device,
    allocator: Arc<Mutex<Allocator>>,
    /// Pool of reusable buffers keyed by size, usage, and memory location
    buffer_pool: Arc<Mutex<HashMap<BufferPoolKey, Vec<VulkanBuffer>>>>,
}

impl std::fmt::Debug for VulkanMemoryManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanMemoryManager")
            .field("device", &"Device")
            .field("allocator", &self.allocator)
            .finish()
    }
}

impl VulkanMemoryManager {
    pub fn new(
        instance: &ash::Instance,
        device: Device,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let mut debug_settings = gpu_allocator::AllocatorDebugSettings::default();
        debug_settings.log_memory_information = true;
        debug_settings.log_leaks_on_shutdown = true;

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings,
            buffer_device_address: false,
            allocation_sizes: AllocationSizes::default(),
        }).map_err(TensorMatchingError::GpuAllocatorError)?;

        Ok(Self {
            device,
            allocator: Arc::new(Mutex::new(allocator)),
            buffer_pool: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Get a buffer from the pool or create a new one if none available
    pub fn get_or_create_buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        memory_location: MemoryLocation,
        name: &str,
    ) -> Result<VulkanBuffer> {
        let key = BufferPoolKey {
            size,
            usage,
            memory_location,
        };

        // Try to get a buffer from the pool
        {
            let mut pool = self.buffer_pool.lock().unwrap();
            if let Some(buffers) = pool.get_mut(&key)
                && let Some(buffer) = buffers.pop() {
                    return Ok(buffer);
                }
        }

        // No buffer available in pool, create a new one
        self.create_tensor_buffer(size, usage, memory_location, name)
    }

    /// Return a buffer to the pool for reuse
    pub fn return_buffer_to_pool(&self, buffer: VulkanBuffer) -> Result<()> {
        let key = BufferPoolKey {
            size: buffer.size,
            usage: buffer.usage,
            memory_location: match buffer.allocation.as_ref() {
                Some(_allocation) => {
                    // This is a simplification - in reality we'd need to track this separately
                    MemoryLocation::GpuOnly // Default assumption
                }
                None => MemoryLocation::Unknown,
            },
        };

        let mut pool = self.buffer_pool.lock().unwrap();
        pool.entry(key).or_default().push(buffer);

        Ok(())
    }

    /// Clear all buffers from the pool (destroys them)
    pub fn clear_buffer_pool(&self) -> Result<()> {
        let mut pool = self.buffer_pool.lock().unwrap();
        for (_, buffers) in pool.drain() {
            for buffer in buffers {
                self.destroy_buffer(buffer)?;
            }
        }
        Ok(())
    }

    /// Get statistics about buffer pool usage
    pub fn get_pool_statistics(&self) -> (usize, usize) {
        let pool = self.buffer_pool.lock().unwrap();
        let total_buffers: usize = pool.values().map(|v| v.len()).sum();
        let pool_types = pool.len();
        (total_buffers, pool_types)
    }

    /// Create a buffer optimised for tensor operations
    pub fn create_tensor_buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        memory_location: MemoryLocation,
        name: &str,
    ) -> Result<VulkanBuffer> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { self.device.create_buffer(&buffer_info, None) }?;
        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let allocation = self.allocator.lock().unwrap().allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: memory_location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;

        unsafe {
            self.device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
        }

        Ok(VulkanBuffer {
            buffer,
            allocation: Some(allocation),
            size,
            usage,
        })
    }

    /// Create a buffer specifically for 2D tensor fields
    pub fn create_tensor_field_buffer(&self, width: u32, height: u32) -> Result<VulkanBuffer> {
        // Each tensor has 8 f32 components (5 real + 3 padding for alignment)
        let tensor_size = std::mem::size_of::<f32>() * 8;
        let total_size = (width * height) as u64 * tensor_size as u64;

        self.create_tensor_buffer(
            total_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            "Tensor Field Buffer",
        )
    }

    /// Create a buffer for image data with optimal memory layout
    pub fn create_image_buffer(&self, width: u32, height: u32, channels: u32) -> Result<VulkanBuffer> {
        let size = (width * height * channels) as u64 * std::mem::size_of::<f32>() as u64;

        self.create_tensor_buffer(
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::CpuToGpu,
            "Image Buffer",
        )
    }

    /// Create a staging buffer for CPU-GPU data transfer
    pub fn create_staging_buffer(&self, size: u64) -> Result<VulkanBuffer> {
        self.create_tensor_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::CpuToGpu,
            "Staging Buffer",
        )
    }

    /// Upload data to GPU buffer
    pub fn upload_data<T: Copy>(&self, buffer: &VulkanBuffer, data: &[T]) -> Result<()> {
        if let Some(allocation) = &buffer.allocation {
            if allocation.mapped_ptr().is_some() {
                unsafe {
                    let mapped_ptr = allocation.mapped_ptr().unwrap().as_ptr() as *mut T;
                    std::ptr::copy_nonoverlapping(data.as_ptr(), mapped_ptr, data.len());
                }
            } else {
                return Err(TensorMatchingError::VulkanError(ash::vk::Result::ERROR_MEMORY_MAP_FAILED));
            }
        }
        Ok(())
    }

    /// Copy data from GPU buffer to host memory
    pub fn device_to_host<T: Copy>(&self, buffer: &VulkanBuffer, data: &mut [T]) -> Result<()> {
        if let Some(allocation) = &buffer.allocation {
            if allocation.mapped_ptr().is_some() {
                unsafe {
                    let mapped_ptr = allocation.mapped_ptr().unwrap().as_ptr() as *const T;
                    std::ptr::copy_nonoverlapping(mapped_ptr, data.as_mut_ptr(), data.len());
                }
            } else {
                return Err(TensorMatchingError::VulkanError(ash::vk::Result::ERROR_MEMORY_MAP_FAILED));
            }
        }
        Ok(())
    }

    pub fn destroy_buffer(&self, buffer: VulkanBuffer) -> Result<()> {
        unsafe {
            self.device.destroy_buffer(buffer.buffer, None);
        }
        if let Some(allocation) = buffer.allocation {
            self.allocator.lock().unwrap().free(allocation)?;
        }
        Ok(())
    }

    /// Destroy buffer or return it to pool for reuse
    pub fn destroy_or_pool_buffer(&self, buffer: VulkanBuffer) -> Result<()> {
        // For now, we'll just destroy the buffer
        // In a more sophisticated implementation, we might return it to a pool
        self.destroy_buffer(buffer)
    }
}

impl Drop for VulkanMemoryManager {
    fn drop(&mut self) {
        // Clear buffer pool on shutdown
        if let Err(e) = self.clear_buffer_pool() {
            eprintln!("Warning: Failed to clear buffer pool: {:?}", e);
        }
        // The gpu_allocator handles cleanup automatically when dropped
        // We just need to make sure all buffers are destroyed before this point
    }
}