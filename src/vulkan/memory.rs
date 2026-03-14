use crate::error::{Result, TensorMatchingError};
use ash::{Device, vk};
use gpu_allocator::vulkan::{
    Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc,
};
use gpu_allocator::{AllocationSizes, MemoryLocation};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};


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
        })
        .map_err(TensorMatchingError::AllocationError)?;

        Ok(Self {
            device,
            allocator: Arc::new(Mutex::new(allocator)),
            buffer_pool: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    
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

        
        {
            let mut pool = self.buffer_pool.lock().unwrap();
            if let Some(buffers) = pool.get_mut(&key)
                && let Some(buffer) = buffers.pop()
            {
                return Ok(buffer);
            }
        }

        
        self.create_tensor_buffer(size, usage, memory_location, name)
    }

    
    pub fn return_buffer_to_pool(&self, buffer: VulkanBuffer) -> Result<()> {
        let key = BufferPoolKey {
            size: buffer.size,
            usage: buffer.usage,
            memory_location: match buffer.allocation.as_ref() {
                Some(_allocation) => {
                    
                    MemoryLocation::GpuOnly 
                }
                None => MemoryLocation::Unknown,
            },
        };

        let mut pool = self.buffer_pool.lock().unwrap();
        pool.entry(key).or_default().push(buffer);

        Ok(())
    }

    
    pub fn clear_buffer_pool(&self) -> Result<()> {
        let mut pool = self.buffer_pool.lock().unwrap();
        for (_, buffers) in pool.drain() {
            for buffer in buffers {
                self.destroy_buffer(buffer)?;
            }
        }
        Ok(())
    }

    
    pub fn get_pool_statistics(&self) -> (usize, usize) {
        let pool = self.buffer_pool.lock().unwrap();
        let total_buffers: usize = pool.values().map(|v| v.len()).sum();
        let pool_types = pool.len();
        (total_buffers, pool_types)
    }

    
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

        let allocation = self
            .allocator
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name,
                requirements,
                location: memory_location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;

        unsafe {
            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
        }

        Ok(VulkanBuffer {
            buffer,
            allocation: Some(allocation),
            size,
            usage,
        })
    }

    
    pub fn create_tensor_field_buffer(&self, width: u32, height: u32) -> Result<VulkanBuffer> {
        
        let tensor_size = std::mem::size_of::<f32>() * 8;
        let total_size = (width * height) as u64 * tensor_size as u64;

        self.create_tensor_buffer(
            total_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            "Tensor Field Buffer",
        )
    }

    
    pub fn create_image_buffer(
        &self,
        width: u32,
        height: u32,
        channels: u32,
    ) -> Result<VulkanBuffer> {
        let size = (width * height * channels) as u64 * std::mem::size_of::<f32>() as u64;

        self.create_tensor_buffer(
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::CpuToGpu,
            "Image Buffer",
        )
    }

    
    pub fn create_staging_buffer(&self, size: u64) -> Result<VulkanBuffer> {
        self.create_tensor_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::CpuToGpu,
            "Staging Buffer",
        )
    }

    
    pub fn upload_data<T: Copy>(&self, buffer: &VulkanBuffer, data: &[T]) -> Result<()> {
        if let Some(allocation) = &buffer.allocation {
            if allocation.mapped_ptr().is_some() {
                unsafe {
                    let mapped_ptr = allocation.mapped_ptr().unwrap().as_ptr() as *mut T;
                    std::ptr::copy_nonoverlapping(data.as_ptr(), mapped_ptr, data.len());
                }
            } else {
                return Err(TensorMatchingError::VulkanError(
                    ash::vk::Result::ERROR_MEMORY_MAP_FAILED,
                ));
            }
        }
        Ok(())
    }

    
    pub fn device_to_host<T: Copy>(&self, buffer: &VulkanBuffer, data: &mut [T]) -> Result<()> {
        if let Some(allocation) = &buffer.allocation {
            if allocation.mapped_ptr().is_some() {
                unsafe {
                    let mapped_ptr = allocation.mapped_ptr().unwrap().as_ptr() as *const T;
                    std::ptr::copy_nonoverlapping(mapped_ptr, data.as_mut_ptr(), data.len());
                }
            } else {
                return Err(TensorMatchingError::VulkanError(
                    ash::vk::Result::ERROR_MEMORY_MAP_FAILED,
                ));
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

    
    pub fn destroy_or_pool_buffer(&self, buffer: VulkanBuffer) -> Result<()> {
        
        
        self.destroy_buffer(buffer)
    }
}

impl Drop for VulkanMemoryManager {
    fn drop(&mut self) {
        
        if let Err(e) = self.clear_buffer_pool() {
            eprintln!("Warning: Failed to clear buffer pool: {:?}", e);
        }
        
        
    }
}
