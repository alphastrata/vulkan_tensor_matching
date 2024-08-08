use ash::{vk, Device, Instance};
use log::{debug, info};
use crate::error::{Result, TensorMatchingError};
use std::ffi::CStr;

#[derive(Clone)]
pub struct VulkanDevice {
    pub physical_device: vk::PhysicalDevice,
    pub device: Device,
    pub compute_queue: vk::Queue,
    pub compute_queue_family_index: u32,
    pub device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub device_properties: vk::PhysicalDeviceProperties,
}

impl std::fmt::Debug for VulkanDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanDevice")
            .field("physical_device", &self.physical_device)
            .field("device", &"Device")
            .field("compute_queue", &self.compute_queue)
            .field("compute_queue_family_index", &self.compute_queue_family_index)
            .field("device_memory_properties", &self.device_memory_properties)
            .field("device_properties", &self.device_properties)
            .finish()
    }
}

impl VulkanDevice {
    pub fn new(instance: &Instance) -> Result<Self> {
        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(TensorMatchingError::VulkanError)?;

        let (physical_device, compute_queue_family_index) = physical_devices
            .iter()
            .find_map(|&device| {
                Self::find_compute_queue_family(instance, device).map(|index| (device, index))
            })
            .ok_or_else(|| TensorMatchingError::NoGpuFound)?;

        let device_name = Self::get_device_name(instance, physical_device);
        info!("Selected GPU: {}", device_name);

        let device_properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let device_memory_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };

        // Print compute capabilities
        let limits = &device_properties.limits;
        debug!("Compute Capabilities:");
        debug!("   Max compute workgroup size: {}x{}x{}",
               limits.max_compute_work_group_size[0],
               limits.max_compute_work_group_size[1],
               limits.max_compute_work_group_size[2]);
        debug!("   Max compute workgroup invocations: {}", limits.max_compute_work_group_invocations);
        debug!("   Max compute shared memory: {} KB", limits.max_compute_shared_memory_size / 1024);

        let queue_priorities = [1.0];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(compute_queue_family_index)
            .queue_priorities(&queue_priorities);

        let device_extension_names = vec![
            // Add tensor operation extensions if available
        ];

        let features = unsafe { instance.get_physical_device_features(physical_device) };
        let _required_features = vk::PhysicalDeviceFeatures {
            shader_float64: if features.shader_float64 == vk::TRUE { vk::TRUE } else { vk::FALSE },
            shader_int64: if features.shader_int64 == vk::TRUE { vk::TRUE } else { vk::FALSE },
            ..Default::default()
        };

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_create_info))
            .enabled_extension_names(&device_extension_names)
            .enabled_features(&features);

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }
            .map_err(TensorMatchingError::VulkanError)?;
        let compute_queue = unsafe { device.get_device_queue(compute_queue_family_index, 0) };

        Ok(Self {
            physical_device,
            device,
            compute_queue,
            compute_queue_family_index,
            device_memory_properties,
            device_properties,
        })
    }

    fn find_compute_queue_family(instance: &Instance, device: vk::PhysicalDevice) -> Option<u32> {
        let queue_family_properties = unsafe {
            instance.get_physical_device_queue_family_properties(device)
        };

        queue_family_properties
            .iter()
            .enumerate()
            .find(|(_, properties)| {
                properties.queue_flags.contains(vk::QueueFlags::COMPUTE)
            })
            .map(|(index, _)| index as u32)
    }

    fn get_device_name(instance: &Instance, device: vk::PhysicalDevice) -> String {
        let properties = unsafe { instance.get_physical_device_properties(device) };
        let device_name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) };
        device_name.to_string_lossy().into_owned()
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }
}