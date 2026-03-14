use crate::error::{Result, TensorMatchingError};
use ash::{Device, Instance, vk};
use log::debug;
use std::sync::Arc;

pub struct VulkanDevice {
    pub physical_device: vk::PhysicalDevice,
    pub device: Device,
    pub compute_queue: vk::Queue,
    pub compute_queue_family_index: u32,
}

impl VulkanDevice {
    pub fn new(instance: &Instance) -> Result<Arc<Self>> {
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };
        let physical_device = physical_devices
            .into_iter()
            .find(|&pd| {
                let props = unsafe { instance.get_physical_device_properties(pd) };
                props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                    || props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
            })
            .ok_or_else(|| TensorMatchingError::Other("No suitable GPU found".to_string()))?;

        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let compute_queue_family_index = queue_families
            .iter()
            .enumerate()
            .find(|(_i, props)| props.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(i, _props)| i as u32)
            .ok_or_else(|| {
                TensorMatchingError::Other("No compute queue family found".to_string())
            })?;

        let queue_priorities = [1.0];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(compute_queue_family_index)
            .queue_priorities(&queue_priorities);

        let device_create_info =
            vk::DeviceCreateInfo::default().queue_create_infos(std::slice::from_ref(&queue_info));

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None)? };
        let compute_queue = unsafe { device.get_device_queue(compute_queue_family_index, 0) };

        Ok(Arc::new(Self {
            physical_device,
            device,
            compute_queue,
            compute_queue_family_index,
        }))
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            debug!("Destroying Vulkan device");
            self.device.destroy_device(None);
        }
    }
}
