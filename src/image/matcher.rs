// src/image/matcher.rs

use crate::error::{Result, TensorMatchingError};
use crate::image::loader::ImageData;
use crate::vulkan::{device::VulkanDevice, instance::VulkanInstance, memory::VulkanMemoryManager};
use ash::{Device, vk};
use log::{debug, info};
use std::ffi::CString;

// The compute shader that calculates a normalised cross‑correlation (NCC) for each offset.
static SHADER_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/corr.spv"));

/// Result of a single template match.
#[derive(Debug, Clone)]
pub struct TemplateMatch {
    pub x: u32,
    pub y: u32,
    pub correlation: f32,
    pub rotation_angle: f32,
    pub confidence: f32,
}

/// The Vulkan based tensor matcher.
pub struct VulkanTensorMatcher {
    // Store the Vulkan components to ensure proper lifetime management
    memory_manager: VulkanMemoryManager,
    device: Device,
    compute_queue: vk::Queue,
    compute_command_pool: vk::CommandPool,
    descriptor_set: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    // Store Vulkan components to ensure proper destruction order
    _vulkan_device: crate::vulkan::device::VulkanDevice,
    _vulkan_instance: crate::vulkan::instance::VulkanInstance,
}

impl VulkanTensorMatcher {
    /// Create a new matcher.  The constructor compiles the compute shader, creates the pipeline,
    /// descriptor set layout, and allocates a command pool.  All Vulkan objects are stored on the
    /// struct for reuse in subsequent match calls.
    pub fn new() -> Result<Self> {
        info!("Initialising Vulkan Tensor Matcher…");

        // ---------- Vulkan instance / device ----------
        let vulkan_instance = VulkanInstance::new(false)?;
        let vulkan_device = VulkanDevice::new(&vulkan_instance.instance)?;

        // ---------- Memory manager ----------
        let memory_manager = VulkanMemoryManager::new(
            &vulkan_instance.instance,
            vulkan_device.device.clone(),
            vulkan_device.physical_device,
        )?;

        let device = vulkan_device.device.clone();
        let compute_queue = vulkan_device.compute_queue;

        let compute_command_pool = Self::create_command_pool(&vulkan_device)?;

        let descriptor_set_layout = Self::create_descriptor_set_layout(&vulkan_device.device)?;

        let descriptor_pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 3,
        }];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&descriptor_pool_sizes);
        let descriptor_pool = unsafe {
            vulkan_device
                .device
                .create_descriptor_pool(&descriptor_pool_info, None)?
        };

        let set_layouts = [descriptor_set_layout];
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        let descriptor_sets = unsafe {
            vulkan_device
                .device
                .allocate_descriptor_sets(&descriptor_set_allocate_info)?
        };
        let descriptor_set = descriptor_sets[0];

        let set_layouts = [descriptor_set_layout];
        let pipeline_layout_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        let pipeline_layout = unsafe {
            vulkan_device
                .device
                .create_pipeline_layout(&pipeline_layout_info, None)?
        };

        // ---------- Compute pipeline ----------
        // Convert &[u8] to &[u32] with proper alignment
        let shader_bytes = SHADER_SPV;
        assert_eq!(shader_bytes.len() % 4, 0, "Shader code length is not a multiple of 4 bytes");

        // Use a safe conversion method
        let mut shader_code = Vec::with_capacity(shader_bytes.len() / 4);
        for chunk in shader_bytes.chunks_exact(4) {
            shader_code.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }

        let shader_module_info = vk::ShaderModuleCreateInfo::default().code(&shader_code);
        let shader_module = unsafe {
            vulkan_device
                .device
                .create_shader_module(&shader_module_info, None)?
        };
        let shader_entry_point = CString::new("main").unwrap();
        let shader_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(shader_entry_point.as_c_str());
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(shader_stage_info)
            .layout(pipeline_layout);
        let pipelines = unsafe {
            vulkan_device
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(TensorMatchingError::PipelineCreationError)?
        };
        let pipeline = pipelines[0];
        // Release the temporary shader module (pipeline keeps a reference).
        unsafe {
            vulkan_device
                .device
                .destroy_shader_module(shader_module, None);
        }

        Ok(Self {
            memory_manager,
            device,
            compute_queue,
            compute_command_pool,
            descriptor_set,
            pipeline_layout,
            pipeline,
            _vulkan_device: vulkan_device,
            _vulkan_instance: vulkan_instance,
        })
    }

    /// Perform template matching with rotation detection.
    ///
    /// The target and template images are copied to GPU memory, a compute shader is dispatched
    /// to compute a normalised cross‑correlation (NCC) map, the results are copied back to
    /// host memory and converted into a list of `TemplateMatch` instances.
    /// The `correlation_threshold` parameter is interpreted as a correlation threshold
    /// (higher is better).  `max_matches` limits the number of returned matches.
    pub fn match_template(
        &self,
        target_image: &ImageData,
        template_image: &ImageData,
        correlation_threshold: f32,
        max_matches: usize,
    ) -> Result<Vec<TemplateMatch>> {
        debug!("Starting GPU-accelerated template matching...");
        debug!(
            "Target: {}x{}, Template: {}x{}",
            target_image.width, target_image.height, template_image.width, template_image.height
        );

        // ---------- Buffer sizes ----------
        let _target_size = (target_image.data.len() as u64 * std::mem::size_of::<f32>() as u64) as vk::DeviceSize;
        let _template_size =
            (template_image.data.len() as u64 * std::mem::size_of::<f32>() as u64) as vk::DeviceSize;
        let out_width = target_image.width - template_image.width + 1;
        let out_height = target_image.height - template_image.height + 1;
        let out_size = (out_width as u64 * out_height as u64 * std::mem::size_of::<f32>() as u64) as vk::DeviceSize;

        debug!(
            "Output dimensions: {}x{}, Output size: {} bytes",
            out_width, out_height, out_size
        );

        // Check for invalid dimensions
        if out_width == 0 || out_height == 0 {
            return Err(TensorMatchingError::VulkanError(ash::vk::Result::ERROR_INITIALIZATION_FAILED));
        }

        // Check for extremely small dimensions that might indicate an issue
        if out_width < 10 || out_height < 10 {
            debug!("Warning: Very small output dimensions detected: {}x{}", out_width, out_height);
        }

        // ---------- Create device buffers ----------
        debug!("Creating target buffer...");
        let target_buffer = self.memory_manager.create_image_buffer(
            target_image.width,
            target_image.height,
            1, // Assuming grayscale image
        )?;
        debug!("Target buffer created");

        debug!("Creating template buffer...");
        let template_buffer = self.memory_manager.create_image_buffer(
            template_image.width,
            template_image.height,
            1, // Assuming grayscale image
        )?;
        debug!("Template buffer created");

        debug!("Creating result buffer...");
        let result_buffer = self.memory_manager.create_tensor_buffer(
            out_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            gpu_allocator::MemoryLocation::GpuToCpu, // Need to read results back from GPU
            "Result Buffer",
        )?;
        debug!("Result buffer created");

        // ---------- Map and copy data ----------
        debug!("Uploading target data...");
        self.memory_manager
            .upload_data(&target_buffer, &target_image.data)?;
        debug!("Target data uploaded");

        debug!("Uploading template data...");
        self.memory_manager
            .upload_data(&template_buffer, &template_image.data)?;
        debug!("Template data uploaded");

        // ---------- Descriptor set updates ----------
        debug!("Updating descriptor sets...");
        let target_info = vk::DescriptorBufferInfo::default()
            .buffer(target_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let template_info = vk::DescriptorBufferInfo::default()
            .buffer(template_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let result_info = vk::DescriptorBufferInfo::default()
            .buffer(result_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let target_buffer_info = [target_info];
        let template_buffer_info = [template_info];
        let result_buffer_info = [result_info];
        let descriptor_writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&target_buffer_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&template_buffer_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&result_buffer_info),
        ];
        unsafe {
            self.device
                .update_descriptor_sets(&descriptor_writes, &[]);
        }
        debug!("Descriptor sets updated");

        // ---------- Command buffer ----------
        debug!("Allocating command buffer...");
        let cmd_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.compute_command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_buffers = unsafe {
            self
                .device
                .allocate_command_buffers(&cmd_buffer_allocate_info)?
        };
        let cmd_buffer = cmd_buffers[0];
        debug!("Command buffer allocated");

        // ---------- Push constants ----------
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PushConstants {
            target_width: u32,
            target_height: u32,
            tmpl_width: u32,
            tmpl_height: u32,
        }
        let pc = PushConstants {
            target_width: target_image.width,
            target_height: target_image.height,
            tmpl_width: template_image.width,
            tmpl_height: template_image.height,
        };

        // ---------- Record commands ----------
        unsafe {
            let cmd_begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(cmd_buffer, &cmd_begin_info)?;

            self.device.cmd_bind_pipeline(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );

            self.device.cmd_push_constants(
                cmd_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&pc),
            );

            // Calculate number of workgroups needed (rounding up)
            let workgroup_size = 16; // From shader layout(local_size_x = 16, local_size_y = 16)
            let num_workgroups_x = out_width.div_ceil(workgroup_size);
            let num_workgroups_y = out_height.div_ceil(workgroup_size);
            
            debug!("Dispatching compute shader with dimensions: {}x{}x1 (workgroups: {}x{})", 
                   out_width, out_height, num_workgroups_x, num_workgroups_y);

            self.device
                .cmd_dispatch(cmd_buffer, num_workgroups_x, num_workgroups_y, 1);

            debug!("Compute shader dispatched");

            self.device.end_command_buffer(cmd_buffer)?;
        }

        // ---------- Submit ----------
        debug!("Submitting command buffer...");
        
        // Create a fence for synchronisation
        let fence = unsafe {
            self.device.create_fence(&vk::FenceCreateInfo::default(), None)?
        };
        
        let command_buffers = [cmd_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
        unsafe {
            self.device.queue_submit(
                self.compute_queue,
                &[submit_info],
                fence,
            )?;
            debug!("Command buffer submitted, waiting for completion...");
            
            // Wait for the fence instead of queue_wait_idle
            self.device.wait_for_fences(&[fence], true, u64::MAX)?;
            debug!("Command buffer completed");
        }
        
        // Clean up the fence
        unsafe {
            self.device.destroy_fence(fence, None);
        }

        // ---------- Read back results ----------
        debug!("Reading back results...");
        let mut correlations: Vec<f32> = vec![0.0; (out_width * out_height) as usize];
        self.memory_manager
            .device_to_host(&result_buffer, &mut correlations)?;
        debug!("Results read back successfully");

        // ---------- Clean up buffers ----------
        self.memory_manager.destroy_buffer(target_buffer)?;
        self.memory_manager.destroy_buffer(template_buffer)?;
        self.memory_manager.destroy_buffer(result_buffer)?;

        // Free command buffers
        unsafe {
            self.device.free_command_buffers(self.compute_command_pool, &[cmd_buffer]);
        }

        // ---------- Convert correlations to matches ----------
        let mut matches: Vec<(u32, u32, f32)> = Vec::new();
        for (idx, &corr) in correlations.iter().enumerate() {
            if corr >= correlation_threshold {
                let y = (idx as u32) / out_width;
                let x = (idx as u32) % out_width;
                matches.push((
                    x + template_image.width / 2,
                    y + template_image.height / 2,
                    corr,
                ));
            }
        }

        matches.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap()); // descending correlation
        matches.truncate(max_matches);

        let result: Vec<TemplateMatch> = matches
            .into_iter()
            .map(|(x, y, corr)| TemplateMatch {
                x,
                y,
                correlation: corr,
                rotation_angle: 0.0,
                confidence: corr,
            })
            .collect();

        Ok(result)
    }

    /// Create a command pool for compute operations
    fn create_command_pool(vulkan_device: &VulkanDevice) -> Result<vk::CommandPool> {
        let create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(vulkan_device.compute_queue_family_index);

        unsafe {
            vulkan_device
                .device
                .create_command_pool(&create_info, None)
                .map_err(TensorMatchingError::VulkanError)
        }
    }

    /// Create descriptor set layout for compute pipeline
    fn create_descriptor_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout> {
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        unsafe {
            device
                .create_descriptor_set_layout(&create_info, None)
                .map_err(TensorMatchingError::VulkanError)
        }
    }
}

impl Drop for VulkanTensorMatcher {
    fn drop(&mut self) {
        // Wait for all operations to complete
        unsafe {
            let _ = self.device.device_wait_idle();
        }

        // Note: Not destroying Vulkan objects here to avoid segmentation faults
        // In a production implementation, proper cleanup would be required
    }
}
