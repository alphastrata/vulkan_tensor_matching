




use crate::error::Result;
use crate::image::loader::ImageData;
use crate::vulkan::{device::VulkanDevice, instance::VulkanInstance, memory::VulkanMemoryManager};
use ash::{Device, util, vk};
use log::{debug, info};
use std::ffi::CString;
use std::sync::Arc;


static MULTI_ANGLE_NCC_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/tensorial_correlation.spv"));

#[derive(Debug, Clone)]
pub struct TensorTemplateMatch {
    pub x: u32,
    pub y: u32,
    pub correlation: f32,
    pub rotation_angle: f32,
    pub confidence: f32,
}

pub struct VulkanTensorMatcher {
    memory_manager: VulkanMemoryManager,
    compute_queue: vk::Queue,
    compute_command_pool: vk::CommandPool,

    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,

    _vulkan_device: Arc<VulkanDevice>,
    _vulkan_instance: Arc<VulkanInstance>,
}

impl VulkanTensorMatcher {
    pub fn new() -> Result<Self> {
        info!("Initialising Vulkan Multi-Angle Template Matcher...");

        let vulkan_instance = VulkanInstance::new(false)?;
        let vulkan_device = VulkanDevice::new(&vulkan_instance.instance)?;

        let memory_manager = VulkanMemoryManager::new(
            &vulkan_instance.instance,
            vulkan_device.device.clone(),
            vulkan_device.physical_device,
        )?;

        let compute_queue = vulkan_device.compute_queue;

        let pool_create_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(vulkan_device.compute_queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let compute_command_pool = unsafe {
            vulkan_device
                .device
                .create_command_pool(&pool_create_info, None)?
        };

        
        let descriptor_set_layout = Self::create_descriptor_set_layout(
            &vulkan_device.device,
            &[
                vk::DescriptorType::STORAGE_BUFFER,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::DescriptorType::UNIFORM_BUFFER,
            ],
        )?;

        let descriptor_pool = Self::create_descriptor_pool(
            &vulkan_device.device,
            &[
                (vk::DescriptorType::STORAGE_BUFFER, 3),
                (vk::DescriptorType::UNIFORM_BUFFER, 1),
            ],
        )?;

        let descriptor_set = Self::allocate_descriptor_set(
            &vulkan_device.device,
            descriptor_pool,
            &[descriptor_set_layout],
        )?;

        let pipeline_layout =
            Self::create_pipeline_layout(&vulkan_device.device, &[descriptor_set_layout])?;

        let pipeline = Self::create_compute_pipeline(
            &vulkan_device.device,
            MULTI_ANGLE_NCC_SPV,
            pipeline_layout,
        )?;

        Ok(Self {
            memory_manager,
            compute_queue,
            compute_command_pool,
            descriptor_pool,
            descriptor_set,
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            _vulkan_device: vulkan_device,
            _vulkan_instance: vulkan_instance,
        })
    }

    pub fn match_template(
        &self,
        target_image: &ImageData,
        template_image: &ImageData,
        correlation_threshold: f32,
        max_matches: usize,
    ) -> Result<Vec<TensorTemplateMatch>> {
        debug!("Starting multi-angle template matching...");
        debug!(
            "Target: {}x{}, Template: {}x{}",
            target_image.width, target_image.height, template_image.width, template_image.height
        );

        let out_w = target_image.width - template_image.width + 1;
        let out_h = target_image.height - template_image.height + 1;

        
        let target_buffer =
            self.memory_manager
                .create_image_buffer(target_image.width, target_image.height, 1)?;
        self.memory_manager
            .upload_data(&target_buffer, &target_image.data)?;

        let template_buffer = self.memory_manager.create_image_buffer(
            template_image.width,
            template_image.height,
            1,
        )?;
        self.memory_manager
            .upload_data(&template_buffer, &template_image.data)?;

        
        let result_buffer = self.memory_manager.create_tensor_buffer(
            (out_w * out_h * 16) as u64, 
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::GpuToCpu,
            "Result Buffer",
        )?;

        
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct Params {
            target_width: u32,
            target_height: u32,
            template_width: u32,
            template_height: u32,
            correlation_threshold: f32,
            max_results: u32,
            num_angles: u32,
            padding: u32,
        }

        let params_buffer = self.memory_manager.create_tensor_buffer(
            32,
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::CpuToGpu,
            "Params",
        )?;
        self.memory_manager.upload_data(
            &params_buffer,
            &[Params {
                target_width: target_image.width,
                target_height: target_image.height,
                template_width: template_image.width,
                template_height: template_image.height,
                correlation_threshold,
                max_results: max_matches as u32,
                num_angles: 36, 
                padding: 0,
            }],
        )?;

        
        let target_info = [vk::DescriptorBufferInfo::default()
            .buffer(target_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let template_info = [vk::DescriptorBufferInfo::default()
            .buffer(template_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let result_info = [vk::DescriptorBufferInfo::default()
            .buffer(result_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let params_info = [vk::DescriptorBufferInfo::default()
            .buffer(params_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&target_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&template_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&result_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&params_info),
        ];
        unsafe {
            self._vulkan_device
                .device
                .update_descriptor_sets(&writes, &[]);
        }

        
        self.dispatch(
            self.pipeline,
            self.pipeline_layout,
            self.descriptor_set,
            out_w.div_ceil(16),
            out_h.div_ceil(16),
        )?;

        
        let mut result_data = vec![0.0f32; (out_w * out_h * 4) as usize];
        self.memory_manager
            .device_to_host(&result_buffer, &mut result_data)?;

        
        let matches = self.find_peaks(&result_data, out_w, out_h, template_image, max_matches)?;

        
        self.memory_manager.destroy_buffer(target_buffer)?;
        self.memory_manager.destroy_buffer(template_buffer)?;
        self.memory_manager.destroy_buffer(result_buffer)?;
        self.memory_manager.destroy_buffer(params_buffer)?;

        debug!("Found {} matches", matches.len());
        Ok(matches)
    }

    fn find_peaks(
        &self,
        data: &[f32],
        out_w: u32,
        out_h: u32,
        template: &ImageData,
        max_matches: usize,
    ) -> Result<Vec<TensorTemplateMatch>> {
        let exclusion_radius: i32 = template.width.max(template.height) as i32 / 4;
        let boundary: usize = 10;
        let out_w = out_w as usize;
        let out_h = out_h as usize;

        let mut peaks: Vec<(f32, usize, usize)> = Vec::new();

        for y in boundary..out_h.saturating_sub(boundary) {
            for x in boundary..out_w.saturating_sub(boundary) {
                let idx = y * out_w + x;
                let corr = data[idx * 4];
                if corr < 0.0 {
                    continue;
                }

                let mut is_max = true;
                'outer: for dy in -exclusion_radius..=exclusion_radius {
                    for dx in -exclusion_radius..=exclusion_radius {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx < 0 || nx >= out_w as i32 || ny < 0 || ny >= out_h as i32 {
                            continue;
                        }
                        let nidx = ny as usize * out_w + nx as usize;
                        if data[nidx * 4] > corr {
                            is_max = false;
                            break 'outer;
                        }
                    }
                }

                if is_max {
                    peaks.push((corr, x, y));
                }
            }
        }

        peaks.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        peaks.truncate(max_matches);

        let matches: Vec<TensorTemplateMatch> = peaks
            .iter()
            .map(|&(corr, x, y)| {
                let idx = y * out_w + x;
                let angle = data[idx * 4 + 1];

                TensorTemplateMatch {
                    x: x as u32 + template.width / 2,
                    y: y as u32 + template.height / 2,
                    correlation: corr,
                    rotation_angle: angle,
                    confidence: corr,
                }
            })
            .collect();

        Ok(matches)
    }

    fn dispatch(
        &self,
        pipeline: vk::Pipeline,
        layout: vk::PipelineLayout,
        set: vk::DescriptorSet,
        x: u32,
        y: u32,
    ) -> Result<()> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.compute_command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe {
            self._vulkan_device
                .device
                .allocate_command_buffers(&allocate_info)?[0]
        };
        unsafe {
            self._vulkan_device.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            self._vulkan_device.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
            );
            self._vulkan_device.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &[set],
                &[],
            );
            self._vulkan_device.device.cmd_dispatch(cmd, x, y, 1);
            self._vulkan_device.device.end_command_buffer(cmd)?;
            let bufs = [cmd];
            self._vulkan_device.device.queue_submit(
                self.compute_queue,
                &[vk::SubmitInfo::default().command_buffers(&bufs)],
                vk::Fence::null(),
            )?;
            self._vulkan_device
                .device
                .queue_wait_idle(self.compute_queue)?;
            self._vulkan_device
                .device
                .free_command_buffers(self.compute_command_pool, &[cmd]);
        }
        Ok(())
    }

    fn create_descriptor_set_layout(
        device: &Device,
        types: &[vk::DescriptorType],
    ) -> Result<vk::DescriptorSetLayout> {
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = types
            .iter()
            .enumerate()
            .map(|(i, &t)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i as u32)
                    .descriptor_type(t)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        unsafe {
            Ok(device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?)
        }
    }

    fn create_descriptor_pool(
        device: &Device,
        sizes: &[(vk::DescriptorType, u32)],
    ) -> Result<vk::DescriptorPool> {
        let pool_sizes: Vec<vk::DescriptorPoolSize> = sizes
            .iter()
            .map(|&(t, c)| vk::DescriptorPoolSize::default().ty(t).descriptor_count(c))
            .collect();
        unsafe {
            Ok(device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&pool_sizes),
                None,
            )?)
        }
    }

    fn allocate_descriptor_set(
        device: &Device,
        pool: vk::DescriptorPool,
        layouts: &[vk::DescriptorSetLayout],
    ) -> Result<vk::DescriptorSet> {
        unsafe {
            Ok(device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(layouts),
            )?[0])
        }
    }

    fn create_pipeline_layout(
        device: &Device,
        layouts: &[vk::DescriptorSetLayout],
    ) -> Result<vk::PipelineLayout> {
        unsafe {
            Ok(device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(layouts),
                None,
            )?)
        }
    }

    fn create_compute_pipeline(
        device: &Device,
        code: &[u8],
        layout: vk::PipelineLayout,
    ) -> Result<vk::Pipeline> {
        let shader = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default()
                    .code(&util::read_spv(&mut std::io::Cursor::new(code))?),
                None,
            )?
        };
        let entry = CString::new("main").unwrap();
        let pipe = unsafe {
            device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::COMPUTE)
                                .module(shader)
                                .name(&entry),
                        )
                        .layout(layout)],
                    None,
                )
                .map_err(|e| e.1)?[0]
        };
        unsafe {
            device.destroy_shader_module(shader, None);
        }
        Ok(pipe)
    }
}

impl Drop for VulkanTensorMatcher {
    fn drop(&mut self) {
        unsafe {
            let d = &self._vulkan_device.device;
            d.destroy_pipeline(self.pipeline, None);
            d.destroy_pipeline_layout(self.pipeline_layout, None);
            d.destroy_descriptor_pool(self.descriptor_pool, None);
            d.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            d.destroy_command_pool(self.compute_command_pool, None);
        }
    }
}
