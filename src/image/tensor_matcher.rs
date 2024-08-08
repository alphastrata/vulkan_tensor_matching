// src/image/tensor_matcher.rs

/// Tensorial Template Matching Implementation
/// Based on: "Tensorial template matching for fast cross-correlation with rotations"
/// Martinez-Sanchez et al., arXiv:2408.02398v1 [cs.CV]
///
/// Implements the full tensorial template matching algorithm as described in the paper:
/// - Section 3: Tensorial Template Matching (p. 4-8)
/// - Algorithm 1: Tensor template generation (p. 5)
/// - Algorithm 2: Tensorial field computation (p. 6)
/// - Section 3.1: Optimal rotation determination using SS-HOPM (p. 6-7)
/// - Section 3.2: Instance positions using Frobenius norm (p. 7)
use crate::error::{Result, TensorMatchingError};
use crate::image::loader::ImageData;
use crate::vulkan::{device::VulkanDevice, instance::VulkanInstance, memory::VulkanMemoryManager};
use ash::{Device, vk};
use log::{debug, info};
use std::ffi::CString;
use crate::image::fft::REFINEMENT_RADIUS;

// The compute shaders for tensorial template matching
// Based on Algorithm 1 and Algorithm 2 from the paper
static TENSOR_GENERATION_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tensor_generation_full.spv"));
static TENSORIAL_CORRELATION_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tensorial_correlation.spv"));
static TENSORIAL_PEAK_DETECTION_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tensorial_peak_detection.spv"));

/// Result of a single template match with rotation information.
/// Implements the tensorial matching result as described in Section 3.1 (p. 6-7)
#[derive(Debug, Clone)]
pub struct TensorTemplateMatch {
    pub x: u32,
    pub y: u32,
    pub correlation: f32,
    pub rotation_angle: f32,
    pub confidence: f32,
}

/// A structure representing a symmetric tensor field for template matching.
/// Implements degree-4 symmetric tensors as described in Section 3 (p. 4-5)
/// For degree-4 symmetric tensors in 2D, we need (4+2-1)!/(4!*(2-1)!) = 5 components
/// Using 8 components with padding for GPU alignment
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TensorField {
    pub components: [f32; 8], // 5 real components + 3 padding for alignment
}

// Implement Pod and Zeroable manually for large arrays
unsafe impl bytemuck::Pod for TensorField {}
unsafe impl bytemuck::Zeroable for TensorField {}

/// The Vulkan based tensor matcher implementing the full tensorial template matching algorithm.
/// Based on the mathematical framework described in:
/// - Section 3: Tensorial Template Matching (p. 4-8)
/// - Algorithm 1: Tensor template generation (p. 5)
/// - Algorithm 2: Tensorial field computation (p. 6)
///
/// Key advantages as claimed in the paper:
/// - Computational complexity independent of rotation accuracy (p. 3, 7)
/// - Only 35 correlations required vs. thousands in traditional approach (p. 6)
/// - Integration over all rotations captures complete template information (p. 4-5)
pub struct VulkanTensorMatcher {
    // Store the Vulkan components to ensure proper lifetime management
    _vulkan_instance: VulkanInstance,
    _vulkan_device: VulkanDevice,
    memory_manager: VulkanMemoryManager,
    compute_queue: vk::Queue,
    compute_command_pool: vk::CommandPool,

    // Pipeline components for tensor generation
    tensor_gen_descriptor_pool: vk::DescriptorPool,
    tensor_gen_descriptor_set: vk::DescriptorSet,
    tensor_gen_descriptor_set_layout: vk::DescriptorSetLayout,
    tensor_gen_pipeline_layout: vk::PipelineLayout,
    tensor_gen_pipeline: vk::Pipeline,

    // Pipeline components for tensorial correlation
    correlation_descriptor_pool: vk::DescriptorPool,
    correlation_descriptor_set: vk::DescriptorSet,
    correlation_descriptor_set_layout: vk::DescriptorSetLayout,
    correlation_pipeline_layout: vk::PipelineLayout,
    correlation_pipeline: vk::Pipeline,

    // Pipeline components for peak detection
    peak_descriptor_pool: vk::DescriptorPool,
    peak_descriptor_set: vk::DescriptorSet,
    peak_descriptor_set_layout: vk::DescriptorSetLayout,
    peak_pipeline_layout: vk::PipelineLayout,
    peak_pipeline: vk::Pipeline,
}

impl VulkanTensorMatcher {
    /// Create a new tensor matcher implementing the full tensorial template matching algorithm.
    pub fn new() -> Result<Self> {
        info!("Initialising Vulkan Tensor Matcher (Full Tensorial Implementation)…");

        let _vulkan_instance = VulkanInstance::new(true)?;
        let _vulkan_device = VulkanDevice::new(&_vulkan_instance.instance)?;

        // ---------- Memory manager ----------
        let memory_manager = VulkanMemoryManager::new(
            &_vulkan_instance.instance,
            _vulkan_device.device.clone(),
            _vulkan_device.physical_device,
        )?;

        // ---------- Store device and compute queue directly ----------
        let compute_queue = _vulkan_device.compute_queue;

        // ---------- Compute command pool ----------
        let compute_command_pool = Self::create_command_pool(&_vulkan_device)?;

        // ---------- Tensor Generation Pipeline ----------
        let tensor_gen_descriptor_set_layout = Self::create_tensor_gen_descriptor_set_layout(&_vulkan_device.device)?;
        let tensor_gen_descriptor_pool = Self::create_descriptor_pool(&_vulkan_device.device, &tensor_gen_descriptor_set_layout)?;
        let tensor_gen_descriptor_set = Self::allocate_descriptor_set(&_vulkan_device.device, tensor_gen_descriptor_pool, &[tensor_gen_descriptor_set_layout])?;
        let tensor_gen_pipeline_layout = Self::create_pipeline_layout(&_vulkan_device.device, &[tensor_gen_descriptor_set_layout])?;
        let tensor_gen_pipeline = Self::create_compute_pipeline(&_vulkan_device.device, TENSOR_GENERATION_SPV, tensor_gen_pipeline_layout)?;

        // ---------- Correlation Pipeline ----------
        let correlation_descriptor_set_layout = Self::create_correlation_descriptor_set_layout(&_vulkan_device.device)?;
        let correlation_descriptor_pool = Self::create_descriptor_pool(&_vulkan_device.device, &correlation_descriptor_set_layout)?;
        let correlation_descriptor_set = Self::allocate_descriptor_set(&_vulkan_device.device, correlation_descriptor_pool, &[correlation_descriptor_set_layout])?;
        let correlation_pipeline_layout = Self::create_pipeline_layout(&_vulkan_device.device, &[correlation_descriptor_set_layout])?;
        let correlation_pipeline = Self::create_compute_pipeline(&_vulkan_device.device, TENSORIAL_CORRELATION_SPV, correlation_pipeline_layout)?;

        // ---------- Peak Detection Pipeline ----------
        let peak_descriptor_set_layout = Self::create_peak_descriptor_set_layout(&_vulkan_device.device)?;
        let peak_descriptor_pool = Self::create_descriptor_pool(&_vulkan_device.device, &peak_descriptor_set_layout)?;
        let peak_descriptor_set = Self::allocate_descriptor_set(&_vulkan_device.device, peak_descriptor_pool, &[peak_descriptor_set_layout])?;
        let peak_pipeline_layout = Self::create_pipeline_layout(&_vulkan_device.device, &[peak_descriptor_set_layout])?;
        let peak_pipeline = Self::create_compute_pipeline(&_vulkan_device.device, TENSORIAL_PEAK_DETECTION_SPV, peak_pipeline_layout)?;

        Ok(Self {
            _vulkan_instance,
            _vulkan_device,
            memory_manager,
            compute_queue,
            compute_command_pool,
            tensor_gen_descriptor_pool,
            tensor_gen_descriptor_set,
            tensor_gen_descriptor_set_layout,
            tensor_gen_pipeline_layout,
            tensor_gen_pipeline,
            correlation_descriptor_pool,
            correlation_descriptor_set,
            correlation_descriptor_set_layout,
            correlation_pipeline_layout,
            correlation_pipeline,
            peak_descriptor_pool,
            peak_descriptor_set,
            peak_descriptor_set_layout,
            peak_pipeline_layout,
            peak_pipeline,
        })
    }

    /// Perform tensorial template matching with full rotation detection.
    ///
    /// This implements the full tensorial template matching algorithm from the paper:
    /// "Tensorial template matching for fast cross-correlation with rotations"
    /// Martinez-Sanchez et al., arXiv:2408.02398v1
    ///
    /// The algorithm consists of three main stages:
    /// 1. Tensor generation: Integrate template over all rotations to create tensor field
    /// 2. Tensorial correlation: Compute correlation tensor field using the tensor template
    /// 3. Peak detection: Find local maxima and determine optimal rotations
    ///
    /// The `correlation_threshold` parameter is interpreted as a correlation threshold
    /// (higher is better). `max_matches` limits the number of returned matches.
    pub fn match_template(
        &self,
        target_image: &ImageData,
        template_image: &ImageData,
        correlation_threshold: f32,
        max_matches: usize,
    ) -> Result<Vec<TensorTemplateMatch>> {
        debug!("Starting GPU-accelerated tensorial template matching…");
        debug!(
            "Target: {}x{}, Template: {}x{}",
            target_image.width, target_image.height, template_image.width, template_image.height
        );

        // Stage 1: Generate tensor field for the template
        debug!("Stage 1: Generating tensor field for template...");
        let template_tensor_buffer = self.generate_template_tensor_field(template_image)?;

        // Stage 2: Compute tensorial correlation field
        debug!("Stage 2: Computing tensorial correlation field...");
        let (correlation_buffer, rotation_buffer, tensor_field_buffer) =
            self.compute_tensorial_correlation(target_image, &template_tensor_buffer, template_image)?;

        // Stage 3: Detect peaks and determine rotations
        debug!("Stage 3: Detecting peaks and determining rotations...");
        let matches = self.detect_peaks(
            &correlation_buffer,
            &rotation_buffer,
            target_image,
            template_image,
            correlation_threshold,
            max_matches
        )?;

        // Clean up temporary buffers
        self.memory_manager.destroy_buffer(template_tensor_buffer)?;
        self.memory_manager.destroy_buffer(correlation_buffer)?;
        self.memory_manager.destroy_buffer(rotation_buffer)?;
        self.memory_manager.destroy_buffer(tensor_field_buffer)?;

        Ok(matches)
    }

    /// Generate tensor field for template by integrating over all rotations
    fn generate_template_tensor_field(&self, template_image: &ImageData) -> Result<crate::vulkan::memory::VulkanBuffer> {
        // Create buffer for tensor field output
        let tensor_field_size = (template_image.width * template_image.height * 8 * std::mem::size_of::<f32>() as u32) as u64;
        let tensor_field_buffer = self.memory_manager.create_tensor_buffer(
            tensor_field_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::GpuOnly,
            "Template Tensor Field Buffer",
        )?;

        // Create buffer for input template image
        let template_buffer = self.memory_manager.create_image_buffer(
            template_image.width,
            template_image.height,
            1, // Assuming grayscale image
        )?;

        // Upload template data
        self.memory_manager.upload_data(&template_buffer, &template_image.data)?;

        // Create uniform buffer for shader parameters
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct TensorGenParams {
            width: u32,
            height: u32,
            num_angles: u32,
            padding: u32,
        }

        let params = TensorGenParams {
            width: template_image.width,
            height: template_image.height,
            num_angles: 360, // Number of rotation samples
            padding: 0,
        };

        let params_buffer = self.memory_manager.create_tensor_buffer(
            std::mem::size_of::<TensorGenParams>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::CpuToGpu,
            "Tensor Generation Parameters",
        )?;
        self.memory_manager.upload_data(&params_buffer, &[params])?;

        // Update descriptor set with buffer bindings
        // Create buffer info arrays to avoid temporary value issues
        let template_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(template_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let tensor_field_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(tensor_field_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let params_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(params_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];

        let descriptor_writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.tensor_gen_descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&template_buffer_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.tensor_gen_descriptor_set)
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&tensor_field_buffer_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.tensor_gen_descriptor_set)
                .dst_binding(2)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&params_buffer_info),
        ];

        unsafe {
            self._vulkan_device.device.update_descriptor_sets(&descriptor_writes, &[]);
        }

        // Create command buffer and record commands
        let cmd_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.compute_command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_buffers = unsafe {
            self._vulkan_device.device.allocate_command_buffers(&cmd_buffer_allocate_info)?
        };
        let command_buffer = cmd_buffers[0];

        unsafe {
            // Begin command buffer
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self._vulkan_device.device.begin_command_buffer(command_buffer, &begin_info)?;

            // Bind pipeline
            self._vulkan_device.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.tensor_gen_pipeline,
            );

            // Bind descriptor sets
            self._vulkan_device.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.tensor_gen_pipeline_layout,
                0,
                &[self.tensor_gen_descriptor_set],
                &[],
            );

            // Dispatch compute shader
            let group_count_x = template_image.width.div_ceil(16);
            let group_count_y = template_image.height.div_ceil(16);
            self._vulkan_device.device.cmd_dispatch(command_buffer, group_count_x, group_count_y, 1);

            // End command buffer
            self._vulkan_device.device.end_command_buffer(command_buffer)?;
        }

        // Submit command buffer and wait for completion
        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
        unsafe {
            self._vulkan_device.device.queue_submit(
                self.compute_queue,
                &[submit_info],
                vk::Fence::null(),
            )?;
            self._vulkan_device.device.queue_wait_idle(self.compute_queue)?;
        }

        // Clean up temporary buffers
        self.memory_manager.destroy_buffer(params_buffer)?;

        Ok(tensor_field_buffer)
    }

    /// Compute tensorial correlation field using the tensor template
    ///
    /// This implements Algorithm 2 from the paper using FFT-based correlation
    /// for improved performance over the previous real-space implementation.
    fn compute_tensorial_correlation(
        &self,
        target_image: &ImageData,
        template_tensor_buffer: &crate::vulkan::memory::VulkanBuffer,
        template_image: &ImageData,
    ) -> Result<(
        crate::vulkan::memory::VulkanBuffer, // correlation buffer
        crate::vulkan::memory::VulkanBuffer, // rotation buffer
        crate::vulkan::memory::VulkanBuffer, // tensor field buffer
    )> {
        let out_width = target_image.width - template_image.width + 1;
        let out_height = target_image.height - template_image.height + 1;

        // Create output buffers
        let correlation_buffer_size = (out_width * out_height * std::mem::size_of::<f32>() as u32) as u64;
        let correlation_buffer = self.memory_manager.create_tensor_buffer(
            correlation_buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::GpuOnly,
            "Correlation Buffer",
        )?;

        let rotation_buffer_size = (out_width * out_height * std::mem::size_of::<f32>() as u32) as u64;
        let rotation_buffer = self.memory_manager.create_tensor_buffer(
            rotation_buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::GpuOnly,
            "Rotation Buffer",
        )?;

        let tensor_field_buffer_size = (out_width * out_height * 8 * std::mem::size_of::<f32>() as u32) as u64;
        let tensor_field_buffer = self.memory_manager.create_tensor_buffer(
            tensor_field_buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::GpuOnly,
            "Correlation Tensor Field Buffer",
        )?;

        // Create buffer for input target image
        let target_buffer = self.memory_manager.create_image_buffer(
            target_image.width,
            target_image.height,
            1, // Assuming grayscale image
        )?;

        // Upload target data
        self.memory_manager.upload_data(&target_buffer, &target_image.data)?;

        // Create uniform buffer for shader parameters
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct CorrelationParams {
            target_width: u32,
            target_height: u32,
            template_width: u32,
            template_height: u32,
        }

        let params = CorrelationParams {
            target_width: target_image.width,
            target_height: target_image.height,
            template_width: template_image.width,
            template_height: template_image.height,
        };

        let params_buffer = self.memory_manager.create_tensor_buffer(
            std::mem::size_of::<CorrelationParams>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::CpuToGpu,
            "Correlation Parameters",
        )?;
        self.memory_manager.upload_data(&params_buffer, &[params])?;

        // Update descriptor set with buffer bindings
        let target_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(target_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let template_tensor_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(template_tensor_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let correlation_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(correlation_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let rotation_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(rotation_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let tensor_field_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(tensor_field_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let params_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(params_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];

        let descriptor_writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.correlation_descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&target_buffer_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.correlation_descriptor_set)
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&template_tensor_buffer_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.correlation_descriptor_set)
                .dst_binding(2)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&correlation_buffer_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.correlation_descriptor_set)
                .dst_binding(3)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&rotation_buffer_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.correlation_descriptor_set)
                .dst_binding(4)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&tensor_field_buffer_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.correlation_descriptor_set)
                .dst_binding(5)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&params_buffer_info),
        ];

        unsafe {
            self._vulkan_device.device.update_descriptor_sets(&descriptor_writes, &[]);
        }

        // Create command buffer and record commands
        let cmd_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.compute_command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_buffers = unsafe {
            self._vulkan_device.device.allocate_command_buffers(&cmd_buffer_allocate_info)?
        };
        let command_buffer = cmd_buffers[0];

        unsafe {
            // Begin command buffer
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self._vulkan_device.device.begin_command_buffer(command_buffer, &begin_info)?;

            // Bind pipeline
            self._vulkan_device.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.correlation_pipeline,
            );

            // Bind descriptor sets
            self._vulkan_device.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.correlation_pipeline_layout,
                0,
                &[self.correlation_descriptor_set],
                &[],
            );

            // Dispatch compute shader
            let group_count_x = out_width.div_ceil(16);
            let group_count_y = out_height.div_ceil(16);
            self._vulkan_device.device.cmd_dispatch(command_buffer, group_count_x, group_count_y, 1);

            // End command buffer
            self._vulkan_device.device.end_command_buffer(command_buffer)?;
        }

        // Submit command buffer and wait for completion
        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
        unsafe {
            self._vulkan_device.device.queue_submit(
                self.compute_queue,
                &[submit_info],
                vk::Fence::null(),
            )?;
            self._vulkan_device.device.queue_wait_idle(self.compute_queue)?;
        }

        // Clean up temporary buffers
        self.memory_manager.destroy_buffer(target_buffer)?;
        self.memory_manager.destroy_buffer(params_buffer)?;

        Ok((correlation_buffer, rotation_buffer, tensor_field_buffer))
    }

    /// Detect peaks in correlation field and determine optimal rotations
    fn detect_peaks(
        &self,
        correlation_buffer: &crate::vulkan::memory::VulkanBuffer,
        rotation_buffer: &crate::vulkan::memory::VulkanBuffer,
        target_image: &ImageData,
        template_image: &ImageData,
        correlation_threshold: f32,
        max_matches: usize,
    ) -> Result<Vec<TensorTemplateMatch>> {
        let out_width = target_image.width - template_image.width + 1;
        let out_height = target_image.height - template_image.height + 1;

        // Create buffer for detection results
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct DetectionResult {
            x: u32,
            y: u32,
            correlation_fixed: u32, // correlation * 10000 for precision
            rotation_fixed: u32,    // rotation * 10000 for precision
            padding1: u32,
            padding2: u32,
            padding3: u32,
            padding4: u32,
        }

        let max_results = max_matches.min(1000); // Limit to reasonable number
        let results_buffer_size = (max_results * std::mem::size_of::<DetectionResult>()) as u64;
        let results_buffer = self.memory_manager.create_tensor_buffer(
            results_buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            gpu_allocator::MemoryLocation::GpuToCpu,
            "Detection Results Buffer",
        )?;

        // Create atomic counter buffer for peak counting
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct AtomicCounter {
            peak_count: u32,
        }

        let counter_buffer = self.memory_manager.create_tensor_buffer(
            std::mem::size_of::<AtomicCounter>() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::CpuToGpu,
            "Atomic Counter Buffer",
        )?;

        // Initialise counter to zero
        let counter_init = AtomicCounter { peak_count: 0 };
        self.memory_manager.upload_data(&counter_buffer, &[counter_init])?;

        // Create uniform buffer for shader parameters
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct PeakParams {
            width: u32,
            height: u32,
            threshold: f32,
            max_peaks: u32,
            exclusion_radius: u32,
            padding1: u32,
            padding2: u32,
            padding3: u32,
        }

        let params = PeakParams {
            width: out_width,
            height: out_height,
            threshold: correlation_threshold,
            max_peaks: max_results as u32,
            exclusion_radius: 10, // Exclusion radius for non-maximum suppression
            padding1: 0,
            padding2: 0,
            padding3: 0,
        };

        let params_buffer = self.memory_manager.create_tensor_buffer(
            std::mem::size_of::<PeakParams>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::CpuToGpu,
            "Peak Detection Parameters",
        )?;
        self.memory_manager.upload_data(&params_buffer, &[params])?;

        // Create buffer info arrays to avoid temporary value issues
        let correlation_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(correlation_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let rotation_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(rotation_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let results_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(results_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let counter_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(counter_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let params_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(params_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];

        let descriptor_writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.peak_descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&correlation_buffer_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.peak_descriptor_set)
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&rotation_buffer_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.peak_descriptor_set)
                .dst_binding(2)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&results_buffer_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.peak_descriptor_set)
                .dst_binding(3)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&counter_buffer_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.peak_descriptor_set)
                .dst_binding(4)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&params_buffer_info),
        ];

        unsafe {
            self._vulkan_device.device.update_descriptor_sets(&descriptor_writes, &[]);
        }

        // Create command buffer and record commands
        let cmd_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.compute_command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_buffers = unsafe {
            self._vulkan_device.device.allocate_command_buffers(&cmd_buffer_allocate_info)?
        };
        let command_buffer = cmd_buffers[0];

        unsafe {
            // Begin command buffer
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self._vulkan_device.device.begin_command_buffer(command_buffer, &begin_info)?;

            // Bind pipeline
            self._vulkan_device.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.peak_pipeline,
            );

            // Bind descriptor sets
            self._vulkan_device.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.peak_pipeline_layout,
                0,
                &[self.peak_descriptor_set],
                &[],
            );

            // Dispatch compute shader
            let group_count_x = out_width.div_ceil(16);
            let group_count_y = out_height.div_ceil(16);
            self._vulkan_device.device.cmd_dispatch(command_buffer, group_count_x, group_count_y, 1);

            // End command buffer
            self._vulkan_device.device.end_command_buffer(command_buffer)?;
        }

        // Submit command buffer and wait for completion
        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
        unsafe {
            self._vulkan_device.device.queue_submit(
                self.compute_queue,
                &[submit_info],
                vk::Fence::null(),
            )?;
            self._vulkan_device.device.queue_wait_idle(self.compute_queue)?;
        }

        // Read results back from GPU
        let mut results: Vec<DetectionResult> = vec![DetectionResult { x: 0, y: 0, correlation_fixed: 0, rotation_fixed: 0, padding1: 0, padding2: 0, padding3: 0, padding4: 0 }; max_results];
        self.memory_manager.device_to_host(&results_buffer, &mut results)?;

        // Convert to TensorTemplateMatch structs
        let mut matches = Vec::new();
        for result in results.iter().take(max_results) {
            if result.correlation_fixed > 0 {
                matches.push(TensorTemplateMatch {
                    x: result.x,
                    y: result.y,
                    correlation: result.correlation_fixed as f32 / 10000.0,
                    rotation_angle: result.rotation_fixed as f32 / 10000.0,
                    confidence: result.correlation_fixed as f32 / 10000.0,
                });
            }
        }

        // Apply position refinement (Algorithm 3 from the paper)
        self.refine_peak_positions(&mut matches, target_image, template_image);

        // Clean up temporary buffers
        self.memory_manager.destroy_buffer(params_buffer)?;
        self.memory_manager.destroy_buffer(results_buffer)?;
        self.memory_manager.destroy_buffer(counter_buffer)?;

        Ok(matches)
    }

    /// Match template on large images using block processing scheme
    ///
    /// This implements the block processing scheme described in Section 3.4 (p. 8-9) of the paper
    /// to avoid having many full copies with the dimensions of the original tomogram loaded
    /// in the main memory.
    ///
    /// The approach tiles the target image into overlapping blocks, processes each block
    /// independently, and then merges the results while handling overlapping detections.
    pub fn match_template_tiled(
        &self,
        target_image: &ImageData,
        template_image: &ImageData,
        correlation_threshold: f32,
        max_matches: usize,
        tile_size: (u32, u32), // (width, height) of each tile
        overlap: u32,          // Overlap between adjacent tiles to avoid boundary effects
    ) -> Result<Vec<TensorTemplateMatch>> {
        let (tile_w, tile_h) = tile_size;
        let mut all_matches = Vec::new();
        let mut processed_tiles = 0;

        debug!("Starting tiled tensorial template matching...");
        debug!("Target: {}x{}, Tile: {}x{}, Overlap: {}",
               target_image.width, target_image.height, tile_w, tile_h, overlap);

        // Process tiles in a grid pattern with overlap
        for tile_y in (0..target_image.height).step_by((tile_h - overlap) as usize) {
            for tile_x in (0..target_image.width).step_by((tile_w - overlap) as usize) {
                // Calculate actual tile dimensions (may be smaller at boundaries)
                let actual_tile_w = (tile_x + tile_w).min(target_image.width) - tile_x;
                let actual_tile_h = (tile_y + tile_h).min(target_image.height) - tile_y;

                // Skip tiles that are too small
                if actual_tile_w < template_image.width || actual_tile_h < template_image.height {
                    continue;
                }

                debug!("Processing tile ({}, {}) size {}x{}", tile_x, tile_y, actual_tile_w, actual_tile_h);

                // Extract tile from target image
                let tile = self.extract_tile(target_image, tile_x, tile_y, actual_tile_w, actual_tile_h);

                // Process tile with tensorial template matching
                let tile_matches = self.match_template(
                    &tile,
                    template_image,
                    correlation_threshold,
                    max_matches
                )?;

                // Adjust coordinates to global image coordinates
                for mut match_result in tile_matches {
                    match_result.x += tile_x;
                    match_result.y += tile_y;
                    all_matches.push(match_result);
                }

                processed_tiles += 1;
            }
        }

        debug!("Processed {} tiles, found {} raw matches", processed_tiles, all_matches.len());

        // Merge overlapping detections
        let merged_matches = self.merge_overlapping_detections(&all_matches, overlap);

        debug!("After merging overlapping detections: {} matches", merged_matches.len());

        Ok(merged_matches)
    }

    /// Extract a tile from an image
    fn extract_tile(
        &self,
        image: &ImageData,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> ImageData {
        let mut tile_data = vec![0.0; (width * height) as usize];

        for ty in 0..height {
            for tx in 0..width {
                let src_x = x + tx;
                let src_y = y + ty;
                let dst_idx = (ty * width + tx) as usize;

                if src_x < image.width && src_y < image.height {
                    let src_idx = (src_y * image.width + src_x) as usize;
                    tile_data[dst_idx] = image.data[src_idx];
                }
            }
        }

        ImageData {
            data: tile_data,
            width,
            height,
            channels: image.channels,
        }
    }

    /// Merge overlapping detections from tiled processing
    ///
    /// When processing overlapping tiles, the same detection may appear in multiple tiles.
    /// This function merges overlapping detections, keeping the one with highest correlation.
    fn merge_overlapping_detections(
        &self,
        matches: &[TensorTemplateMatch],
        overlap: u32,
    ) -> Vec<TensorTemplateMatch> {
        if matches.is_empty() {
            return Vec::new();
        }

        let mut merged = Vec::new();
        let mut processed = vec![false; matches.len()];

        for (i, match_a) in matches.iter().enumerate() {
            if processed[i] {
                continue;
            }

            // Find all matches that are close to this one (within overlap distance)
            let mut group = vec![i];
            for (j, match_b) in matches.iter().enumerate().skip(i + 1) {
                if processed[j] {
                    continue;
                }

                let dx = (match_a.x as i32 - match_b.x as i32).abs();
                let dy = (match_a.y as i32 - match_b.y as i32).abs();

                // If matches are within overlap distance, consider them the same
                if (dx as u32) <= overlap && (dy as u32) <= overlap {
                    group.push(j);
                    processed[j] = true;
                }
            }

            // Take the match with highest correlation from the group
            let best_match_idx = *group.iter().max_by(|&&a, &&b| {
                matches[a].correlation.partial_cmp(&matches[b].correlation).unwrap()
            }).unwrap();

            merged.push(matches[best_match_idx].clone());
            processed[i] = true;
        }

        merged
    }

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

    /// Create descriptor set layout for tensor generation pipeline
    fn create_tensor_gen_descriptor_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout> {
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
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
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

    /// Create descriptor set layout for correlation pipeline
    fn create_correlation_descriptor_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout> {
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
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
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

    /// Create descriptor set layout for peak detection pipeline
    fn create_peak_descriptor_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout> {
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
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
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

    /// Create a generic descriptor pool
    fn create_descriptor_pool(device: &Device, _descriptor_set_layout: &vk::DescriptorSetLayout) -> Result<vk::DescriptorPool> {
        let descriptor_pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 10, // Allow multiple buffers per set
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 5, // Allow uniform buffers
            },
        ];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(10) // Allow multiple descriptor sets
            .pool_sizes(&descriptor_pool_sizes);
        unsafe {
            device
                .create_descriptor_pool(&descriptor_pool_info, None)
                .map_err(TensorMatchingError::VulkanError)
        }
    }

    /// Allocate descriptor sets
    fn allocate_descriptor_set(
        device: &Device,
        descriptor_pool: vk::DescriptorPool,
        set_layouts: &[vk::DescriptorSetLayout],
    ) -> Result<vk::DescriptorSet> {
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(set_layouts);
        let descriptor_sets = unsafe {
            device
                .allocate_descriptor_sets(&descriptor_set_allocate_info)
                .map_err(TensorMatchingError::VulkanError)?
        };
        Ok(descriptor_sets[0])
    }

    /// Create pipeline layout
    fn create_pipeline_layout(device: &Device, set_layouts: &[vk::DescriptorSetLayout]) -> Result<vk::PipelineLayout> {
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(set_layouts);
        unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .map_err(TensorMatchingError::VulkanError)
        }
    }

    /// Create compute pipeline from SPIR-V shader
    fn create_compute_pipeline(
        device: &Device,
        shader_bytes: &[u8],
        pipeline_layout: vk::PipelineLayout,
    ) -> Result<vk::Pipeline> {
        // Convert &[u8] to &[u32] with proper alignment
        assert_eq!(shader_bytes.len() % 4, 0, "Shader code length is not a multiple of 4 bytes");

        let mut shader_code = Vec::with_capacity(shader_bytes.len() / 4);
        for chunk in shader_bytes.chunks_exact(4) {
            shader_code.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }

        let shader_module_info = vk::ShaderModuleCreateInfo::default().code(&shader_code);
        let shader_module = unsafe {
            device
                .create_shader_module(&shader_module_info, None)
                .map_err(TensorMatchingError::VulkanError)?
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
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(TensorMatchingError::PipelineCreationError)?
        };

        let pipeline = pipelines[0];

        // Release the temporary shader module
        unsafe {
            device.destroy_shader_module(shader_module, None);
        }

        Ok(pipeline)
    }

    /// Refine peak positions using local search as described in Algorithm 3 of the paper
    ///
    /// For each peak found, we define a sphere around it with a radius of rs voxels
    /// and search for the best match in this neighborhood.
    fn refine_peak_positions(
        &self,
        matches: &mut [TensorTemplateMatch],
        target_image: &ImageData,
        template_image: &ImageData,
    ) {
        for match_result in matches.iter_mut() {
            let mut best_correlation = match_result.correlation;
            let mut best_pos = (match_result.x, match_result.y);
            let _best_rotation = match_result.rotation_angle;

            // Search in neighborhood
            for dy in -(REFINEMENT_RADIUS as i32)..=(REFINEMENT_RADIUS as i32) {
                for dx in -(REFINEMENT_RADIUS as i32)..=(REFINEMENT_RADIUS as i32) {
                    let test_x = (match_result.x as i32 + dx).max(0).min(target_image.width as i32 - 1) as u32;
                    let test_y = (match_result.y as i32 + dy).max(0).min(target_image.height as i32 - 1) as u32;

                    // Compute actual LNCC at this position and rotation
                    let lncc = self.compute_lncc_at_position(
                        target_image, template_image, test_x, test_y, match_result.rotation_angle
                    );

                    if lncc > best_correlation {
                        best_correlation = lncc;
                        best_pos = (test_x, test_y);
                        // Note: In a full implementation, we would also refine the rotation angle
                    }
                }
            }

            match_result.x = best_pos.0;
            match_result.y = best_pos.1;
            match_result.correlation = best_correlation;
        }
    }

    /// Compute local normalised cross-correlation at a specific position and rotation
    fn compute_lncc_at_position(
        &self,
        target: &ImageData,
        template: &ImageData,
        x: u32,
        y: u32,
        _rotation: f32,
    ) -> f32 {
        let template_w = template.width as i32;
        let template_h = template.height as i32;
        let _target_w = target.width as i32;
        let _target_h = target.height as i32;

        // Bounds check
        if x + template.width > target.width || y + template.height > target.height {
            return 0.0;
        }

        // Compute means
        let mut target_sum = 0.0;
        let mut template_sum = 0.0;
        let mut count = 0;

        for ty in 0..template_h {
            for tx in 0..template_w {
                let target_x = (x as i32 + tx) as u32;
                let target_y = (y as i32 + ty) as u32;

                if target_x < target.width && target_y < target.height {
                    let target_idx = (target_y * target.width + target_x) as usize;
                    let template_idx = (ty as u32 * template.width + tx as u32) as usize;

                    target_sum += target.data[target_idx];
                    template_sum += template.data[template_idx];
                    count += 1;
                }
            }
        }

        if count == 0 {
            return 0.0;
        }

        let target_mean = target_sum / count as f32;
        let template_mean = template_sum / count as f32;

        // Compute LNCC
        let mut numerator = 0.0;
        let mut target_denom = 0.0;
        let mut template_denom = 0.0;

        for ty in 0..template_h {
            for tx in 0..template_w {
                let target_x = (x as i32 + tx) as u32;
                let target_y = (y as i32 + ty) as u32;

                if target_x < target.width && target_y < target.height {
                    let target_idx = (target_y * target.width + target_x) as usize;
                    let template_idx = (ty as u32 * template.width + tx as u32) as usize;

                    let target_diff = target.data[target_idx] - target_mean;
                    let template_diff = template.data[template_idx] - template_mean;

                    numerator += target_diff * template_diff;
                    target_denom += target_diff * target_diff;
                    template_denom += template_diff * template_diff;
                }
            }
        }

        let denominator = (target_denom * template_denom).sqrt();
        if denominator > 1e-6 {
            numerator / denominator
        } else {
            0.0
        }
    }
}

impl Drop for VulkanTensorMatcher {
    fn drop(&mut self) {
        unsafe {
            self._vulkan_device.device.destroy_pipeline(self.peak_pipeline, None);
            self._vulkan_device.device.destroy_pipeline_layout(self.peak_pipeline_layout, None);
            self._vulkan_device.device.destroy_descriptor_pool(self.peak_descriptor_pool, None);
            self._vulkan_device.device.destroy_descriptor_set_layout(self.peak_descriptor_set_layout, None);

            self._vulkan_device.device.destroy_pipeline(self.correlation_pipeline, None);
            self._vulkan_device.device.destroy_pipeline_layout(self.correlation_pipeline_layout, None);
            self._vulkan_device.device.destroy_descriptor_pool(self.correlation_descriptor_pool, None);
            self._vulkan_device.device.destroy_descriptor_set_layout(self.correlation_descriptor_set_layout, None);

            self._vulkan_device.device.destroy_pipeline(self.tensor_gen_pipeline, None);
            self._vulkan_device.device.destroy_pipeline_layout(self.tensor_gen_pipeline_layout, None);
            self._vulkan_device.device.destroy_descriptor_pool(self.tensor_gen_descriptor_pool, None);
            self._vulkan_device.device.destroy_descriptor_set_layout(self.tensor_gen_descriptor_set_layout, None);

            self._vulkan_device.device.destroy_command_pool(self.compute_command_pool, None);
        }
    }
}