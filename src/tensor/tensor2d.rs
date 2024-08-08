use crate::image::fft::{apply_s_operator, generate_soft_mask, MASK_INNER_RATIO, MASK_OUTER_RATIO};
use bytemuck::{Pod, Zeroable};
use std::f32::consts::PI;

/// Optimised 2D tensor for Vulkan compute shaders
#[derive(Debug, Clone, Copy)]
#[repr(C, align(32))] // 32-byte alignment for optimal GPU access
pub struct VulkanTensor2D {
    /// Degree-4 symmetric tensor components for 2D rotations
    /// [cos⁴θ, 4cos³θsinθ, 6cos²θsin²θ, 4cosθsin³θ, sin⁴θ, 0, 0, 0]
    pub components: [f32; 8], // Padded to 8 for GPU alignment
}

unsafe impl Pod for VulkanTensor2D {}
unsafe impl Zeroable for VulkanTensor2D {}

impl VulkanTensor2D {
    pub fn zero() -> Self {
        Self {
            components: [0.0; 8],
        }
    }

    /// Create tensor from rotation angle and intensity
    pub fn from_rotation(angle: f32, intensity: f32) -> Self {
        let cos_theta = angle.cos();
        let sin_theta = angle.sin();

        // Precompute powers for efficiency
        let cos2 = cos_theta * cos_theta;
        let sin2 = sin_theta * sin_theta;
        let cos_sin = cos_theta * sin_theta;

        Self {
            components: [
                intensity * cos2 * cos2,              // cos⁴θ
                intensity * 4.0 * cos2 * cos_sin,     // 4cos³θsinθ
                intensity * 6.0 * cos2 * sin2,        // 6cos²θsin²θ
                intensity * 4.0 * cos_sin * sin2,     // 4cosθsin³θ
                intensity * sin2 * sin2,              // sin⁴θ
                0.0, 0.0, 0.0, // GPU alignment padding
            ],
        }
    }

    /// Compute Frobenius norm for correlation strength
    pub fn frobenius_norm(&self) -> f32 {
        self.components[..5].iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt()
    }

    /// Find optimal rotation angle using analytical solution
    pub fn optimal_rotation_angle(&self) -> f32 {
        let c0 = self.components[0]; // cos⁴θ coefficient
        let c1 = self.components[1]; // 4cos³θsinθ coefficient
        let c4 = self.components[4]; // sin⁴θ coefficient

        // For degree-4 tensors in 2D, we can solve analytically
        // The optimal angle maximises the tensor contraction
        if c1.abs() > 1e-6 {
            // Use atan2 for robust angle computation
            let numerator = 2.0 * c1;
            let denominator = c0 - c4;
            0.25 * numerator.atan2(denominator)
        } else {
            // Handle degenerate cases
            if c0 > c4 {
                0.0
            } else {
                PI / 2.0
            }
        }
    }

    /// Tensor addition for accumulating rotations
    pub fn add(&self, other: &Self) -> Self {
        let mut result = *self;
        for (a, b) in result.components.iter_mut().zip(other.components.iter()) {
            *a += b;
        }
        result
    }

    /// Scalar multiplication
    pub fn scale(&self, factor: f32) -> Self {
        let mut result = *self;
        for component in result.components.iter_mut() {
            *component *= factor;
        }
        result
    }

    /// Compute tensor contraction for correlation
    pub fn contract(&self, other: &Self) -> f32 {
        self.components[..5].iter()
            .zip(other.components[..5].iter())
            .map(|(a, b)| a * b)
            .sum()
    }
}

/// Tensor field for an entire 2D template
#[derive(Debug, Clone)]
pub struct TensorField2D {
    pub tensors: Vec<VulkanTensor2D>,
    pub width: u32,
    pub height: u32,
    pub total_intensity: f32, // For normalisation
}

impl TensorField2D {
    /// Create tensor field from image data with proper normalisation as per the paper
    ///
    /// Implements Equation 4 from the paper:
    /// t' = m(S(t) - μ) / sqrt(<S(t)², 1> - <S(t), 1>² / M)
    pub fn from_image(image_data: &[f32], width: u32, height: u32, num_angles: usize) -> Self {
        // Generate soft mask
        let template_radius = ((width.min(height) as f32) / 2.0).max(1.0);
        let inner_radius = template_radius * MASK_INNER_RATIO;
        let outer_radius = template_radius * MASK_OUTER_RATIO;
        let mask = generate_soft_mask(width, height, inner_radius, outer_radius);

        // Apply S operator (mask + low-pass filter)
        let s_t = apply_s_operator(image_data, &mask, width, height);

        // Compute mask sum (M in Equation 4)
        let m: f32 = mask.iter().sum();

        // Compute mean (μ in Equation 4)
        let sum_masked: f32 = s_t.iter().zip(mask.iter()).map(|(s, m)| s * m).sum();
        let mu = if m > 1e-6 { sum_masked / m } else { 0.0 };

        // Compute normalisation factor
        let sum_sq: f32 = s_t.iter().map(|x| x * x).sum();
        let sum_masked_sq: f32 = s_t.iter().zip(mask.iter()).map(|(s, m)| s * m).sum();
        let variance = (sum_sq - sum_masked_sq * sum_masked_sq / m.max(1e-6)).max(0.0);
        let norm = variance.sqrt().max(1e-6);

        // Normalise template
        let normalised_template: Vec<f32> = s_t.iter().zip(mask.iter())
            .map(|(s, m)| m * (s - mu) / norm)
            .collect();

        // Generate tensor field from normalised template
        let mut tensors = vec![VulkanTensor2D::zero(); (width * height) as usize];
        let angle_step = 2.0 * PI / num_angles as f32;
        let mut total_intensity = 0.0;

        // Process each pixel
        (0..height).flat_map(|y| (0..width).map(move |x| (y, x)))
            .for_each(|(y, x)| {
                let pixel_idx = (y * width + x) as usize;
                let base_intensity = normalised_template[pixel_idx];
                total_intensity += base_intensity;

                let mut accumulated_tensor = VulkanTensor2D::zero();

                // Integrate over all rotation angles
                for angle_idx in 0..num_angles {
                    let angle = angle_idx as f32 * angle_step;

                    // Get intensity at this position for this rotation
                    // In practice, this would involve proper interpolation
                    let rotated_intensity = Self::get_rotated_intensity(
                        &normalised_template, width, height, x, y, angle
                    );

                    // Create tensor for this rotation
                    let rotation_tensor = VulkanTensor2D::from_rotation(angle, rotated_intensity);
                    accumulated_tensor = accumulated_tensor.add(&rotation_tensor);
                }

                // Normalise by number of samples
                tensors[pixel_idx] = accumulated_tensor.scale(1.0 / num_angles as f32);
            });

        Self {
            tensors,
            width,
            height,
            total_intensity,
        }
    }

    /// Get pixel intensity after rotation with bilinear interpolation
    fn get_rotated_intensity(
        image: &[f32],
        width: u32,
        height: u32,
        x: u32,
        y: u32,
        angle: f32
    ) -> f32 {
        let centre_x = width as f32 / 2.0;
        let centre_y = height as f32 / 2.0;

        // Translate to centre
        let dx = x as f32 - centre_x;
        let dy = y as f32 - centre_y;

        // Apply rotation
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let rotated_x = dx * cos_a - dy * sin_a + centre_x;
        let rotated_y = dx * sin_a + dy * cos_a + centre_y;

        // Bilinear interpolation
        if rotated_x >= 0.0 && rotated_x < (width - 1) as f32
            && rotated_y >= 0.0 && rotated_y < (height - 1) as f32 {

            let x0 = rotated_x.floor() as u32;
            let y0 = rotated_y.floor() as u32;
            let x1 = (x0 + 1).min(width - 1);
            let y1 = (y0 + 1).min(height - 1);

            let fx = rotated_x - x0 as f32;
            let fy = rotated_y - y0 as f32;

            let i00 = image[(y0 * width + x0) as usize];
            let i10 = image[(y0 * width + x1) as usize];
            let i01 = image[(y1 * width + x0) as usize];
            let i11 = image[(y1 * width + x1) as usize];

            let i0 = i00 * (1.0 - fx) + i10 * fx;
            let i1 = i01 * (1.0 - fx) + i11 * fx;

            i0 * (1.0 - fy) + i1 * fy
        } else {
            0.0
        }
    }

    /// Prepare data for GPU upload
    pub fn as_gpu_data(&self) -> &[VulkanTensor2D] {
        &self.tensors
    }

    /// Get tensor at specific pixel coordinates
    pub fn get_tensor(&self, x: u32, y: u32) -> Option<&VulkanTensor2D> {
        if x < self.width && y < self.height {
            Some(&self.tensors[(y * self.width + x) as usize])
        } else {
            None
        }
    }
}