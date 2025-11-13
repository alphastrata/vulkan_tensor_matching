use crate::image::fft::{MASK_INNER_RATIO, MASK_OUTER_RATIO, apply_s_operator, generate_soft_mask};
use bytemuck::{Pod, Zeroable};
use std::f32::consts::PI;

/// Bilinear interpolation helper for tensor field computation
fn bilinear_interpolate(
    data: &[f32],
    width: u32,
    height: u32,
    x: f32,
    y: f32,
) -> f32 {
    // Bounds check - return 0 for out-of-bounds positions
    if x < 0.0 || x >= (width - 1) as f32 || y < 0.0 || y >= (height - 1) as f32 {
        return 0.0;
    }

    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let i00 = data[(y0 * width + x0) as usize];
    let i10 = data[(y0 * width + x1) as usize];
    let i01 = data[(y1 * width + x0) as usize];
    let i11 = data[(y1 * width + x1) as usize];

    let i0 = i00 * (1.0 - fx) + i10 * fx;
    let i1 = i01 * (1.0 - fx) + i11 * fx;

    i0 * (1.0 - fy) + i1 * fy
}

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
                intensity * cos2 * cos2,          // cos⁴θ
                intensity * 4.0 * cos2 * cos_sin, // 4cos³θsinθ
                intensity * 6.0 * cos2 * sin2,    // 6cos²θsin²θ
                intensity * 4.0 * cos_sin * sin2, // 4cosθsin³θ
                intensity * sin2 * sin2,          // sin⁴θ
                0.0,
                0.0,
                0.0, // GPU alignment padding
            ],
        }
    }

    /// Compute Frobenius norm for correlation strength (scalar field ĉₙ for peak detection)
    ///
    /// Per Martinez-Sanchez et al. Section 3.2, the Frobenius norm serves as an
    /// excellent proxy for the spectral norm: ‖T‖_σ ≥ ‖T‖_F / 2^{n-1}
    /// Large Frobenius norm implies large spectral norm, making it suitable
    /// for identifying match locations via peak detection.
    pub fn frobenius_norm(&self) -> f32 {
        self.components[..5]
            .iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt()
    }

    /// Find optimal rotation angle using closed-form solution for 2D degree-4 tensors.
    ///
    /// The tensor contraction C·R^{⊙4} gives a trigonometric polynomial:
    ///   f(θ) = C₀cos⁴θ + C₁cos³θsinθ + C₂cos²θsin²θ + C₃cosθsin³θ + C₄sin⁴θ
    ///
    /// Using double-angle identities, this can be rewritten as:
    ///   f(θ) = a₀ + a₂cos(2θ) + b₂sin(2θ) + a₄cos(4θ) + b₄sin(4θ)
    ///
    /// The maximum is found by solving df/dθ = 0. For numerical stability,
    /// we use a combination of analytical solution and refinement.
    pub fn optimal_rotation_angle(&self) -> f32 {
        let c = &self.components;
        
        // Convert to double-angle representation for cleaner optimization
        // Using identities:
        //   cos⁴θ = (3 + 4cos2θ + cos4θ)/8
        //   sin⁴θ = (3 - 4cos2θ + cos4θ)/8
        //   cos³θsinθ = (sin2θ + sin4θ/2)/4
        //   cosθsin³θ = (sin2θ - sin4θ/2)/4
        //   cos²θsin²θ = (1 - cos4θ)/8
        
        // Coefficients for 4θ terms (dominant for degree-4)
        let a4 = c[0] - 3.0 * c[2] + c[4];  // cos(4θ) coefficient
        let b4 = c[1] - c[3];                // sin(4θ) coefficient
        
        // Coefficients for 2θ terms
        let a2 = c[0] - c[4];  // cos(2θ) coefficient  
        let b2 = c[1] + c[3];  // sin(2θ) coefficient
        
        // Initial estimate from 4θ terms (dominant frequency)
        let theta_4theta = if a4.abs() > 1e-10 || b4.abs() > 1e-10 {
            0.25 * b4.atan2(a4)
        } else {
            0.0
        };
        
        // Initial estimate from 2θ terms
        let theta_2theta = if a2.abs() > 1e-10 || b2.abs() > 1e-10 {
            0.5 * b2.atan2(a2)
        } else {
            0.0
        };
        
        // Evaluate f(θ) at candidate angles and pick the best
        let candidates = [
            theta_4theta,
            theta_4theta + std::f32::consts::PI / 2.0,  // 180° symmetry
            theta_2theta,
            theta_2theta + std::f32::consts::PI,  // 360° symmetry
        ];
        
        let mut best_theta = theta_4theta;
        let mut max_val = f32::NEG_INFINITY;
        
        for &theta in &candidates {
            let val = self.evaluate_at_angle(theta);
            if val > max_val {
                max_val = val;
                best_theta = theta;
            }
        }
        
        // Fine refinement: search small neighborhood around best candidate
        let step = 0.01;  // ~0.57 degrees
        for i in -10..=10 {
            let theta = best_theta + i as f32 * step;
            let val = self.evaluate_at_angle(theta);
            if val > max_val {
                max_val = val;
                best_theta = theta;
            }
        }
        
        // Normalize to [0, 2π)
        let two_pi = 2.0 * std::f32::consts::PI;
        ((best_theta % two_pi) + two_pi) % two_pi
    }
    
    /// Evaluate the tensor contraction at a specific angle θ
    /// f(θ) = C·R(θ)^{⊙4}
    fn evaluate_at_angle(&self, theta: f32) -> f32 {
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let cos2 = cos_t * cos_t;
        let sin2 = sin_t * sin_t;
        
        // Direct evaluation of the trigonometric polynomial
        self.components[0] * cos2 * cos2 +                    // C₀cos⁴θ
        self.components[1] * cos2 * cos_t * sin_t +           // C₁cos³θsinθ
        self.components[2] * cos2 * sin2 +                    // C₂cos²θsin²θ
        self.components[3] * cos_t * sin_t * sin2 +           // C₃cosθsin³θ
        self.components[4] * sin2 * sin2                      // C₄sin⁴θ
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
        self.components[..5]
            .iter()
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
    /// Build template tensor field for rotation-invariant matching.
    ///
    /// Implements Algorithm 1 from Martinez-Sanchez et al. (arXiv:2408.02398):
    ///   T(t) = ∫_{SO(2)} R^{⊙4} S(t')_R dR
    ///
    /// For 2D degree-4 tensors, this produces 5 independent components per pixel:
    ///   T₀(p) = (1/N) Σ_θ t'(R_{-θ}p) × cos⁴(θ)
    ///   T₁(p) = (1/N) Σ_θ t'(R_{-θ}p) × 4cos³(θ)sin(θ)
    ///   T₂(p) = (1/N) Σ_θ t'(R_{-θ}p) × 6cos²(θ)sin²(θ)
    ///   T₃(p) = (1/N) Σ_θ t'(R_{-θ}p) × 4cos(θ)sin³(θ)
    ///   T₄(p) = (1/N) Σ_θ t'(R_{-θ}p) × sin⁴(θ)
    ///
    /// The binomial coefficients C(4,k) = [1, 4, 6, 4, 1] are included so that
    /// the tensor contraction C·R^{⊙4} correctly evaluates the trigonometric
    /// polynomial for max-projection.
    pub fn from_image_angular(image_data: &[f32], width: u32, height: u32) -> Self {
        let n = (width * height) as usize;
        let num_angles = 360;
        let angle_step = 2.0 * std::f32::consts::PI / num_angles as f32;

        // Normalize template first (Eq. 4 from paper)
        let mean: f32 = image_data.iter().sum::<f32>() / n as f32;
        let var: f32 = image_data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
        let std_dev = var.sqrt().max(1e-6);
        let inv_norm = 1.0 / (std_dev * (n as f32).sqrt());

        let mut tensors = vec![VulkanTensor2D::zero(); n];
        let mut total_intensity = 0.0f32;

        let center_x = width as f32 / 2.0;
        let center_y = height as f32 / 2.0;

        // For each pixel, integrate over all rotations
        for (pixel_idx, tensor) in tensors.iter_mut().enumerate() {
            let px = (pixel_idx % width as usize) as f32;
            let py = (pixel_idx / width as usize) as f32;

            // Accumulate contributions from all rotation angles
            for angle_idx in 0..num_angles {
                let theta = angle_idx as f32 * angle_step;
                let cos_t = theta.cos();
                let sin_t = theta.sin();

                // Compute rotated coordinates (rotate pixel position around center)
                // To sample t'(R_{-θ}p), we rotate the coordinate by +θ
                let dx = px - center_x;
                let dy = py - center_y;
                let rx = dx * cos_t - dy * sin_t + center_x;
                let ry = dx * sin_t + dy * cos_t + center_y;

                // Bilinear interpolation to get t' at rotated position
                let rotated_val = bilinear_interpolate(image_data, width, height, rx, ry);

                // Apply normalization
                let t_prime = (rotated_val - mean) * inv_norm;

                // Accumulate weighted by tensor components R^{⊙4}
                // Binomial coefficients C(4,k) = [1, 4, 6, 4, 1]
                let cos2 = cos_t * cos_t;
                let sin2 = sin_t * sin_t;
                tensor.components[0] += t_prime * cos2 * cos2;              // cos⁴θ
                tensor.components[1] += t_prime * 4.0 * cos2 * cos_t * sin_t; // 4cos³θsinθ
                tensor.components[2] += t_prime * 6.0 * cos2 * sin2;          // 6cos²θsin²θ
                tensor.components[3] += t_prime * 4.0 * cos_t * sin_t * sin2; // 4cosθsin³θ
                tensor.components[4] += t_prime * sin2 * sin2;                // sin⁴θ
            }

            // Normalize by number of angles
            let norm_factor = 1.0 / num_angles as f32;
            for i in 0..5 {
                tensor.components[i] *= norm_factor;
            }

            total_intensity += tensor.frobenius_norm();
        }

        Self { tensors, width, height, total_intensity }
    }

    /// Create tensor field from image data with proper normalisation as per the paper
    ///
    /// Implements Equation 4 from the paper:
    /// t' = m(S(t) - μ) / sqrt(<S(t)², 1> - <S(t), 1>² / M)
    pub fn from_image(image_data: &[f32], width: u32, height: u32, num_angles: usize) -> Self {
        // Generate soft mask
        let template_radius = ((width.min(height) as f32) / 2.0).max(1.0);
        let inner_radius = template_radius * (MASK_INNER_RATIO as f32 / 1000.0);
        let outer_radius = template_radius * (MASK_OUTER_RATIO as f32 / 1000.0);
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
        let normalised_template: Vec<f32> = s_t
            .iter()
            .zip(mask.iter())
            .map(|(s, m)| m * (s - mu) / norm)
            .collect();

        // Generate tensor field from normalised template
        let mut tensors = vec![VulkanTensor2D::zero(); (width * height) as usize];
        let angle_step = 2.0 * PI / num_angles as f32;
        let mut total_intensity = 0.0;

        // Process each pixel
        (0..height)
            .flat_map(|y| (0..width).map(move |x| (y, x)))
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
                        &normalised_template,
                        width,
                        height,
                        x,
                        y,
                        angle,
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
        angle: f32,
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
        if rotated_x >= 0.0
            && rotated_x < (width - 1) as f32
            && rotated_y >= 0.0
            && rotated_y < (height - 1) as f32
        {
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
