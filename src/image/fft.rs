//! FFT-based correlation implementation for tensorial template matching
//!
//! Implements the Fourier-domain correlation as described in the paper:
//! "Tensorial template matching for fast cross-correlation with rotations"
//! Martinez-Sanchez et al., arXiv:2408.02398v1 [cs.CV], Section 3 (p. 4-8)

use rustfft::{FftPlanner, num_complex::Complex};
use crate::tensor::tensor2d::VulkanTensor2D;
use crate::error::Result;

/// Constants from the paper
pub const MASK_INNER_RATIO: f32 = 0.8;  // Inner radius = 0.8 * template_radius
pub const MASK_OUTER_RATIO: f32 = 1.0;  // Outer radius = template_radius
pub const FILTER_PARAMETER_A: f32 = 0.2; // a = 1/5 from Equation 3
pub const REFINEMENT_RADIUS: u32 = 3;    // rs = 3 voxels from Section 3.3

/// Generate a soft mask with cosine tapering between inner and outer radii
///
/// Creates a circular mask that equals 1 within a certain radius around the centre
/// and 0 outside a slightly larger radius, with smooth interpolation between.
pub fn generate_soft_mask(width: u32, height: u32, inner_radius: f32, outer_radius: f32) -> Vec<f32> {
    let mut mask = vec![0.0; (width * height) as usize];
    let centre_x = width as f32 / 2.0;
    let centre_y = height as f32 / 2.0;

    (0..height).flat_map(|y| (0..width).map(move |x| (y, x)))
        .for_each(|(y, x)| {
            let dx = x as f32 - centre_x;
            let dy = y as f32 - centre_y;
            let dist = (dx * dx + dy * dy).sqrt();

            let value = if dist <= inner_radius {
                1.0
            } else if dist >= outer_radius {
                0.0
            } else {
                // Smooth interpolation (cosine taper)
                let t = (dist - inner_radius) / (outer_radius - inner_radius);
                0.5 * (1.0 + (t * std::f32::consts::PI).cos())
            };

            mask[(y * width + x) as usize] = value;
        });

    mask
}

/// Apply separable 1D filter as described in Equation 3 of the paper
///
/// The low-pass filter h has z-transform: 1 + a(z + z⁻¹ - 2) where a = 1/5
/// In spatial domain: [a, 1-2a, a] kernel
fn apply_separable_filter_1d(data: &[f32], width: u32, height: u32, a: f32) -> Vec<f32> {
    let kernel = [a, 1.0 - 2.0 * a, a];
    let mut result = vec![0.0; data.len()];

    // Apply horizontally
    (0..height).flat_map(|y| (0..width).map(move |x| (y, x)))
        .for_each(|(y, x)| {
            let mut sum = 0.0;
            for (k, &kernel_val) in kernel.iter().enumerate().take(3) {
                let src_x = (x as i32 + k as i32 - 1).max(0).min(width as i32 - 1) as usize;
                sum += data[(y * width + src_x as u32) as usize] * kernel_val;
            }
            result[(y * width + x) as usize] = sum;
        });

    // Apply vertically
    let mut final_result = vec![0.0; data.len()];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for (k, &kernel_val) in kernel.iter().enumerate().take(3) {
                let src_y = (y as i32 + k as i32 - 1).max(0).min(height as i32 - 1) as usize;
                sum += result[src_y * width as usize + x as usize] * kernel_val;
            }
            final_result[(y * width + x) as usize] = sum;
        }
    }

    final_result
}

/// Apply the S operator as defined in Equation 3 of the paper
///
/// S operator combines:
/// 1. Low-pass filter h with z-transform: 1 + a(z + z⁻¹ - 2) where a = 1/5
/// 2. Mask m that equals 1 within a certain radius and 0 outside
pub fn apply_s_operator(image: &[f32], mask: &[f32], width: u32, height: u32) -> Vec<f32> {
    // 1. Apply low-pass filter
    let filtered = apply_separable_filter_1d(image, width, height, FILTER_PARAMETER_A);

    // 2. Apply mask
    let result: Vec<f32> = mask.iter()
        .zip(filtered.iter())
        .map(|(&mask_val, &filtered_val)| mask_val * filtered_val)
        .collect();

    result
}

/// Compute 2D FFT of real data
pub fn fft_2d_real(
    data: &[f32],
    width: usize,
    height: usize,
) -> Result<Vec<Complex<f32>>> {
    let mut planner = FftPlanner::new();

    // Convert real data to complex
    let mut complex_data: Vec<Complex<f32>> = data.iter().map(|&x| Complex::new(x, 0.0)).collect();

    // Apply FFT in horizontal direction
    let fft_width = planner.plan_fft_forward(width);
    (0..height)
        .for_each(|row| {
            let row_start = row * width;
            let row_end = row_start + width;
            fft_width.process(&mut complex_data[row_start..row_end]);
        });

    // Transpose for vertical processing
    let mut transposed = vec![Complex::new(0.0, 0.0); width * height];
    (0..height).flat_map(|y| (0..width).map(move |x| (y, x)))
        .for_each(|(y, x)| {
            transposed[x * height + y] = complex_data[y * width + x];
        });

    // Apply FFT in vertical direction
    let fft_height = planner.plan_fft_forward(height);
    (0..width)
        .for_each(|col| {
            let col_start = col * height;
            let col_end = col_start + height;
            fft_height.process(&mut transposed[col_start..col_end]);
        });

    // Transpose back
    let mut result = vec![Complex::new(0.0, 0.0); width * height];
    (0..height).flat_map(|y| (0..width).map(move |x| (y, x)))
        .for_each(|(y, x)| {
            result[y * width + x] = transposed[x * height + y];
        });

    Ok(result)
}

/// Compute 2D inverse FFT
pub fn ifft_2d(
    data: &[Complex<f32>],
    width: usize,
    height: usize,
) -> Result<Vec<f32>> {
    let mut planner = FftPlanner::new();
    let mut complex_data = data.to_vec();

    // Apply inverse FFT in horizontal direction
    let ifft_width = planner.plan_fft_inverse(width);
    (0..height)
        .for_each(|row| {
            let row_start = row * width;
            let row_end = row_start + width;
            ifft_width.process(&mut complex_data[row_start..row_end]);
        });

    // Transpose for vertical processing
    let mut transposed = vec![Complex::new(0.0, 0.0); width * height];
    (0..height).flat_map(|y| (0..width).map(move |x| (y, x)))
        .for_each(|(y, x)| {
            transposed[x * height + y] = complex_data[y * width + x];
        });

    // Apply inverse FFT in vertical direction
    let ifft_height = planner.plan_fft_inverse(height);
    (0..width)
        .for_each(|col| {
            let col_start = col * height;
            let col_end = col_start + height;
            ifft_height.process(&mut transposed[col_start..col_end]);
        });

    // Transpose back and normalise
    let mut result = vec![0.0; width * height];
    let normalisation = (width * height) as f32;
    (0..height).flat_map(|y| (0..width).map(move |x| (y, x)))
        .for_each(|(y, x)| {
            result[y * width + x] = transposed[x * height + y].re / normalisation;
        });

    Ok(result)
}

/// Compute local normalisation factor w(x) as defined in Equation 2 of the paper
///
/// w(x) = 1 / sqrt(<S(τx(f))², 1> - <S(τx(f)), 1>² / M)
///
/// Where:
/// - S is the operator combining low-pass filter and mask
/// - τx(f) is the target image translated by x
/// - M is the sum of mask values
fn compute_local_normalisation(
    target: &[f32],
    width: u32,
    height: u32,
    template_width: u32,
    template_height: u32,
) -> Vec<f32> {
    let mut normalisation_factors = vec![1.0; (width * height) as usize];
    let template_radius = ((template_width.min(template_height) as f32) / 2.0).max(1.0);
    let inner_radius = template_radius * MASK_INNER_RATIO;
    let outer_radius = template_radius * MASK_OUTER_RATIO;
    let mask = generate_soft_mask(template_width, template_height, inner_radius, outer_radius);

    // For each position in the target image
    (0..height).flat_map(|y| (0..width).map(move |x| (y, x)))
        .for_each(|(y, x)| {
            // Create a local patch around this position
            let mut patch = vec![0.0; (template_width * template_height) as usize];

            // Extract patch and apply mask
            for ty in 0..template_height {
                for tx in 0..template_width {
                    let target_x = x.wrapping_add(tx).wrapping_sub(template_width / 2);
                    let target_y = y.wrapping_add(ty).wrapping_sub(template_height / 2);

                    if target_x < width && target_y < height {
                        let target_idx = (target_y * width + target_x) as usize;
                        let mask_idx = (ty * template_width + tx) as usize;
                        let patch_idx = (ty * template_width + tx) as usize;

                        let masked_value = target[target_idx] * mask[mask_idx];
                        patch[patch_idx] = masked_value;
                    }
                }
            }

            // Apply S operator to the patch
            let s_patch = apply_s_operator(&patch, &mask, template_width, template_height);

            // Compute local statistics
            let mut s_sum = 0.0;
            let mut s_sq_sum = 0.0;
            let mut mask_sum = 0.0;
            for (i, &s_val) in s_patch.iter().enumerate() {
                s_sum += s_val * mask[i];
                s_sq_sum += s_val * s_val * mask[i];
                mask_sum += mask[i];
            }

            // Compute normalisation factor
            if mask_sum > 1e-6 {
                let variance = (s_sq_sum - s_sum * s_sum / mask_sum) / mask_sum;
                let std_dev = variance.sqrt().max(1e-6);
                normalisation_factors[(y * width + x) as usize] = 1.0 / std_dev;
            }
        });

    normalisation_factors
}

/// Compute correlation in Fourier domain
///
/// This implements the core performance advantage of the tensorial approach
/// by computing correlations in the Fourier domain rather than real space.
pub fn compute_correlation_fourier(
    target: &[f32],
    target_width: u32,
    target_height: u32,
    template_tensors: &[VulkanTensor2D],
    template_width: u32,
    template_height: u32,
) -> Result<Vec<VulkanTensor2D>> {
    let target_w = target_width as usize;
    let target_h = target_height as usize;
    let template_w = template_width as usize;
    let template_h = template_height as usize;

    // Compute local normalisation factors
    let normalisation_factors = compute_local_normalisation(
        target, target_width, target_height, template_width, template_height
    );

    // Convert target to frequency domain (once)
    let target_fft = fft_2d_real(target, target_w, target_h)?;

    // For each tensor component, compute correlation in Fourier domain
    let mut result_tensors = vec![VulkanTensor2D::zero(); target_w * target_h];

    // Process each of the 5 independent tensor components
    for component_idx in 0..5 {
        // Extract this component from all template tensors
        let component_field: Vec<f32> = template_tensors.iter()
            .map(|t| t.components[component_idx])
            .collect();

        // FFT of template component
        let template_fft = fft_2d_real(&component_field, template_w, template_h)?;

        // Multiply in frequency domain (correlation theorem)
        let mut product = vec![Complex::new(0.0, 0.0); target_w * target_h];

        // For correlation, we need to pad the template FFT to match target size
        // and conjugate it for cross-correlation
        (0..target_h).flat_map(|y| (0..target_w).map(move |x| (y, x)))
            .for_each(|(y, x)| {
                let target_idx = y * target_w + x;

                // Handle padding by only computing where template overlaps
                if x < template_w && y < template_h {
                    let template_idx = y * template_w + x;
                    product[target_idx] = target_fft[target_idx] * template_fft[template_idx].conj();
                }
                // Zero-padding for regions outside template
            });

        // Inverse FFT to get correlation result
        let correlation = ifft_2d(&product, target_w, target_h)?;

        // Store in result tensors
        // Only store in valid output region (target - template + 1)
        let out_width = target_w - template_w + 1;
        let out_height = target_h - template_h + 1;

        (0..out_height).flat_map(|y| (0..out_width).map(move |x| (y, x)))
            .for_each(|(y, x)| {
                let out_idx = y * out_width + x;
                let target_idx = y * target_w + x;
                // Apply local normalisation
                let normalised_value = correlation[target_idx] * normalisation_factors[target_idx];
                result_tensors[out_idx].components[component_idx] = normalised_value;
            });
    }

    Ok(result_tensors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_correlation_simple() {
        // Simple test case
        let target = vec![1.0, 2.0, 3.0, 4.0];
        let template_tensors = vec![VulkanTensor2D::from_rotation(0.0, 1.0)];

        let result = compute_correlation_fourier(
            &target, 2, 2,
            &template_tensors, 1, 1
        );

        assert!(result.is_ok());
    }
}