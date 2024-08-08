use crate::error::Result;
use image::{DynamicImage, GenericImageView, ImageBuffer};
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub enum TestShape {
    Square,
    Circle,
    Cross,
}

#[derive(Debug, Clone, Copy)]
pub enum MatchTemplateMethod {
    /// Sum of the squares of the difference between image and template pixel intensities. Smaller values indicate a better match.
    SumOfSquaredErrors,
    /// Sum of the squares of the difference between image and template pixel intensities, normalised to a 0-1 range.
    SumOfSquaredErrorsNormalized,
    /// Cross-correlation of the template and image.
    CrossCorrelation,
    /// Cross-correlation of the template and image, normalised to a 0-1 range.
    CrossCorrelationNormalized,
}

/// Represents an extreme value (min/max) and its location (value, (x, y))
pub type Extreme = (f32, (u32, u32));

#[derive(Debug, Clone)]
pub struct ImageData {
    pub data: Vec<f32>,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
}

impl ImageData {
    /// Load image from file and convert to normalised float data
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let img = image::open(path)?;
        Self::from_dynamic_image(img)
    }

    /// Convert DynamicImage to normalised float data
    pub fn from_dynamic_image(img: DynamicImage) -> Result<Self> {
        let (width, height) = img.dimensions();

        match img {
            DynamicImage::ImageRgb8(img) => {
                let data = img
                    .pixels()
                    .flat_map(|pixel| {
                        // Convert to grayscale and normalise to [0,1]
                        let gray = (0.299 * pixel[0] as f32
                            + 0.587 * pixel[1] as f32
                            + 0.114 * pixel[2] as f32)
                            / 255.0;
                        std::iter::once(gray)
                    })
                    .collect();

                Ok(Self {
                    data,
                    width,
                    height,
                    channels: 1,
                })
            }
            DynamicImage::ImageRgba8(img) => {
                let data = img
                    .pixels()
                    .flat_map(|pixel| {
                        // Convert RGBA to grayscale, ignore alpha
                        let gray = (0.299 * pixel[0] as f32
                            + 0.587 * pixel[1] as f32
                            + 0.114 * pixel[2] as f32)
                            / 255.0;
                        std::iter::once(gray)
                    })
                    .collect();

                Ok(Self {
                    data,
                    width,
                    height,
                    channels: 1,
                })
            }
            DynamicImage::ImageLuma8(img) => {
                let data = img.pixels().map(|pixel| pixel[0] as f32 / 255.0).collect();

                Ok(Self {
                    data,
                    width,
                    height,
                    channels: 1,
                })
            }
            _ => {
                // Convert any other format to RGB8 first
                let rgb_img = img.to_rgb8();
                let img = DynamicImage::ImageRgb8(rgb_img);
                Self::from_dynamic_image(img)
            }
        }
    }

    /// Create a synthetic test template (useful for benchmarking)
    pub fn create_test_template(size: u32, shape: TestShape) -> Self {
        let mut data = vec![0.0; (size * size) as usize];
        let centre = size as f32 / 2.0;

        match shape {
            TestShape::Square => {
                let half_size = size as f32 * 0.3;
                for y in 0..size {
                    for x in 0..size {
                        let dx = x as f32 - centre;
                        let dy = y as f32 - centre;

                        if dx.abs() <= half_size && dy.abs() <= half_size {
                            data[(y * size + x) as usize] = 1.0;
                        }
                    }
                }
            }
            TestShape::Circle => {
                let radius = size as f32 * 0.3;
                for y in 0..size {
                    for x in 0..size {
                        let dx = x as f32 - centre;
                        let dy = y as f32 - centre;
                        let distance = (dx * dx + dy * dy).sqrt();

                        if distance <= radius {
                            data[(y * size + x) as usize] = 1.0;
                        }
                    }
                }
            }
            TestShape::Cross => {
                let thickness = size / 10;
                let arm_length = size / 3;

                for y in 0..size {
                    for x in 0..size {
                        let dx = (x as i32 - centre as i32).unsigned_abs();
                        let dy = (y as i32 - centre as i32).unsigned_abs();

                        if (dx <= arm_length && dy <= thickness)
                            || (dy <= arm_length && dx <= thickness)
                        {
                            data[(y * size + x) as usize] = 1.0;
                        }
                    }
                }
            }
        }

        Self {
            data,
            width: size,
            height: size,
            channels: 1,
        }
    }

    /// Apply preprocessing (contrast enhancement, noise reduction)
    pub fn preprocess(&mut self) -> Result<()> {
        // Normalise intensity range
        let min_val = self.data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        if max_val > min_val {
            let range = max_val - min_val;
            for pixel in &mut self.data {
                *pixel = (*pixel - min_val) / range;
            }
        }

        // Apply slight Gaussian blur to reduce noise
        self.gaussian_blur(0.5)?;

        Ok(())
    }

    /// Simple Gaussian blur implementation
    fn gaussian_blur(&mut self, sigma: f32) -> Result<()> {
        if sigma <= 0.0 {
            return Ok(());
        }

        let kernel_size = ((sigma * 6.0) as usize).max(3) | 1; // Ensure odd
        let kernel = Self::generate_gaussian_kernel(kernel_size, sigma);

        let original_data = self.data.clone();

        // Horizontal pass
        (0..self.height).flat_map(|y| (0..self.width).map(move |x| (y, x)))
            .for_each(|(y, x)| {
                let mut sum = 0.0;
                let mut weight_sum = 0.0;

                for (i, &weight) in kernel.iter().enumerate() {
                    let offset = i as i32 - kernel_size as i32 / 2;
                    let sample_x = (x as i32 + offset).max(0).min(self.width as i32 - 1) as usize;

                    sum += original_data[y as usize * self.width as usize + sample_x] * weight;
                    weight_sum += weight;
                }

                self.data[y as usize * self.width as usize + x as usize] = sum / weight_sum;
            });

        let horizontal_result = self.data.clone();

        // Vertical pass
        (0..self.height).flat_map(|y| (0..self.width).map(move |x| (y, x)))
            .for_each(|(y, x)| {
                let mut sum = 0.0;
                let mut weight_sum = 0.0;

                for (i, &weight) in kernel.iter().enumerate() {
                    let offset = i as i32 - kernel_size as i32 / 2;
                    let sample_y = (y as i32 + offset).max(0).min(self.height as i32 - 1) as usize;

                    sum += horizontal_result[sample_y * self.width as usize + x as usize] * weight;
                    weight_sum += weight;
                }

                self.data[y as usize * self.width as usize + x as usize] = sum / weight_sum;
            });

        Ok(())
    }

    fn generate_gaussian_kernel(size: usize, sigma: f32) -> Vec<f32> {
        let mut kernel = vec![0.0; size];
        let centre = size / 2;
        let variance = sigma * sigma;

        for (i, value) in kernel.iter_mut().enumerate().take(size) {
            let x = i as i32 - centre as i32;
            *value = (-(x * x) as f32 / (2.0 * variance)).exp();
        }

        kernel
    }

    /// Save processed image to file (for debugging)
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut img_buffer = ImageBuffer::new(self.width, self.height);

        for (x, y, pixel) in img_buffer.enumerate_pixels_mut() {
            let intensity = self.data[(y * self.width + x) as usize];
            let byte_val = (intensity * 255.0).round().clamp(0.0, 255.0) as u8;
            *pixel = image::Luma([byte_val]);
        }

        img_buffer.save(path)?;
        Ok(())
    }

    /// Finds the largest and smallest values in an image and their locations.
    /// If there are multiple such values then the lexicographically smallest is returned.
    pub fn find_extremes(&self) -> (Extreme, Extreme) {
        let mut max_val = f32::NEG_INFINITY;
        let mut min_val = f32::INFINITY;
        let mut max_pos = (0u32, 0u32);
        let mut min_pos = (0u32, 0u32);

        for (i, &value) in self.data.iter().enumerate() {
            let y = (i as u32) / self.width;
            let x = (i as u32) % self.width;

            if value > max_val
                || (value == max_val && (y < max_pos.0 || (y == max_pos.0 && x < max_pos.1)))
            {
                max_val = value;
                max_pos = (x, y);
            }

            if value < min_val
                || (value == min_val && (y < min_pos.0 || (y == min_pos.0 && x < min_pos.1)))
            {
                min_val = value;
                min_pos = (x, y);
            }
        }

        ((max_val, max_pos), (min_val, min_pos))
    }

    /// Slides a template over an image and scores the match at each point using the requested method.
    pub fn match_template(&self, template: &ImageData, method: MatchTemplateMethod) -> ImageData {
        if template.width > self.width || template.height > self.height {
            panic!("Template dimensions must be less than or equal to image dimensions");
        }

        let out_width = self.width - template.width + 1;
        let out_height = self.height - template.height + 1;
        let mut result_data = vec![0.0f32; (out_width * out_height) as usize];

        for ty in 0..out_height {
            for tx in 0..out_width {
                let score = self.score_at(tx, ty, template, method);
                let idx = (ty * out_width + tx) as usize;
                result_data[idx] = score;
            }
        }

        ImageData {
            data: result_data,
            width: out_width,
            height: out_height,
            channels: 1,
        }
    }

    /// Helper function to calculate score at a specific position
    fn score_at(&self, x: u32, y: u32, template: &ImageData, method: MatchTemplateMethod) -> f32 {
        match method {
            MatchTemplateMethod::SumOfSquaredErrors => {
                let mut score = 0.0f32;
                for ty in 0..template.height {
                    for tx in 0..template.width {
                        let img_idx = ((y + ty) * self.width + (x + tx)) as usize;
                        let templ_idx = (ty * template.width + tx) as usize;

                        let diff = self.data[img_idx] - template.data[templ_idx];
                        score += diff * diff;
                    }
                }
                score
            }
            MatchTemplateMethod::SumOfSquaredErrorsNormalized => {
                let mut score = 0.0f32;
                let mut img_sum = 0.0f32;
                let mut templ_sum = 0.0f32;

                for ty in 0..template.height {
                    for tx in 0..template.width {
                        let img_idx = ((y + ty) * self.width + (x + tx)) as usize;
                        let templ_idx = (ty * template.width + tx) as usize;

                        img_sum += self.data[img_idx];
                        templ_sum += template.data[templ_idx];
                    }
                }

                let img_mean = img_sum / (template.width * template.height) as f32;
                let templ_mean = templ_sum / (template.width * template.height) as f32;

                for ty in 0..template.height {
                    for tx in 0..template.width {
                        let img_idx = ((y + ty) * self.width + (x + tx)) as usize;
                        let templ_idx = (ty * template.width + tx) as usize;

                        let img_diff = self.data[img_idx] - img_mean;
                        let templ_diff = template.data[templ_idx] - templ_mean;
                        let diff = img_diff - templ_diff;
                        score += diff * diff;
                    }
                }

                // Normalise to 0-1 range
                score / ((template.width * template.height) as f32)
            }
            MatchTemplateMethod::CrossCorrelation => {
                let mut score = 0.0f32;
                for ty in 0..template.height {
                    for tx in 0..template.width {
                        let img_idx = ((y + ty) * self.width + (x + tx)) as usize;
                        let templ_idx = (ty * template.width + tx) as usize;

                        score += self.data[img_idx] * template.data[templ_idx];
                    }
                }
                score
            }
            MatchTemplateMethod::CrossCorrelationNormalized => {
                let mut img_sum = 0.0f32;
                let mut templ_sum = 0.0f32;
                let mut img_sum_sq = 0.0f32;
                let mut templ_sum_sq = 0.0f32;
                let mut prod_sum = 0.0f32;

                for ty in 0..template.height {
                    for tx in 0..template.width {
                        let img_idx = ((y + ty) * self.width + (x + tx)) as usize;
                        let templ_idx = (ty * template.width + tx) as usize;

                        let img_val = self.data[img_idx];
                        let templ_val = template.data[templ_idx];

                        img_sum += img_val;
                        templ_sum += templ_val;
                        img_sum_sq += img_val * img_val;
                        templ_sum_sq += templ_val * templ_val;
                        prod_sum += img_val * templ_val;
                    }
                }

                let n = (template.width * template.height) as f32;
                let img_mean = img_sum / n;
                let templ_mean = templ_sum / n;

                let img_var = img_sum_sq / n - img_mean * img_mean;
                let templ_var = templ_sum_sq / n - templ_mean * templ_mean;

                let numerator = prod_sum / n - img_mean * templ_mean;
                let denominator = (img_var * templ_var).sqrt();

                if denominator > 1e-6 {
                    numerator / denominator
                } else {
                    0.0
                }
            }
        }
    }

    /// Compresses the image by reducing its size, averaging pixels in blocks
    pub fn compress(&self, factor: u32) -> ImageData {
        if factor == 1 {
            return self.clone();
        }

        let new_width = self.width / factor;
        let new_height = self.height / factor;
        let mut compressed_data = vec![0.0f32; (new_width * new_height) as usize];

        for ny in 0..new_height {
            for nx in 0..new_width {
                let mut sum = 0.0f32;
                let mut count = 0u32;

                for fy in 0..factor {
                    for fx in 0..factor {
                        let orig_x = nx * factor + fx;
                        let orig_y = ny * factor + fy;

                        if orig_x < self.width && orig_y < self.height {
                            let orig_idx = (orig_y * self.width + orig_x) as usize;
                            sum += self.data[orig_idx];
                            count += 1;
                        }
                    }
                }

                let avg = if count > 0 { sum / count as f32 } else { 0.0 };
                let new_idx = (ny * new_width + nx) as usize;
                compressed_data[new_idx] = avg;
            }
        }

        ImageData {
            data: compressed_data,
            width: new_width,
            height: new_height,
            channels: 1,
        }
    }
}

// Public functions that mirror imageproc functionality
pub fn find_extremes(image: &ImageData) -> (Extreme, Extreme) {
    image.find_extremes()
}

pub fn match_template(
    image: &ImageData,
    template: &ImageData,
    method: MatchTemplateMethod,
) -> ImageData {
    image.match_template(template, method)
}

pub fn compress(image: &ImageData, factor: u32) -> ImageData {
    image.compress(factor)
}
