use crate::error::Result;
use image::{DynamicImage, GenericImageView};
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub enum TestShape {
    Square,
    Circle,
    Cross,
}

#[derive(Debug, Clone)]
pub struct ImageData {
    pub data: Vec<f32>,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
}

impl ImageData {
    /// Create new ImageData from raw data
    pub fn new(data: Vec<f32>, width: u32, height: u32, channels: u32) -> Self {
        Self {
            data,
            width,
            height,
            channels,
        }
    }

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
            DynamicImage::ImageLumaA8(img) => {
                let data = img.pixels().map(|pixel| pixel[0] as f32 / 255.0).collect();

                Ok(Self {
                    data,
                    width,
                    height,
                    channels: 1,
                })
            }
            _ => {
                // Fallback: convert to grayscale (Luma8)
                let luma8 = img.to_luma8();
                let data = luma8
                    .pixels()
                    .map(|pixel| pixel[0] as f32 / 255.0)
                    .collect();

                Ok(Self {
                    data,
                    width,
                    height,
                    channels: 1,
                })
            }
        }
    }

    /// Extract a region from the image
    pub fn extract_region(&self, x: u32, y: u32, width: u32, height: u32) -> Result<Self> {
        if x + width > self.width || y + height > self.height {
            return Err(crate::error::TensorMatchingError::Other(
                "Region out of bounds".to_string(),
            ));
        }

        let mut data = Vec::with_capacity((width * height) as usize);
        for ry in 0..height {
            for rx in 0..width {
                let idx = ((y + ry) * self.width + (x + rx)) as usize;
                data.push(self.data[idx]);
            }
        }

        Ok(Self {
            data,
            width,
            height,
            channels: 1,
        })
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
}
