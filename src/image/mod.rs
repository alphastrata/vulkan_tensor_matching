pub mod loader;
pub mod tensor_matcher;

use ab_glyph::{FontRef, PxScale};
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use std::path::Path;


pub struct DebugColors;
impl DebugColors {
    
    pub const STRONG: Rgb<u8> = Rgb([0, 255, 0]);
    
    pub const GOOD: Rgb<u8> = Rgb([255, 255, 0]);
    
    pub const FAIR: Rgb<u8> = Rgb([255, 165, 0]);
    
    pub const WEAK: Rgb<u8> = Rgb([255, 0, 0]);
    
    pub const BACKGROUND: Rgb<u8> = Rgb([255, 255, 255]);
    
    pub const TEXT: Rgb<u8> = Rgb([0, 0, 0]);
}


pub fn annotate_image_with_tensor_matches(
    image: &mut RgbImage,
    matches: &[TensorTemplateMatch],
    template_width: u32,
    template_height: u32,
) -> crate::error::Result<()> {
    
    const FONT_DATA: &[u8] = include_bytes!("../DejaVuSans.ttf");
    let font = FontRef::try_from_slice(FONT_DATA)
        .map_err(|_| crate::error::TensorMatchingError::Other("Failed to load font".to_string()))?;
    let scale = PxScale::from(16.0);

    for (index, m) in matches.iter().enumerate() {
        
        let colour = if m.correlation > 0.9 {
            DebugColors::STRONG
        } else if m.correlation > 0.8 {
            DebugColors::GOOD
        } else if m.correlation > 0.7 {
            DebugColors::FAIR
        } else {
            DebugColors::WEAK
        };

        
        let half_width = if template_width > 0 {
            template_width / 2
        } else {
            15
        };
        let half_height = if template_height > 0 {
            template_height / 2
        } else {
            15
        };
        let x1 = (m.x.saturating_sub(half_width)) as i32;
        let y1 = (m.y.saturating_sub(half_height)) as i32;
        let x2 = (m.x + half_width) as i32;
        let y2 = (m.y + half_height) as i32;

        
        let rect = Rect::at(x1, y1).of_size((x2 - x1) as u32, (y2 - y1) as u32);
        draw_hollow_rect_mut(image, rect, colour);

        
        let centre_x = m.x as i32;
        let centre_y = m.y as i32;

        
        let cross_size = 8.min(image.width().min(image.height()) as i32 / 4); 
        for dx in -cross_size..=cross_size {
            let x = centre_x + dx;
            if x >= 0 && x < image.width() as i32 {
                image.put_pixel(x as u32, centre_y as u32, colour);
            }
        }
        for dy in -cross_size..=cross_size {
            let y = centre_y + dy;
            if y >= 0 && y < image.height() as i32 {
                image.put_pixel(centre_x as u32, y as u32, colour);
            }
        }

        
        let text = format!(
            "T{}: ({}, {}) corr: {:.2} @ {:.1}°",
            index + 1,
            m.x,
            m.y,
            m.correlation,
            m.rotation_angle.to_degrees()
        );

        
        draw_text_mut(
            image,
            DebugColors::TEXT,
            (x1 - 10).max(0), 
            (y1 - 25).max(0), 
            scale,
            &font,
            &text,
        );

        log::debug!(
            "Tensor match annotation: ({}, {}) correlation: {:.3}, rotation: {:.1}°",
            m.x,
            m.y,
            m.correlation,
            m.rotation_angle.to_degrees()
        );
    }

    Ok(())
}


#[derive(Debug, Clone)]
pub struct DebugOutputConfig {
    
    pub enabled: bool,
    
    pub output_dir: Option<String>,
}

impl Default for DebugOutputConfig {
    fn default() -> Self {
        Self {
            enabled: true, 
            output_dir: None,
        }
    }
}


pub fn save_debug_output<P: AsRef<Path>>(
    image: &RgbImage,
    filename: P,
    matches_count: usize,
    processing_time: std::time::Duration,
    methodology: &str,
    config: Option<&DebugOutputConfig>,
) -> crate::error::Result<()> {
    let default_config = DebugOutputConfig::default();
    let config = config.unwrap_or(&default_config);

    if config.enabled {
        let path = if let Some(ref dir) = config.output_dir {
            std::path::Path::new(dir).join(filename.as_ref())
        } else {
            filename.as_ref().to_path_buf()
        };

        image.save(&path)?;
        log::info!(
            "Debug output saved: {} with {} matches. Processing time: {:.3}ms, Method: {}",
            path.display(),
            matches_count,
            processing_time.as_millis() as f64 + processing_time.subsec_nanos() as f64 * 1e-6,
            methodology
        );
    }

    Ok(())
}



pub fn save_debug_output_if_enabled<P: AsRef<Path>>(
    enabled: bool,
    image: &RgbImage,
    filename: P,
    matches_count: usize,
    processing_time: std::time::Duration,
    methodology: &str,
) -> crate::error::Result<()> {
    let config = DebugOutputConfig {
        enabled,
        output_dir: None,
    };
    save_debug_output(
        image,
        filename,
        matches_count,
        processing_time,
        methodology,
        Some(&config),
    )
}


pub fn image_data_to_rgb_image(image_data: &ImageData) -> RgbImage {
    
    let mut rgb_image = RgbImage::new(image_data.width, image_data.height);

    
    for (i, pixel) in rgb_image.pixels_mut().enumerate() {
        
        let gray_value = (image_data.data[i].clamp(0.0, 1.0) * 255.0) as u8;
        
        *pixel = image::Rgb([gray_value, gray_value, gray_value]);
    }

    rgb_image
}

pub use loader::{ImageData, TestShape};
pub use tensor_matcher::{TensorTemplateMatch, VulkanTensorMatcher};
