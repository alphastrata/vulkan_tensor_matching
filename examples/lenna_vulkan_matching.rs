//! Example demonstrating pure Vulkan-accelerated template matching
//!
//! This implements the basic GPU-accelerated approach without tensor mathematics
//! for comparison with the full tensorial approach.

use vulkan_tensor_matching::{ImageData, image_data_to_rgb_image};
use vulkan_tensor_matching::image::{annotate_image_with_matches, save_debug_output};
use vulkan_tensor_matching::image::matcher::VulkanTensorMatcher;
use image::RgbImage;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the target image (lenna.png) and templates
    let target_path = "test_assets/lenna.png";
    let template_path1 = "test_assets/templates/test1.png";
    let template_path2 = "test_assets/templates/test2.png";

    let target_image = ImageData::from_file(target_path)?;
    let template1 = ImageData::from_file(template_path1)?;
    let template2 = ImageData::from_file(template_path2)?;

    // Create Vulkan matcher (basic implementation)
    match VulkanTensorMatcher::new() {
        Ok(matcher) => {
            let start_time = Instant::now();

            // Perform template matching for the first template
            let matches1 = matcher.match_template(
                &target_image,
                &template1,
                0.8, // correlation threshold
                10   // max matches
            )?;

            // Perform template matching for the second template
            let matches2 = matcher.match_template(
                &target_image,
                &template2,
                0.8, // correlation threshold
                10   // max matches
            )?;

            let elapsed_time = start_time.elapsed();

            // Annotate image with matches
            let mut annotated_image: RgbImage = image_data_to_rgb_image(&target_image);
            annotate_image_with_matches(
                &mut annotated_image,
                &matches1,
                template1.width,
                template1.height
            )?;
            annotate_image_with_matches(
                &mut annotated_image,
                &matches2,
                template2.width,
                template2.height
            )?;

            // Save annotated image with debug output
            let output_path = "vulkan_multi_matches_annotated.png";
            let total_matches = matches1.len() + matches2.len();

            save_debug_output(
                &annotated_image,
                output_path,
                total_matches,
                elapsed_time,
                "Pure Vulkan Multi-Template Matching",
                None  // Use default config
            )?;
        }
        Err(_e) => {
            // This is expected in environments without proper Vulkan setup.
            // No fallback needed as there is no operation to perform without Vulkan
        }
    }

    Ok(())
}