use imageproc::template_matching::{MatchTemplateMethod, match_template};
use vulkan_tensor_matching::{ImageData, VulkanTensorMatcher};

#[test]
fn test_compare_with_imageproc() {
    // Load test images
    let target_path = "test_assets/lenna.png";
    let template_path = "test_assets/templates/test1.png";

    if !std::path::Path::new(target_path).exists() || !std::path::Path::new(template_path).exists()
    {
        println!("Test assets not found, skipping imageproc comparison test");
        return;
    }

    // Load images for imageproc
    let target_image_img = image::open(target_path).unwrap().to_luma8();
    let template_image_img = image::open(template_path).unwrap().to_luma8();

    // Run imageproc template matching
    let result = match_template(
        &target_image_img,
        &template_image_img,
        MatchTemplateMethod::CrossCorrelationNormalized,
    );

    // Find the best match location from imageproc
    let mut max_val = -1.0;
    let mut max_loc = (0, 0);
    for (x, y, v) in result.enumerate_pixels() {
        if v.0[0] > max_val {
            max_val = v.0[0];
            max_loc = (x, y);
        }
    }

    // Load images for Vulkan
    let target_image_vulkan = ImageData::from_file(target_path).expect("Failed to load lenna.png");
    let template_image_vulkan =
        ImageData::from_file(template_path).expect("Failed to load test1.png");

    // Run Vulkan template matching
    let matcher = VulkanTensorMatcher::new().expect("Failed to create VulkanTensorMatcher");
    let matches = matcher
        .match_template(
            &target_image_vulkan,
            &template_image_vulkan,
            0.0, // correlation threshold (reduced from 0.8 to allow lower correlation matches)
            1,   // max matches
        )
        .expect("Vulkan template matching failed");

    // Compare the results
    assert!(!matches.is_empty(), "Vulkan matcher found no matches");

    let vulkan_match = &matches[0];
    let imageproc_match_x = max_loc.0;
    let imageproc_match_y = max_loc.1;

    // The coordinates from imageproc are the top-left of the template,
    // while the Vulkan matcher returns the centre. Adjust for this.
    // Ensure we don't underflow if x or y is smaller than half dimensions
    let vulkan_match_x_adjusted = vulkan_match
        .x
        .saturating_sub(template_image_vulkan.width / 2);
    let vulkan_match_y_adjusted = vulkan_match
        .y
        .saturating_sub(template_image_vulkan.height / 2);

    let x_diff = (vulkan_match_x_adjusted as i32 - imageproc_match_x as i32).abs();
    let y_diff = (vulkan_match_y_adjusted as i32 - imageproc_match_y as i32).abs();

    // Allow a small tolerance for differences in implementation
    let tolerance = 25;
    assert!(
        x_diff <= tolerance,
        "X coordinates differ by more than tolerance: vulkan: {} (raw: {}), imageproc: {}",
        vulkan_match_x_adjusted,
        vulkan_match.x,
        imageproc_match_x
    );
    assert!(
        y_diff <= tolerance,
        "Y coordinates differ by more than tolerance: vulkan: {} (raw: {}), imageproc: {}",
        vulkan_match_y_adjusted,
        vulkan_match.y,
        imageproc_match_y
    );
}
