//! Integration tests for template matching using generated test data.
//!
//! Test data format:
//! - test_data/<test_name>/source.png - The source image
//! - test_data/<test_name>/template.png - The template to match
//! - test_data/<test_name>/expected.txt - Expected matches (x,y,threshold per line)
//!
//! These tests validate the CPU template matching implementation against known ground truth.

use vulkan_tensor_matching::image::{
    loader::ImageData,
    loader::MatchTemplateMethod,
};
use std::fs;
use std::path::Path;

/// Parse expected matches from expected.txt file
fn parse_expected(path: &Path) -> Vec<(u32, u32, f32)> {
    let content = fs::read_to_string(path).expect("Failed to read expected.txt");
    content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let parts: Vec<&str> = line.split(',').collect();
            let x = parts[0].parse::<u32>().expect("Invalid x coordinate");
            let y = parts[1].parse::<u32>().expect("Invalid y coordinate");
            let threshold = parts[2].parse::<f32>().expect("Invalid threshold");
            (x, y, threshold)
        })
        .collect()
}

/// Find all peaks above a threshold, masking out found peaks to find subsequent ones
fn find_all_peaks(result: &ImageData, min_corr: f32, max_peaks: usize) -> Vec<(f32, (u32, u32))> {
    let mut peaks = Vec::new();
    let mut data = result.data.clone();
    
    for _ in 0..max_peaks {
        // Find max in current data
        let ((max_val, max_pos), _) = find_extremes_in_slice(&data, result.width);
        
        if max_val < min_corr {
            break;
        }
        
        peaks.push((max_val, max_pos));
        
        // Mask out area around this peak to find next one
        let mask_radius = 10;
        for dy in -mask_radius..=mask_radius {
            for dx in -mask_radius..=mask_radius {
                let nx = max_pos.0 as i32 + dx;
                let ny = max_pos.1 as i32 + dy;
                if nx >= 0 && ny >= 0 && nx < result.width as i32 && ny < result.height as i32 {
                    let idx = (ny as u32 * result.width + nx as u32) as usize;
                    if idx < data.len() {
                        data[idx] = 0.0;
                    }
                }
            }
        }
    }
    
    peaks
}

fn find_extremes_in_slice(data: &[f32], width: u32) -> ((f32, (u32, u32)), (f32, (u32, u32))) {
    let mut max_val = f32::NEG_INFINITY;
    let mut min_val = f32::INFINITY;
    let mut max_pos = (0u32, 0u32);
    let mut min_pos = (0u32, 0u32);

    for (i, &value) in data.iter().enumerate() {
        let y = (i as u32) / width;
        let x = (i as u32) % width;

        if value > max_val {
            max_val = value;
            max_pos = (x, y);
        }
        if value < min_val {
            min_val = value;
            min_pos = (x, y);
        }
    }

    ((max_val, max_pos), (min_val, min_pos))
}

#[test]
fn test_t1_single_match() {
    test_template_match("t1_single_match", 5, 1);
}

#[test]
fn test_t2_multi_match() {
    test_template_match("t2_multi_match", 10, 3);
}

#[test]
fn test_t3_corner_match() {
    test_template_match("t3_corner_match", 5, 2);
}

fn test_template_match(test_name: &str, tolerance: u32, expected_count: usize) {
    let test_dir = Path::new("test_data").join(test_name);
    
    let source_path = test_dir.join("source.png");
    let template_path = test_dir.join("template.png");
    let expected_path = test_dir.join("expected.txt");
    
    assert!(source_path.exists(), "Source image not found: {:?}", source_path);
    assert!(template_path.exists(), "Template image not found: {:?}", template_path);
    assert!(expected_path.exists(), "Expected file not found: {:?}", expected_path);
    
    // Load images
    let source = ImageData::from_file(&source_path)
        .expect("Failed to load source image");
    let template = ImageData::from_file(&template_path)
        .expect("Failed to load template image");
    
    // Parse expected matches
    let expected = parse_expected(&expected_path);
    assert!(!expected.is_empty(), "No expected matches defined");
    
    println!("Test '{}': Image {}x{}, Template {}x{}, Expected {} matches", 
             test_name, source.width, source.height, template.width, template.height, expected.len());
    
    // Run CPU template matching with cross-correlation
    let result = source.match_template(&template, MatchTemplateMethod::CrossCorrelation);
    
    // Find all peaks above threshold
    let peaks = find_all_peaks(&result, 50.0, expected.len() + 2);
    
    println!("  Found {} peaks above threshold", peaks.len());
    for (i, (corr, pos)) in peaks.iter().enumerate() {
        println!("    Peak {}: ({}, {}) corr={:.4}", i+1, pos.0, pos.1, corr);
    }
    
    // Verify we found at least the expected number of matches
    assert!(
        peaks.len() >= expected.len(),
        "Expected at least {} matches, found {}",
        expected.len(),
        peaks.len()
    );
    
    // Verify each expected location has a match nearby
    for (exp_x, exp_y, _threshold) in &expected {
        let expected_tl_x = exp_x - template.width / 2;
        let expected_tl_y = exp_y - template.height / 2;
        
        let found = peaks.iter().any(|(_, pos)| {
            let dx = (pos.0 as i32 - expected_tl_x as i32).abs() as u32;
            let dy = (pos.1 as i32 - expected_tl_y as i32).abs() as u32;
            dx <= tolerance && dy <= tolerance
        });
        
        assert!(
            found,
            "No match found near expected ({}, {}) [tolerance={}]",
            expected_tl_x, expected_tl_y, tolerance
        );
    }
    
    println!("  ✓ Test passed!");
}
