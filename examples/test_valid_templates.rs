//! Test with VALID templates (extracted from known locations)

use std::fs;
use std::path::Path;
use std::time::Instant;
use vulkan_tensor_matching::{ImageData, VulkanTensorMatcher};

fn main() {
    println!("{:=<60}", "");
    println!("Test with VALID Templates (Known GT)");
    println!("{:=<60}", "");
    
    let test_data = Path::new("test_data");
    let answers_file = test_data.join("valid_answers.jsonl");
    
    let matcher = VulkanTensorMatcher::new().expect("Failed to create matcher");
    
    let mut pass_count = 0;
    let mut total = 0;
    
    for (case_idx, line) in fs::read_to_string(&answers_file).expect("Failed to read").lines().enumerate() {
        let answer: serde_json::Value = serde_json::from_str(line).expect("Failed to parse");
        
        let img_path = test_data.join(answer["image_path"].as_str().unwrap());
        let tmpl_path = test_data.join(answer["template_path"].as_str().unwrap());
        let gt = &answer["expected_matches"][0];
        
        let gt_x = gt["x"].as_u64().unwrap() as u32;
        let gt_y = gt["y"].as_u64().unwrap() as u32;
        let gt_w = gt["w"].as_u64().unwrap() as u32;
        let gt_h = gt["h"].as_u64().unwrap() as u32;
        
        println!("\nCase {}: {}", case_idx, img_path.display());
        println!("  GT: ({}, {}) {}x{}", gt_x, gt_y, gt_w, gt_h);
        
        let img = ImageData::from_file(img_path.to_str().unwrap()).unwrap();
        let template = ImageData::from_file(tmpl_path.to_str().unwrap()).unwrap();
        
        let start = Instant::now();
        let matches = matcher.match_template(&img, &template, 0.3, 1).unwrap();
        let duration = start.elapsed();
        
        let gt_cx = gt_x as f32 + gt_w as f32 / 2.0;
        let gt_cy = gt_y as f32 + gt_h as f32 / 2.0;
        
        total += 1;
        
        if matches.is_empty() {
            println!("  ✗ No matches ({:.2}s)", duration.as_secs_f32());
            continue;
        }
        
        let best = &matches[0];
        let dist = ((best.x as f32 - gt_cx).powi(2) + (best.y as f32 - gt_cy).powi(2)).sqrt();
        let passed = dist < 10.0 && best.correlation > 0.7;
        
        if passed {
            pass_count += 1;
            println!("  ✓ PASS: ({}, {}) corr={:.3} dist={:.1}px ({:.2}s)", 
                     best.x, best.y, best.correlation, dist, duration.as_secs_f32());
        } else {
            println!("  ✗ FAIL: ({}, {}) corr={:.3} dist={:.1}px ({:.2}s)", 
                     best.x, best.y, best.correlation, dist, duration.as_secs_f32());
        }
    }
    
    println!("\n{:=<60}", "");
    println!("SUMMARY: {}/{} passed ({:.0}%)", pass_count, total, 100.0 * pass_count as f32 / total as f32);
    println!("{:=<60}", "");
    
    if pass_count == total {
        println!("✓✓✓ ALL TESTS PASSED ✓✓✓");
    }
}
