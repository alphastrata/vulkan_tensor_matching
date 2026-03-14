//! Test with ORIGINAL templates from answers.jsonl

use std::fs;
use std::path::Path;
use std::time::Instant;

use vulkan_tensor_matching::{ImageData, VulkanTensorMatcher};

fn main() {
    println!("{:=<60}", "");
    println!("Test with ORIGINAL Templates");
    println!("{:=<60}", "");

    let test_data_dir = Path::new("test_data");
    let answers_file = test_data_dir.join("extensive/answers.jsonl");

    let matcher = VulkanTensorMatcher::new().expect("Failed to create matcher");

    let mut results = Vec::new();

    for (case_idx, line) in fs::read_to_string(&answers_file)
        .expect("Failed to read answers")
        .lines()
        .take(10)
        .enumerate()
    {
        let answer: serde_json::Value = serde_json::from_str(line).expect("Failed to parse JSON");

        let img_path = test_data_dir.join(answer["image_path"].as_str().unwrap());
        let tmpl_path = test_data_dir.join(answer["template_path"].as_str().unwrap());
        let gt = &answer["expected_matches"][0];

        let gt_x = gt["x"].as_u64().unwrap() as u32;
        let gt_y = gt["y"].as_u64().unwrap() as u32;
        let gt_w = gt["w"].as_u64().unwrap() as u32;
        let gt_h = gt["h"].as_u64().unwrap() as u32;
        let gt_angle = gt["angle"].as_f64().unwrap_or(0.0);

        println!("\nCase {}: {}", case_idx, img_path.display());
        println!("  Template: {} ({}x{})", tmpl_path.display(), gt_w, gt_h);
        println!(
            "  GT: ({}, {}) angle={:.1}°",
            gt_x,
            gt_y,
            gt_angle.to_degrees()
        );

        if !img_path.exists() {
            println!("  SKIP: Image not found");
            continue;
        }
        if !tmpl_path.exists() {
            println!("  SKIP: Template not found");
            continue;
        }

        let img = ImageData::from_file(img_path.to_str().unwrap()).expect("Failed to load image");
        let template =
            ImageData::from_file(tmpl_path.to_str().unwrap()).expect("Failed to load template");

        println!("  Image: {}x{}", img.width, img.height);
        println!("  Template: {}x{}", template.width, template.height);

        let start = Instant::now();
        let matches = matcher
            .match_template(&img, &template, 0.3, 3)
            .expect("Matching failed");
        let duration = start.elapsed();

        let gt_cx = gt_x as f32 + gt_w as f32 / 2.0;
        let gt_cy = gt_y as f32 + gt_h as f32 / 2.0;

        if matches.is_empty() {
            println!("  ✗ No matches found ({:.1}s)", duration.as_secs_f32());
            results.push((case_idx, false, 999.0, 0.0));
            continue;
        }

        let best = &matches[0];
        let dx = best.x as f32 - gt_cx;
        let dy = best.y as f32 - gt_cy;
        let dist = (dx * dx + dy * dy).sqrt();

        println!(
            "  Match: ({}, {}) corr={:.3} dist={:.1}px ({:.1}s)",
            best.x,
            best.y,
            best.correlation,
            dist,
            duration.as_secs_f32()
        );

        let success = dist < 15.0 && best.correlation > 0.5;
        if success {
            println!("  ✓ PASS");
        } else {
            println!("  ✗ FAIL");
        }

        results.push((case_idx, success, dist, best.correlation));
    }

    println!("\n{:=<60}", "");
    println!("SUMMARY");
    println!("{:=<60}", "");

    let pass_count = results.iter().filter(|(_, success, _, _)| *success).count();
    println!("Passed: {}/{}", pass_count, results.len());

    for (idx, success, dist, corr) in &results {
        let status = if *success { "✓" } else { "✗" };
        println!(
            "  Case {}: {} dist={:.1}px corr={:.3}",
            idx, status, dist, corr
        );
    }
}
