//! Generate PROOF.md using ORIGINAL templates, verifying GT is correct

use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use image::{ImageBuffer, Rgb};
use vulkan_tensor_matching::ImageData;

const DISTANCE_THRESHOLD: f32 = 15.0;
const CORR_THRESHOLD: f32 = 0.5;

struct CaseResult {
    case_idx: usize,
    image_path: String,
    template_path: String,
    gt_x: u32,
    gt_y: u32,
    gt_w: u32,
    gt_h: u32,
    gt_angle: f64,
    vulkan_x: u32,
    vulkan_y: u32,
    vulkan_corr: f32,
    vulkan_rot: f32,
    distance: f32,
    duration_s: f32,
    gt_valid: bool,  // Does GT location actually match template?
    passed: bool,
}

fn extract_patch(img: &ImageData, x: u32, y: u32, w: u32, h: u32) -> Vec<f32> {
    let mut patch = Vec::with_capacity((w * h) as usize);
    for row in 0..h {
        for col in 0..w {
            let px = x + col;
            let py = y + row;
            if py < img.height && px < img.width {
                patch.push(img.data[(py * img.width + px) as usize]);
            } else {
                patch.push(0.0);
            }
        }
    }
    patch
}

fn correlation(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let n = a.len() as f32;
    let a_mean: f32 = a.iter().sum::<f32>() / n;
    let b_mean: f32 = b.iter().sum::<f32>() / n;
    let mut num = 0.0f32;
    let mut a_var = 0.0f32;
    let mut b_var = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let ax = x - a_mean;
        let bx = y - b_mean;
        num += ax * bx;
        a_var += ax * ax;
        b_var += bx * bx;
    }
    let denom = (a_var * b_var).sqrt();
    if denom < 1e-6 { return 0.0; }
    num / denom
}

fn save_patch(path: &Path, patch: &[f32], w: u32, h: u32) {
    let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(w, h);
    for (i, val) in patch.iter().enumerate() {
        let x = i as u32 % w;
        let y = i as u32 / w;
        img.put_pixel(x, y, Rgb([((val.clamp(0.0, 1.0) * 255.0) as u8); 3]));
    }
    img.save(path).ok();
}

fn main() {
    println!("{:=<60}", "");
    println!("Generate PROOF.md with Original Templates");
    println!("{:=<60}", "");
    
    let test_data = Path::new("test_data");
    let answers_file = test_data.join("extensive/answers.jsonl");
    let output_dir = test_data.join("proof_output");
    let proof_path = test_data.join("PROOF.md");
    let gt_path = test_data.join("answers.jsonl");  // Will overwrite with verified GT
    
    fs::create_dir_all(&output_dir).expect("Failed to create output dir");
    
    let matcher = vulkan_tensor_matching::VulkanTensorMatcher::new().expect("Failed to create matcher");
    
    let mut results = Vec::new();
    let mut verified_gt = Vec::new();
    
    for (case_idx, line) in fs::read_to_string(&answers_file).expect("Failed to read answers").lines().enumerate() {
        let answer: serde_json::Value = serde_json::from_str(line).expect("Failed to parse JSON");
        
        let img_path = test_data.join(answer["image_path"].as_str().unwrap());
        let tmpl_path = test_data.join(answer["template_path"].as_str().unwrap());
        let gt = &answer["expected_matches"][0];
        
        let gt_x = gt["x"].as_u64().unwrap() as u32;
        let gt_y = gt["y"].as_u64().unwrap() as u32;
        let gt_w = gt["w"].as_u64().unwrap() as u32;
        let gt_h = gt["h"].as_u64().unwrap() as u32;
        let gt_angle = gt["angle"].as_f64().unwrap_or(0.0);
        
        println!("\nCase {}: {}", case_idx, img_path.display());
        
        if !img_path.exists() || !tmpl_path.exists() {
            println!("  SKIP: Files not found");
            continue;
        }
        
        let img = ImageData::from_file(img_path.to_str().unwrap()).expect("Failed to load image");
        let template = ImageData::from_file(tmpl_path.to_str().unwrap()).expect("Failed to load template");
        
        // Verify GT location actually matches template
        let gt_patch = extract_patch(&img, gt_x, gt_y, gt_w, gt_h);
        let tmpl_patch = template.data.clone();
        let gt_corr = correlation(&gt_patch, &tmpl_patch);
        let gt_valid = gt_corr > CORR_THRESHOLD;
        
        println!("  GT: ({}, {}) {}x{} angle={:.1}°", gt_x, gt_y, gt_w, gt_h, gt_angle.to_degrees());
        println!("  GT patch vs Template correlation: {:.3} {}", gt_corr, if gt_valid { "✓" } else { "✗" });
        
        // Run Vulkan matching
        let start = Instant::now();
        let matches = matcher.match_template(&img, &template, 0.3, 1).expect("Matching failed");
        let duration_s = start.elapsed().as_secs_f32();
        
        let gt_cx = gt_x as f32 + gt_w as f32 / 2.0;
        let gt_cy = gt_y as f32 + gt_h as f32 / 2.0;
        
        if matches.is_empty() {
            println!("  ✗ No matches found");
            results.push(CaseResult {
                case_idx, gt_valid, passed: false, distance: 999.0, vulkan_corr: 0.0,
                image_path: img_path.strip_prefix(test_data).unwrap().to_string_lossy().to_string(),
                template_path: tmpl_path.strip_prefix(test_data).unwrap().to_string_lossy().to_string(),
                gt_x, gt_y, gt_w, gt_h, gt_angle,
                vulkan_x: 0, vulkan_y: 0, vulkan_rot: 0.0, duration_s,
            });
            continue;
        }
        
        let best = &matches[0];
        let dist = ((best.x as f32 - gt_cx).powi(2) + (best.y as f32 - gt_cy).powi(2)).sqrt();
        let passed = dist < DISTANCE_THRESHOLD && best.correlation > CORR_THRESHOLD;
        
        println!("  Vulkan: ({}, {}) corr={:.3} dist={:.1}px ({:.1}s) {}", 
                 best.x, best.y, best.correlation, dist, duration_s, if passed { "✓" } else { "✗" });
        
        // Generate visualization
        let viz_path = output_dir.join(format!("case_{:03}_proof.png", case_idx));
        let orig_rgb = image::open(&img_path).expect("Failed to open image").to_rgb8();
        let mut viz = orig_rgb.clone();
        
        // Draw GT (purple)
        for x in gt_x..gt_x + gt_w {
            if x < viz.width() { viz.put_pixel(x, gt_y, Rgb([128, 0, 128])); }
            if x < viz.width() { viz.put_pixel(x, gt_y + gt_h - 1, Rgb([128, 0, 128])); }
        }
        for y in gt_y..gt_y + gt_h {
            if y < viz.height() { viz.put_pixel(gt_x, y, Rgb([128, 0, 128])); }
            if y < viz.height() { viz.put_pixel(gt_x + gt_w - 1, y, Rgb([128, 0, 128])); }
        }
        
        // Draw Vulkan match (green if pass, orange if fail)
        let det_x = (best.x as i32 - template.width as i32 / 2) as u32;
        let det_y = (best.y as i32 - template.height as i32 / 2) as u32;
        let match_color = if passed { Rgb([0, 255, 0]) } else { Rgb([255, 165, 0]) };
        for x in det_x..det_x + template.width {
            if x < viz.width() { viz.put_pixel(x, det_y, match_color); }
            if x < viz.width() { viz.put_pixel(x, det_y + template.height - 1, match_color); }
        }
        for y in det_y..det_y + template.height {
            if y < viz.height() { viz.put_pixel(det_x, y, match_color); }
            if y < viz.height() { viz.put_pixel(det_x + template.width - 1, y, match_color); }
        }
        
        viz.save(&viz_path).expect("Failed to save viz");
        
        // Save template patch
        let tmpl_viz_path = output_dir.join(format!("case_{:03}_template.png", case_idx));
        save_patch(&tmpl_viz_path, &tmpl_patch, template.width, template.height);
        
        results.push(CaseResult {
            case_idx, gt_valid, passed, distance: dist, vulkan_corr: best.correlation,
            image_path: img_path.strip_prefix(test_data).unwrap().to_string_lossy().to_string(),
            template_path: tmpl_path.strip_prefix(test_data).unwrap().to_string_lossy().to_string(),
            gt_x, gt_y, gt_w, gt_h, gt_angle,
            vulkan_x: best.x, vulkan_y: best.y, vulkan_rot: best.rotation_angle, duration_s,
        });
        
        // Add to verified GT if valid
        if gt_valid {
            verified_gt.push(serde_json::json!({
                "image_path": img_path.strip_prefix(test_data).unwrap().to_string_lossy().to_string(),
                "template_path": tmpl_path.strip_prefix(test_data).unwrap().to_string_lossy().to_string(),
                "expected_matches": [{
                    "x": gt_x,
                    "y": gt_y,
                    "w": gt_w,
                    "h": gt_h,
                    "angle": gt_angle.to_degrees()
                }]
            }));
        }
    }
    
    // Generate PROOF.md
    let mut md = String::new();
    md.push_str("# Vulkan Tensorial Template Matching - Visual Proof\n\n");
    md.push_str(&format!("**Generated:** {}\n\n", chrono::Local::now().format("%Y-%m-%d %H:%M:%S")));
    
    md.push_str("## Summary\n\n");
    let pass_count = results.iter().filter(|r| r.passed).count();
    let gt_valid_count = results.iter().filter(|r| r.gt_valid).count();
    md.push_str(&format!("- Total cases: {}\n", results.len()));
    md.push_str(&format!("- Vulkan TTM passed: {}/{}\n", pass_count, results.len()));
    md.push_str(&format!("- GT locations verified: {}/{}\n\n", gt_valid_count, results.len()));
    
    md.push_str("## Results Table\n\n");
    md.push_str("| Case | Image | GT Valid | Vulkan Corr | Distance | Status |\n");
    md.push_str("|------|-------|----------|-------------|----------|--------|\n");
    
    for r in &results {
        let gt_status = if r.gt_valid { "✓" } else { "✗" };
        let status = if r.passed { "✓" } else if !r.gt_valid { "⚠️ GT invalid" } else { "✗" };
        md.push_str(&format!(
            "| {} | {} | {} | {:.3} | {:.1}px | {} |\n",
            r.case_idx,
            Path::new(&r.image_path).file_name().unwrap_or_default().to_string_lossy(),
            gt_status, r.vulkan_corr, r.distance, status
        ));
    }
    
    md.push_str("\n---\n\n");
    
    // Detailed results
    for r in &results {
        md.push_str(&format!("### Case {}: {}\n\n", r.case_idx, Path::new(&r.image_path).file_name().unwrap_or_default().to_string_lossy()));
        md.push_str(&format!("**Ground Truth**: ({}, {}) {}x{}\n\n", r.gt_x, r.gt_y, r.gt_w, r.gt_h));
        md.push_str(&format!("**GT Valid**: {}\n\n", if r.gt_valid { "✓ Yes" } else { "✗ No - template doesn't match at this location!" }));
        
        md.push_str("**Vulkan TTM Result**:\n");
        md.push_str(&format!("- Position: ({}, {})\n", r.vulkan_x, r.vulkan_y));
        md.push_str(&format!("- Correlation: {:.3}\n", r.vulkan_corr));
        md.push_str(&format!("- Rotation: {:.1}°\n", r.vulkan_rot.to_degrees()));
        md.push_str(&format!("- Distance from GT: {:.1}px\n", r.distance));
        md.push_str(&format!("- Duration: {:.1}s\n\n", r.duration_s));
        
        md.push_str(&format!("![Proof](proof_output/case_{:03}_proof.png)\n\n", r.case_idx));
        md.push_str(&format!("![Template](proof_output/case_{:03}_template.png)\n\n", r.case_idx));
        md.push_str("---\n\n");
    }
    
    fs::write(&proof_path, md).expect("Failed to write PROOF.md");
    
    // Write verified GT
    let mut gt_file = File::create(&gt_path).expect("Failed to create GT file");
    for gt in &verified_gt {
        writeln!(gt_file, "{}", gt).expect("Failed to write GT");
    }
    
    println!("\n{:=<60}", "");
    println!("SUMMARY");
    println!("{:=<60}", "");
    println!("Total cases: {}", results.len());
    println!("Vulkan TTM passed: {}/{}", pass_count, results.len());
    println!("GT locations verified: {}/{}", gt_valid_count, results.len());
    println!("\nOutput:");
    println!("  - {}", proof_path.display());
    println!("  - {}", gt_path.display());
    println!("  - {} visualizations in {}/", results.len(), output_dir.display());
}
