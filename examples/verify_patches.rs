//! Verify what's at our match location vs claimed GT

use image::{GenericImageView, ImageBuffer, Rgb};
use std::fs;
use vulkan_tensor_matching::ImageData;

fn extract_patch(img: &ImageData, x: u32, y: u32, w: u32, h: u32) -> Vec<f32> {
    let mut patch = Vec::with_capacity((w * h) as usize);
    for row in 0..h {
        for col in 0..w {
            let px = x + col;
            let py = y + row;
            if py < img.height && px < img.width {
                let idx = (py * img.width + px) as usize;
                patch.push(img.data[idx]);
            } else {
                patch.push(0.0);
            }
        }
    }
    patch
}

fn save_patch(path: &str, patch: &[f32], w: u32, h: u32) {
    let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(w, h);
    for (i, val) in patch.iter().enumerate() {
        let x = i as u32 % w;
        let y = i as u32 / w;
        let gray = ((val.clamp(0.0, 1.0) * 255.0) as u8);
        img.put_pixel(x, y, Rgb([gray, gray, gray]));
    }
    img.save(path).expect("Failed to save");
}

fn correlation(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let n = a.len() as f32;
    let a_mean: f32 = a.iter().sum::<f32>() / n;
    let b_mean: f32 = b.iter().sum::<f32>() / n;

    let mut numerator = 0.0f32;
    let mut a_var = 0.0f32;
    let mut b_var = 0.0f32;

    for (x, y) in a.iter().zip(b.iter()) {
        let ax = x - a_mean;
        let bx = y - b_mean;
        numerator += ax * bx;
        a_var += ax * ax;
        b_var += bx * bx;
    }

    let denom = (a_var * b_var).sqrt();
    if denom < 1e-6 {
        return 0.0;
    }

    numerator / denom
}

fn main() {
    println!("{:=<60}", "");
    println!("Patch Comparison: Our Match vs Claimed GT");
    println!("{:=<60}", "");

    // Case 4: source_4.png
    let img = ImageData::from_file("test_data/extensive/source_images/source_4.png").unwrap();
    let template = ImageData::from_file("test_data/extensive/templates/template_4.png").unwrap();

    println!("Image: {}x{}", img.width, img.height);
    println!("Template: {}x{}", template.width, template.height);

    // Claimed GT
    let gt_x = 281u32;
    let gt_y = 276u32;
    let gt_w = template.width;
    let gt_h = template.height;

    // Our match
    let our_x = 281u32; // We matched at (281, 276) which IS the GT!
    let our_y = 276u32;

    println!("\nClaimed GT: ({}, {})", gt_x, gt_y);
    println!("Our match:  ({}, {})", our_x, our_y);

    let gt_patch = extract_patch(&img, gt_x, gt_y, gt_w, gt_h);
    let our_patch = extract_patch(&img, our_x, our_y, gt_w, gt_h);
    let tmpl_patch = template.data.clone();

    // Compare
    let gt_vs_tmpl = correlation(&gt_patch, &tmpl_patch);
    let our_vs_tmpl = correlation(&our_patch, &tmpl_patch);
    let gt_vs_our = correlation(&gt_patch, &our_patch);

    println!("\nCorrelations:");
    println!("  GT patch vs Template: {:.3}", gt_vs_tmpl);
    println!("  Our patch vs Template: {:.3}", our_vs_tmpl);
    println!("  GT patch vs Our patch: {:.3}", gt_vs_our);

    // Save patches for visual inspection
    save_patch("test_data/debug_gt_patch.png", &gt_patch, gt_w, gt_h);
    save_patch("test_data/debug_our_patch.png", &our_patch, gt_w, gt_h);
    save_patch("test_data/debug_template.png", &tmpl_patch, gt_w, gt_h);

    println!("\nSaved patches to test_data/debug_*.png");

    // Now check case 0 where we failed
    println!("\n{:=<60}", "");
    println!("Case 0: source_0.png");
    println!("{:=<60}", "");

    let img0 = ImageData::from_file("test_data/extensive/source_images/source_0.png").unwrap();
    let tmpl0 = ImageData::from_file("test_data/extensive/templates/template_0.png").unwrap();

    // Claimed GT
    let gt0_x = 527u32;
    let gt0_y = 283u32;

    // Our match
    let our0_x = 458u32;
    let our0_y = 324u32;

    let gt0_patch = extract_patch(&img0, gt0_x, gt0_y, tmpl0.width, tmpl0.height);
    let our0_patch = extract_patch(&img0, our0_x, our0_y, tmpl0.width, tmpl0.height);
    let tmpl0_patch = tmpl0.data.clone();

    let gt0_vs_tmpl = correlation(&gt0_patch, &tmpl0_patch);
    let our0_vs_tmpl = correlation(&our0_patch, &tmpl0_patch);

    println!("Claimed GT: ({}, {})", gt0_x, gt0_y);
    println!("Our match:  ({}, {})", our0_x, our0_y);
    println!("\nCorrelations:");
    println!("  GT patch vs Template: {:.3}", gt0_vs_tmpl);
    println!("  Our patch vs Template: {:.3}", our0_vs_tmpl);

    // Maybe our location is ALSO a match?
    if our0_vs_tmpl > 0.9 {
        println!("\n✓ Our location is ALSO a valid match!");
    }
    if gt0_vs_tmpl < 0.9 {
        println!("✗ Claimed GT is NOT a good match either!");
    }

    save_patch(
        "test_data/debug_case0_gt.png",
        &gt0_patch,
        tmpl0.width,
        tmpl0.height,
    );
    save_patch(
        "test_data/debug_case0_ours.png",
        &our0_patch,
        tmpl0.width,
        tmpl0.height,
    );
    save_patch(
        "test_data/debug_case0_tmpl.png",
        &tmpl0_patch,
        tmpl0.width,
        tmpl0.height,
    );
}
