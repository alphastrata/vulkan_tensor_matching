//! Benchmark for GPU-accelerated Vulkan template matching
//! Using the same lenna.png and test1.png test case

use criterion::{criterion_group, criterion_main, Criterion};
use vulkan_tensor_matching::{ImageData, VulkanTensorMatcher};

fn benchmark_vulkan_template_matching(c: &mut Criterion) {
    // Configure criterion with smaller sample size
    let mut group = c.benchmark_group("vulkan_template_matching");
    group.sample_size(10);

    // Load test images
    let target_path = "test_assets/lenna.png";
    let template_path = "test_assets/templates/test1.png";

    if std::path::Path::new(target_path).exists() && std::path::Path::new(template_path).exists() {
        // Try to create Vulkan tensor matcher first, to detect if Vulkan is available
        match VulkanTensorMatcher::new() {
            Ok(matcher) => {
                let target_image = ImageData::from_file(target_path).expect("Failed to load lenna.png");
                let template_image = ImageData::from_file(template_path).expect("Failed to load test1.png");

                group.bench_function("vulkan_template_matching_lenna_test1", |b| {
                    b.iter(|| {
                        let _matches = matcher.match_template(
                            &target_image,
                            &template_image,
                            0.8, // correlation threshold
                            10   // max matches
                        ).expect("Template matching failed");
                    })
                });
            }
            Err(e) => {
                panic!("Vulkan not available on this system, skipping Vulkan benchmark: {}", e);
            }
        }
   
    }

    group.finish();
}

criterion_group!(benches, benchmark_vulkan_template_matching);
criterion_main!(benches);