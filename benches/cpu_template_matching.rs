//! Benchmark comparing CPU-based template matching using imageproc

use criterion::{criterion_group, criterion_main, Criterion};
use imageproc::template_matching::{match_template, MatchTemplateMethod};

fn benchmark_cpu_template_matching(c: &mut Criterion) {
    // Configure criterion with smaller sample size
    let mut group = c.benchmark_group("imageproc_cpu_template_matching");
    group.sample_size(10);

    // Load test images
    let target_path = "test_assets/lenna.png";
    let template_path = "test_assets/templates/test1.png";

    if std::path::Path::new(target_path).exists() && std::path::Path::new(template_path).exists() {
        // Load images using image crate
        let target_img = image::open(target_path).expect("Failed to load lenna.png");
        let template_img = image::open(template_path).expect("Failed to load test1.png");

        // Convert to grayscale
        let target_gray = target_img.to_luma8();
        let template_gray = template_img.to_luma8();

        group.bench_function("cpu_template_matching_lenna_test1", |b| {
            b.iter(|| {
                let _result = match_template(&target_gray, &template_gray, MatchTemplateMethod::SumOfSquaredErrors);
            })
        });
    } else {
        panic!("Test assets not found, skipping CPU benchmark");
    }

    group.finish();
}

criterion_group!(benches, benchmark_cpu_template_matching);
criterion_main!(benches);