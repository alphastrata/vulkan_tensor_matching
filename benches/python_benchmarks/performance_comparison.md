# Performance Comparison Report

## Benchmark Results

| Implementation | Average Time | Notes |
|---------------|-------------|-------|
| CPU (Rust) | ~728 ms | imageproc crate |
| GPU (Rust/Vulkan) | ~2.1 ms | Custom Vulkan implementation (**~347x faster than CPU**) |
| CPU (Python/OpenCV) | ~4.6 ms | cv2.matchTemplate |

## Test Details

- Target image: lenna.png (512x512)
- Template image: test1.png (64x64)
- Correlation threshold: 0.8
- Max matches: 10
- Hardware used:
  - OS: Pop!_OS 24.04 LTS
  - Kernel: 6.16.3-76061603-generic
  - CPU: AMD Ryzen 9 5950X 16-Core Processor
  - Memory: 31Gi

## Visualization

Performance comparison plot:

![](performance_comparison.svg)

This plot is generated from CSV data exported from both Rust Criterion benchmarks and Python OpenCV benchmarks, allowing direct comparison across implementations.