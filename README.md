# Vulkan Tensorial Template Matching

An 'allegedly' High-performance rotation-invariant template matching library using Vulkan compute shaders and tensor mathematics accelerated image processing (template matching specifically).
Based on this [Tensor-based template matching paper](https://arxiv.org/abs/2408.02398), by Antonio Martinez-Sanchez.

I have done my very best to faithfully interpret the parper and its algorithms here using Vulkan (Not an API I am super familiar with, so feedback is most welcome).

I say 'allegedly' above, but in truth the implementation is _quite_ fast, especially when compared to the cpu implementation from `imageproc` (Which is a great create, I'm not throwing any shade).

## Usage:

### 1. Loading Images
```rust
use vulkan_tensor_matching::ImageData;

// Load target image from disk (e.g., lenna.png)
let target = ImageData::from_file("test_assets/lenna.png")?;

// Load template from disk (e.g., test1.png or test2.png)
let template = ImageData::from_file("test_assets/templates/test1.png")?;

// Or extract template from region of interest
let template = ImageData::extract_region(&target, 100, 100, 50, 50)?;
```

### 2. Performing Template Matching
```rust
use vulkan_tensor_matching::VulkanTensorMatcher;

// Initialise matcher (Vulkan GPU acceleration)
let matcher = VulkanTensorMatcher::new(); // Non-async API

// Find matches with correlation threshold
let matches = matcher.match_template(
    &target,     // Target image
    &template,   // Template to find
    0.8,         // Minimum correlation (0.0-1.0)
    10           // Maximum matches to return
)?;
```

### 3. Understanding Matches
Each match contains:
- **Position (x,y)**: Location coordinates in the target image
- **Correlation**: Similarity score (0.0 = no match, 1.0 = perfect match)
- **Rotation**: Template orientation in degrees (-180° to +180°)
- **Confidence**: Algorithm's certainty in the detection

```rust
for (i, match) in matches.iter().enumerate() {
    println!("Match {}: ({}, {}) correlation={:.3} rotation={:.1}°",
             i + 1, match.x, match.y, match.correlation,
             match.rotation_angle.to_degrees());
}
```

### 4. Interpreting Results
Correlation Score Guide:
- **0.95-1.00**: Excellent match
- **0.85-0.95**: Good match
- **0.70-0.85**: Fair match
- **0.50-0.70**: Poor match
- **Below 0.50**: Likely not a match

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
vulkan_tensor_matching = { git = "https://github.com/alphastrata/vulkan_tensor_matching?tab=readme-ov-file" }
```

### Async Runtime Configuration
The library supports multiple async runtimes via feature flags:
```toml
# Default (uses pollster)
vulkan_tensor_matching = "0.1"

# Or with specific runtime
vulkan_tensor_matching = { version = "0.1", features = ["tokio"] }
vulkan_tensor_matching = { version = "0.1", features = ["smol"] }
```

## Examples

Run any example with:
```bash
cargo run --example <example_name>
```

### Benchmark Results

| Implementation | Average Time | Notes |
|---------------|-------------|-------|
| CPU (Rust) | ~725 ms | imageproc crate |
| GPU (Rust/Vulkan) | ~2.12 ms | Custom Vulkan implementation |
| CPU (Python/OpenCV) | ~4.87 ms | cv2.matchTemplate |

### Performance Comparison Scripts

OpenCV benchmark scripts are available in the `benches/python_benchmarks` directory for comparison purposes. These scripts can be run using `uv` for dependency management:

```bash
cd benches/python_benchmarks
uv venv
source .venv/bin/activate
uv pip install opencv-python numpy pandas seaborn
python bench.py
```

The scripts generate performance comparison plots in SVG format.

## License
MIT
