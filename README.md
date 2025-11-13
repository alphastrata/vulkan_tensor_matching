# Vulkan Tensorial Template Matching

An 'allegedly' High-performance rotation-invariant template matching library using Vulkan compute shaders and tensor mathematics accelerated image processing (template matching specifically).
Based on this [Tensor-based template matching paper](https://arxiv.org/abs/2408.02398v1 [cs.CV]), by Antonio Martinez-Sanchez.

I have done my very best to faithfully interpret the paper and its algorithms here using Vulkan.

This is **NOT** as well tested as the things it compares to, i.e imageproc or opencv -- so if you're looking for something extremely robust, this is not for you.

## Usage:

### 1. Loading Images
```rust
use vulkan_tensor_matching::ImageData;

// Load target image from disk (e.g., lenna.png)
let target = ImageData::from_file("test_data/lenna.png")?;

// Load template from disk (e.g., test1.png)
let template = ImageData::from_file("test_data/templates/test1.png")?;

// Or extract template from region of interest
let template = target.extract_region(100, 100, 50, 50)?;
```

### 2. Performing Template Matching
```rust
use vulkan_tensor_matching::VulkanTensorMatcher;

// Initialise matcher (Vulkan GPU acceleration)
let matcher = VulkanTensorMatcher::new()?;

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
- **Position (x,y)**: Location coordinates in the target image (centre of the match)
- **Correlation**: Similarity score (0.0 to 1.0)
- **Rotation**: Template orientation in radians
- **Confidence**: Algorithm's certainty in the detection

```rust
for (i, m) in matches.iter().enumerate() {
    println!("Match {}: ({}, {}) correlation={:.3} rotation={:.1}°",
             i + 1, m.x, m.y, m.correlation,
             m.rotation_angle.to_degrees());
}
```

## Installation
NOTE: You need `vulkan-sdk` and `shaderc` installed on your system.

Add to your `Cargo.toml`:

```toml
[dependencies]
vulkan_tensor_matching = { git = "https://github.com/alphastrata/vulkan_tensor_matching" }
```

## Examples

Run the Lenna example with:
```bash
cargo run --release --example lenna_vulkan_matching
```

## Testing

### Vulkan GPU vs CPU Comparison Test

```bash
# Run Vulkan GPU comparison against imageproc CPU
cargo test --test imageproc_comparison_test --release -- --nocapture
```

### Proof Visualization

A Python script is provided to generate a comprehensive proof document with visual annotations:

```bash
# Install dependencies
pip install pillow

# Run the visualizer
python tests/proof_visualizer.py
```

Results will be saved to `test_data/proof.md`.

## License
MIT
