# Python Benchmarks for Template Matching

This directory contains Python benchmarks using OpenCV to establish a baseline for comparison with the Rust implementations.

## Setup

Install dependencies using uv (recommended):

```bash
uv venv
source .venv/bin/activate
uv pip install .
```

Or install dependencies directly with uv:

```bash
uv venv
source .venv/bin/activate
uv pip install opencv-python numpy pandas seaborn
```

Note: We always use `uv` for dependency management in our Python benchmarks.

## Running the Benchmark

```bash
source .venv/bin/activate
python bench.py
```

This will run template matching on the Lenna test image and provide performance metrics for comparison with the Rust CPU and GPU implementations.

## Comparing Performance

To compare all implementations and generate plots:

```bash
# Run Python benchmark
python bench.py

# Run Rust benchmarks (from project root)
cd ../..
CARGO_TARGET_DIR=target cargo bench --bench cpu_template_matching
CARGO_TARGET_DIR=target cargo bench --bench vulkan_template_matching
cd benches/python_benchmarks

# Generate plots
python plot.py
```

See `performance_comparison.md` for a summary of results and analysis.