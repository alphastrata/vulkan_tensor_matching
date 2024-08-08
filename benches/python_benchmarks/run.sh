#!/bin/bash

# Simple script to run all benchmarks and generate plots

echo "Running Python benchmark..."
source .venv/bin/activate
python bench.py

echo "Running Rust CPU benchmark..."
cd ../..
CARGO_TARGET_DIR=target cargo bench --bench cpu_template_matching
cd benches/python_benchmarks

echo "Running Rust GPU benchmark..."
cd ../..
CARGO_TARGET_DIR=target cargo bench --bench vulkan_template_matching
cd benches/python_benchmarks

echo "Generating plots..."
python plot.py

echo "Done! Check performance_comparison.svg for results."