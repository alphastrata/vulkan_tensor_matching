# Justfile for Vulkan Tensor Matching
# Unified commands for both Rust and Python

default:
    @just --list

# Build everything
build:
    @echo "Building Rust library..."
    cargo build --release
    @echo "Building Python bindings..."
    uv run maturin develop --features python

# Run all tests
test:
    @echo "Running Rust tests..."
    cargo test --release --lib --tests
    @echo "Running Python tests..."
    uv run pytest tests/test_vulkan_matching.py tests/test_tensor_rotation.py -v

# Run benchmarks
bench:
    cargo bench

# Run proof pipeline (generates test_data/proof.html)
proof:
    @echo "Running proof pipeline..."
    uv run python tests/proof_pipeline.py
    @echo "View results: open test_data/proof.html"

# Format all code
fmt:
    @echo "Formatting Rust..."
    cargo fmt --all
    @echo "Formatting Python..."
    uv run ruff format .

# Lint all code (Rust clippy + Python ruff)
lint:
    @echo "Running clippy on library..."
    cargo clippy --lib -- -D warnings
    @echo "Running ruff on Python..."
    uv run ruff check tests/*.py

# Full check: build + test + lint
check: build test lint

# Clean build artifacts
clean:
    cargo clean
    rm -rf target/
    rm -f vulkan_*.png extracted_*.png
    rm -rf test_data/proof_output/*

# Run Lenna example
example:
    cargo run --release --example lenna_vulkan_matching

# Compare with imageproc
compare:
    cargo test --test imageproc_comparison_test --release -- --nocapture
