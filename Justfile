# Build configurations
rust_target := "target/release"
python_package := "rust_python_lib"
module_name := "rust_python_lib"
venv_dir := "venv"

# Default target
default: test

# Setup...
setup:
    uv venv {{ venv_dir }}
    uv sync --dev

# Build Rust library in release mode
build:
    uv run --with maturin maturin build --release

# Run Python example
run:
    uv run python python/main.py

# Run Python tests
test-python:
    uv run python tests/test_vulkan_matching.py

# Run all tests
test:
    cargo test
    uv run python tests/test_vulkan_matching.py

# Clean build artifacts
clean:
    cargo clean
    rm -rf __pycache__ tests/__pycache__
    find . -name "*.so" -delete
    find . -name "*.pyc" -delete

# Build wheels for distribution
wheel:
    uv run --with maturin maturin build --release

# Format code
fmt:
    cargo fmt
    uv run black python/ tests/

# Lint code
lint:
    cargo clippy -- -W warnings
    uv run black --check python/ tests/

# Build and run benchmarks
bench:
    cargo bench
    uv run python tests/benchmark.py 2>/dev/null || true

# Development mode (fast rebuilds)
dev:
    uv run --with maturin maturin develop

# Show help
help:
    @just --list
