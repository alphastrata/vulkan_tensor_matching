# Build configurations
rust_target := "target/release"
python_package := "rust_python_lib"
module_name := "rust_python_lib"
venv_dir := "venv"

# Default target
default: test

# Setup development environment
setup:
    uv venv {{ venv_dir }}
    uv sync --dev

# Build Rust library in release mode
build:
    uv run --with maturin maturin build --release

# Run Python example
run:
    uv run python python/main.py

# Run all tests (Rust + Python)
# Uses cargo-nextest if available, otherwise cargo test --release
test:
    # Rust tests
    @if command -v cargo-nextest &> /dev/null; then \
        echo "Using cargo-nextest..."; \
        cargo nextest run --release; \
    else \
        echo "cargo-nextest not found, using cargo test --release..."; \
        cargo test --release; \
    fi
    # Python tests
    uv run python tests/test_vulkan_matching.py

# Run Python tests only
test-python:
    uv run python tests/test_vulkan_matching.py

# Run Rust tests only
test-rust:
    @if command -v cargo-nextest &> /dev/null; then \
        cargo nextest run --release; \
    else \
        cargo test --release; \
    fi

# Format all code (Rust + Python)
fmt:
    cargo fmt
    uv run ruff format python/ tests/

# Lint all code (Rust + Python)
check:
    # Rust lint
    cargo clippy -- -W warnings
    # Python lint
    uv run ruff check python/ tests/

# Check formatting without making changes
fmt-check:
    cargo fmt --check
    uv run ruff format --check python/ tests/

# Clean build artifacts
clean:
    cargo clean
    rm -rf __pycache__ tests/__pycache__
    find . -name "*.so" -delete
    find . -name "*.pyc" -delete

# Build wheels for distribution
wheel:
    uv run --with maturin maturin build --release

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
