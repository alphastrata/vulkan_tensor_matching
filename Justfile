# Justfile for Vulkan Tensor Matching

default:
    @just --list

build:
    @echo "Building Rust library..."
    cargo build --release
    @echo "Building Python bindings..."
    uv run maturin develop

test:
    @echo "Running Rust tests..."
    cargo test --release --lib --tests

bench:
    cargo bench

proof:
    @echo "Running proof pipeline..."
    uv run python tests/proof_pipeline.py

fmt:
    @echo "Formatting Rust..."
    cargo fmt --all
    @echo "Formatting Python..."
    uv run ruff format .

lint:
    @echo "Running clippy on library..."
    cargo clippy --lib -- -D warnings
    @echo "Running ruff on Python..."
    uv run ruff check tests/proof_pipeline.py

check: build test lint

clean:
    cargo clean
    rm -f vulkan_*.png extracted_*.png
    rm -rf test_data/proof_output/*

example:
    cargo run --release --example lenna_vulkan_matching

compare:
    cargo test --test imageproc_comparison_test --release -- --nocapture
