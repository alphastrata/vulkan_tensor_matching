# Justfile for Vulkan Tensor Matching

default:
    @just --list

build:
    cargo build --release

build-py:
    uv run maturin develop --features python

test-rs:
    cargo test --release --lib --tests

test-py: build-py
    uv run pytest tests/test_vulkan_matching.py tests/test_tensor_rotation.py

test: test-rs test-py

test-comparison:
    cargo test --test imageproc_comparison_test --release -- --nocapture

bench:
    cargo bench

example: build
    cargo run --release --example lenna_vulkan_matching

proof: build-py
    uv run python3 tests/proof_visualizer.py

fmt: fmt-rs fmt-py

fmt-rs:
    cargo fmt --all

fmt-py:
    uv run ruff format .

check: clippy fmt test

clippy:
    cargo clippy --all-targets --all-features -- -D warnings

clean:
    cargo clean
    rm -f vulkan_match_*.png vulkan_lenna_match.png vulkan_synthetic_match.png vulkan_exact_match.png
    rm -rf test_data/proof_output/*
