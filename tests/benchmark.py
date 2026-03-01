"""
Simple benchmark tests for the Rust-Python library.
"""

import time

from rust_python_lib import fibonacci, process_numbers


def benchmark_fibonacci():
    """Benchmark the fibonacci function."""
    print("Benchmarking fibonacci...")

    start_time = time.time()
    result = fibonacci(35)  # Calculate first 35 Fibonacci numbers
    end_time = time.time()

    print(f"Fibonacci(35) took {end_time - start_time:.4f} seconds")
    print(f"Result length: {len(result)}")


def benchmark_process_numbers():
    """Benchmark the process_numbers function."""
    print("Benchmarking process_numbers...")

    # Create a large list of numbers
    numbers = [float(i) for i in range(1000000)]

    start_time = time.time()
    result = process_numbers(numbers)
    end_time = time.time()

    print(f"Processing 1,000,000 numbers took {end_time - start_time:.4f} seconds")
    print(f"Sum: {result.sum}, Average: {result.average}")


if __name__ == "__main__":
    print("Running benchmarks...")
    benchmark_fibonacci()
    print()
    benchmark_process_numbers()
    print("Benchmarks completed!")
