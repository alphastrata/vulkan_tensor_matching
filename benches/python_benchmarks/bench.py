#!/usr/bin/env python3

import cv2
import numpy as np
import time
import os
import json
import csv
from typing import List, Tuple, Dict


def load_image_as_grayscale(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return img


def benchmark_opencv_template_matching(
    target_path: str, template_path: str, iterations: int = 10
) -> Tuple[List[float], List[Tuple[int, int, float]]]:
    target_img = load_image_as_grayscale(target_path)
    template_img = load_image_as_grayscale(template_path)

    times = []
    result = None

    for i in range(iterations):
        start_time = time.perf_counter()
        result = cv2.matchTemplate(target_img, template_img, cv2.TM_CCOEFF_NORMED)
        end_time = time.perf_counter()

        elapsed_time = (end_time - start_time) * 1000
        times.append(elapsed_time)

    threshold = 0.8
    locations = np.where(result >= threshold)
    matches = list(zip(locations[1], locations[0], result[locations]))

    matches.sort(key=lambda x: x[2], reverse=True)
    matches = matches[:10]

    return times, matches


def parse_criterion_json_results(criterion_dir: str) -> Dict[str, Dict[str, float]]:
    results = {}

    for root, dirs, files in os.walk(criterion_dir):
        if "base" in dirs:
            base_path = os.path.join(root, "base")
            if os.path.exists(os.path.join(base_path, "estimates.json")):
                relative_path = os.path.relpath(root, criterion_dir)
                benchmark_name = relative_path.replace("/", "_").replace("\\", "_")

                try:
                    with open(os.path.join(base_path, "estimates.json"), "r") as f:
                        estimates = json.load(f)

                    with open(os.path.join(base_path, "sample.json"), "r") as f:
                        sample_data = json.load(f)

                    mean_time = estimates["mean"]["point_estimate"] / 1_000_000
                    min_time = min(sample_data["times"]) / 1_000_000
                    max_time = max(sample_data["times"]) / 1_000_000

                    results[benchmark_name] = {
                        "mean_time_ms": mean_time,
                        "min_time_ms": min_time,
                        "max_time_ms": max_time,
                        "sample_count": len(sample_data["times"]),
                    }

                except (FileNotFoundError, json.JSONDecodeError, KeyError):
                    continue

    return results


def export_benchmark_csv(
    criterion_dir: str, target_path: str, template_path: str
) -> None:
    # Export Python benchmark results
    all_times = []
    for batch in range(5):
        times, matches = benchmark_opencv_template_matching(
            target_path, template_path, iterations=10
        )
        all_times.extend(times)

    mean_time_ms = sum(all_times) / len(all_times)
    mean_time_ns = mean_time_ms * 1_000_000

    with open("python_results.csv", "w", newline="") as f:
        fieldnames = [
            "group",
            "function",
            "value",
            "iteration_count",
            "sample_measured_value",
            "unit",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "group": "python_opencv",
                "function": "template_matching_lenna_test1",
                "value": mean_time_ns,
                "iteration_count": 1.0,
                "sample_measured_value": mean_time_ns,
                "unit": "ns",
            }
        )

    # Export Criterion results
    if os.path.exists(criterion_dir):
        criterion_results = parse_criterion_json_results(criterion_dir)

        csv_data = []
        for benchmark_name, stats in criterion_results.items():
            # We'll use the mean time from the stats, but we need to get the actual benchmark data
            # Let's parse the estimates.json files directly
            benchmark_path = os.path.join(
                criterion_dir, benchmark_name.replace("_", "/", 1), "base"
            )
            if os.path.exists(os.path.join(benchmark_path, "estimates.json")):
                try:
                    with open(os.path.join(benchmark_path, "benchmark.json"), "r") as f:
                        benchmark_info = json.load(f)

                    with open(os.path.join(benchmark_path, "estimates.json"), "r") as f:
                        estimates_data = json.load(f)

                    mean_time_ns = estimates_data["mean"]["point_estimate"]

                    csv_data.append(
                        {
                            "group": benchmark_info.get("group_id", "unknown"),
                            "function": benchmark_info.get(
                                "function_id", benchmark_name
                            ),
                            "value": mean_time_ns,
                            "iteration_count": 1.0,
                            "sample_measured_value": mean_time_ns,
                            "unit": "ns",
                        }
                    )
                except (FileNotFoundError, json.JSONDecodeError, KeyError):
                    continue

        if csv_data:
            with open("criterion_results.csv", "w", newline="") as f:
                fieldnames = [
                    "group",
                    "function",
                    "value",
                    "iteration_count",
                    "sample_measured_value",
                    "unit",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)


def main() -> None:
    target_path = "../../test_assets/lenna.png"
    template_path = "../../test_assets/templates/test1.png"

    if not os.path.exists(target_path) or not os.path.exists(template_path):
        return

    # Run detailed benchmark
    all_times = []
    all_matches = []

    for batch in range(5):
        times, matches = benchmark_opencv_template_matching(
            target_path, template_path, iterations=10
        )
        all_times.extend(times)
        if batch == 0:
            all_matches = matches

    avg_time = sum(all_times) / len(all_times)
    min_time = min(all_times)
    max_time = max(all_times)
    std_dev = (sum((x - avg_time) ** 2 for x in all_times) / len(all_times)) ** 0.5

    print(f"Total runs: {len(all_times)}")
    print(f"Average time: {avg_time:.3f} ms")
    print(f"Min time: {min_time:.3f} ms")
    print(f"Max time: {max_time:.3f} ms")
    print(f"Standard deviation: {std_dev:.3f} ms")
    print(f"Number of matches: {len(all_matches)}")

    # Export CSV data
    criterion_dir = "../../../target/criterion"
    if not os.path.exists(criterion_dir):
        criterion_dir = "../../target/criterion"

    export_benchmark_csv(criterion_dir, target_path, template_path)

    # Show Criterion results
    if os.path.exists(criterion_dir):
        criterion_results = parse_criterion_json_results(criterion_dir)
        for benchmark_name, stats in criterion_results.items():
            print(f"\n{benchmark_name}:")
            print(f"  Mean time: {stats['mean_time_ms']:.3f} ms")
            print(f"  Min time: {stats['min_time_ms']:.3f} ms")
            print(f"  Max time: {stats['max_time_ms']:.3f} ms")
            print(f"  Sample count: {stats['sample_count']}")


if __name__ == "__main__":
    main()
