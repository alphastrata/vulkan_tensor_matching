#!/usr/bin/env python3

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def load_benchmark_data(criterion_csv: str, python_csv: str) -> pd.DataFrame:
    if os.path.exists(criterion_csv):
        criterion_df = pd.read_csv(criterion_csv)
        criterion_df["time_ms"] = criterion_df["sample_measured_value"] / 1_000_000
    else:
        criterion_df = pd.DataFrame()

    if os.path.exists(python_csv):
        python_df = pd.read_csv(python_csv)
        python_df["time_ms"] = python_df["sample_measured_value"] / 1_000_000
    else:
        python_df = pd.DataFrame()

    return pd.concat([criterion_df, python_df], ignore_index=True)


def create_performance_plot(df: pd.DataFrame, output_file: str) -> None:
    if df.empty:
        return

    plt.style.use("default")
    sns.set_palette("husl")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="group", y="time_ms", ax=ax)

    ax.set_xlabel("Implementation")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Template Matching Performance Comparison")
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_file, format="svg", bbox_inches="tight")
    plt.close()


def main() -> None:
    df = load_benchmark_data("criterion_results.csv", "python_results.csv")
    if not df.empty:
        create_performance_plot(df, "performance_comparison.svg")


if __name__ == "__main__":
    main()
