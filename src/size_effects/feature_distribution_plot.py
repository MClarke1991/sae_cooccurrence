import os
from os.path import join as pj
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

from sae_cooccurrence.utils.set_paths import get_git_root


def load_boxplot_stats(file_path: str) -> dict[str, Any]:
    """Load boxplot statistics from an NPZ file, falling back to H5 if not available."""
    if file_path.endswith('.npz'):
        try:
            with np.load(file_path) as data:
                stats = {
                    "observed": {key: data[f"observed_{key}"] for key in ["outliers", "median"]},
                    "expected": {key: data[f"expected_{key}"] for key in ["outliers", "median"]},
                    "total_tokens": data["total_tokens"],
                    "sae_size": data["sae_size"],
                    "activation_threshold": data["activation_threshold"],
                }
            return stats
        except FileNotFoundError:
            # If NPZ file not found, fall back to H5
            file_path = file_path.replace('.npz', '.h5')

    # Load from H5 file
    with h5py.File(file_path, "r") as f:
        stats = {
            "observed": {key: f["observed"][key][()] for key in f["observed"].keys()},  # type: ignore
            "expected": {key: f["expected"][key][()] for key in f["expected"].keys()},  # type: ignore
            "total_tokens": f["total_tokens"][()],  # type: ignore
            "sae_size": f["sae_size"][()],  # type: ignore
            "activation_threshold": f["activation_threshold"][()],  # type: ignore
        }
    return stats


def plot_boxplots(
    stats: dict[str, Any], output_dir: str, show_fliers: bool = True
) -> None:
    """Create and save boxplots for observed and expected co-occurrences."""
    fig, ax = plt.subplots(figsize=(10, 6))

    positions = [1, 2]
    colors = ["blue", "red"]
    labels = ["Observed", "Expected"]

    for i, (key, color, label) in enumerate(
        zip(["observed", "expected"], colors, labels)
    ):
        data = stats[key]
        bp = ax.boxplot(
            [data["outliers"]],
            positions=[positions[i]],
            patch_artist=True,
            widths=0.6,
            showfliers=show_fliers,
        )

        for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
            plt.setp(bp[element], color=color)

        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Plot the median
        ax.plot(
            positions[i],
            data["median"],
            color="white",
            marker="o",
            markersize=8,
            markeredgecolor="black",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Co-occurrence frequency (per token)")
    ax.set_title(
        f'Co-occurrence Distribution (SAE size: {stats["sae_size"]}, '
        f'Activation threshold: {stats["activation_threshold"]})'
    )

    if show_fliers:
        suffix = "w_outliers"
    else:
        suffix = "no_outliers"

    plt.tight_layout()
    plt.savefig(
        pj(output_dir, f'boxplot_observed_expected_{stats["sae_size"]}_{suffix}.png')
    )
    plt.close()


def load_histogram_data(file_path: str) -> dict[str, dict[str, np.ndarray]]:
    """Load histogram data from an H5 file or NPZ file."""
    if file_path.endswith('.h5'):
        try:
            with h5py.File(file_path, "r") as f:
                data = {
                    "observed": {
                        "bin_edges": f["observed/bin_edges"][()],  # type: ignore
                        "density": f["observed/density"][()],  # type: ignore
                        "log_bin_edges": f["observed/log_bin_edges"][()],  # type: ignore
                        "log_density": f["observed/log_density"][()],  # type: ignore
                    },
                    "expected": {
                        "bin_edges": f["expected/bin_edges"][()],  # type: ignore
                        "density": f["expected/density"][()],  # type: ignore
                        "log_bin_edges": f["expected/log_bin_edges"][()],  # type: ignore
                        "log_density": f["expected/log_density"][()],  # type: ignore
                    },
                }
        except OSError:
            print(f"Failed to load H5 file: {file_path}. Falling back to NPZ.")
            file_path = file_path.replace('.h5', '.npz')
    
    if file_path.endswith('.npz'):
        with np.load(file_path) as f:
            data = {
                "observed": {
                    "bin_edges": f["observed_bin_edges"],
                    "density": f["observed_density"],
                    "log_bin_edges": f["observed_log_bin_edges"],
                    "log_density": f["observed_log_density"],
                },
                "expected": {
                    "bin_edges": f["expected_bin_edges"],
                    "density": f["expected_density"],
                    "log_bin_edges": f["expected_log_bin_edges"],
                    "log_density": f["expected_log_density"],
                },
            }
    
    return data  # type: ignore


def plot_histogram(
    histogram_data: dict[str, dict[str, np.ndarray]],
    stats: dict[str, Any],
    output_dir: str,
) -> None:
    """Create and save histogram plots for observed and expected co-occurrences."""
    # Log-transformed histogram
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        histogram_data["observed"]["log_bin_edges"],
        histogram_data["observed"]["log_density"],
        label="Observed",
        color="blue",
    )
    ax.plot(
        histogram_data["expected"]["log_bin_edges"],
        histogram_data["expected"]["log_density"],
        label="Expected",
        color="red",
    )

    ax.set_xlabel("Log10 Co-occurrence (per token)")
    ax.set_ylabel("Density")
    ax.set_title(
        f'Log-transformed Co-occurrence Density Plot\n'
        f'(SAE size: {stats["sae_size"]}, Activation threshold: {stats["activation_threshold"]})'
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        pj(output_dir, f'histogram_log_observed_expected_{stats["sae_size"]}.png')
    )
    plt.close()

    # Unlogged histogram
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        histogram_data["observed"]["bin_edges"],
        histogram_data["observed"]["density"],
        label="Observed",
        color="blue",
    )
    ax.plot(
        histogram_data["expected"]["bin_edges"],
        histogram_data["expected"]["density"],
        label="Expected",
        color="red",
    )

    ax.set_xlabel("Co-occurrence (per token)")
    ax.set_ylabel("Density")
    ax.set_title(
        f'Unlogged Co-occurrence Density Plot\n'
        f'(SAE size: {stats["sae_size"]}, Activation threshold: {stats["activation_threshold"]})'
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        pj(output_dir, f'histogram_unlogged_observed_expected_{stats["sae_size"]}.png')
    )
    plt.close()


def plot_combined_boxplots(
    all_stats: list[dict[str, Any]], output_dir: str, show_fliers: bool = True
) -> None:
    """Create and save a combined boxplot for all SAE sizes."""
    # Sort all_stats by SAE size
    all_stats.sort(key=lambda x: x["sae_size"])

    fig, ax = plt.subplots(figsize=(12, 6))

    positions = []
    labels = []
    colors = ["blue", "red"]

    for i, stats in enumerate(all_stats):
        sae_size = str(stats["sae_size"])
        positions.extend([i * 2 + 1, i * 2 + 2])
        labels.append(sae_size)

        for j, key in enumerate(["observed", "expected"]):
            data = stats[key]
            bp = ax.boxplot(
                [data["outliers"]],
                positions=[positions[-2 + j]],
                patch_artist=True,
                widths=0.6,
                showfliers=show_fliers,
            )

            for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
                plt.setp(bp[element], color=colors[j])

            for patch in bp["boxes"]:
                patch.set_facecolor(colors[j])
                patch.set_alpha(0.7)

            # Plot the median
            ax.plot(
                positions[-2 + j],
                data["median"],
                color="white",
                marker="o",
                markersize=8,
                markeredgecolor="black",
            )

    # Center the labels between each pair of boxplots
    ax.set_xticks([p + 0.5 for p in positions[::2]])
    ax.set_xticklabels(labels)
    ax.set_xlabel("SAE Size")
    ax.set_ylabel("Co-occurrence frequency (per token)")
    ax.set_title(
        f'Co-occurrence Distribution for Different SAE Sizes\n'
        f'(Activation threshold: {all_stats[0]["activation_threshold"]})'
    )

    # Add legend
    ax.plot([], [], color="blue", label="Observed")
    ax.plot([], [], color="red", label="Expected")
    ax.legend()

    plt.tight_layout()
    suffix = "w_outliers" if show_fliers else "no_outliers"
    plt.savefig(pj(output_dir, f"combined_boxplot_{suffix}.png"))
    plt.close()


def plot_feature_activation_stats(input_dir: str, output_dir: str):
    """
    Read feature activation statistics and create a plot showing mean number
    and mean fraction of active features for different SAE sizes.
    """
    raise NotImplementedError(
        "This needs to be updated to match old script, currently reports mean of 1 for everything"
    )
    # Read the CSV file
    df = pd.read_csv(pj(input_dir, "feature_activation_stats.csv"))

    # Sort by SAE size
    df = df.sort_values("sae_size")

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot mean number of active features
    color = "tab:blue"
    ax1.set_xlabel("SAE Size (log scale)")
    ax1.set_ylabel("Mean Number of Active Features", color=color)
    ax1.plot(
        df["sae_size"],
        df["mean_active_features"],
        color=color,
        marker="o",
        label="Mean Number",
    )
    ax1.fill_between(
        df["sae_size"],
        df["mean_active_features"] - df["se_active_features"],
        df["mean_active_features"] + df["se_active_features"],
        color=color,
        alpha=0.2,
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_xscale("log")

    # Create a second y-axis for mean fraction
    ax2 = ax1.twinx()
    color = "tab:orange"
    ax2.set_ylabel("Mean Fraction of Active Features", color=color)
    ax2.plot(
        df["sae_size"],
        df["mean_fraction_active"],
        color=color,
        marker="s",
        label="Mean Fraction",
    )
    ax2.fill_between(
        df["sae_size"],
        df["mean_fraction_active"] - df["se_fraction_active"],
        df["mean_fraction_active"] + df["se_fraction_active"],
        color=color,
        alpha=0.2,
    )
    ax2.tick_params(axis="y", labelcolor=color)

    # Add a title
    plt.title("Mean Number and Fraction of Active Features vs SAE Size")

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # Adjust layout and save
    fig.tight_layout()
    plt.savefig(pj(output_dir, "feature_activation_stats_plot.png"))
    plt.close()

    print(f"Feature activation statistics plot saved to {output_dir}")


def main():
    git_root = get_git_root()
    # model_name = "gpt2-small"
    # sae_release_short = "res-jb-feature-splitting"
    # sae_sizes = [768, 1536, 3072, 6144, 12288, 24576, 49152, 98304]
    
    # model_name = "gemma-2-2b"
    # sae_release_short = "gemma-scope-2b-pt-res"
    # sae_sizes = [176, 22, 41, 445, 82]
    
    model_name = "gemma-2-2b"
    sae_release_short = "gemma-scope-2b-pt-res-canonical"
    sae_sizes = [16, 32, 65]
    
    activation_thresholds = [0.0]
    n_batches = 10
    

    input_dir = pj(
        git_root, f"results/cooc/cooccurrence_analysis/{model_name}/{sae_release_short}/n_batches_{n_batches}"
    )
    output_dir = pj(
        git_root,
        f"results/cooc/cooccurrence_from_summary/{model_name}/{sae_release_short}/n_batches_{n_batches}",
    )
    os.makedirs(output_dir, exist_ok=True)

    all_stats = []

    for sae_size in tqdm(sae_sizes, desc="SAE sizes:"):
        for threshold in activation_thresholds:
            safe_threshold = str(threshold).replace(".", "_")
            stats_file = pj(input_dir, f"boxplot_stats_{sae_size}_{safe_threshold}.h5")
            if not os.path.exists(stats_file):
                stats_file = stats_file.replace('.h5', '.npz')
            
            histogram_file = pj(input_dir, f"histogram_data_{sae_size}_{safe_threshold}.h5")
            if not os.path.exists(histogram_file):
                histogram_file = histogram_file.replace('.h5', '.npz')

            if not os.path.exists(stats_file):
                print(f"Stats file not found: {stats_file}")
                continue

            print(f"Loading stats for SAE size {sae_size} and threshold {threshold}")
            stats = load_boxplot_stats(stats_file)
            print(f"Loaded stats for SAE size {sae_size} and threshold {threshold}")

            print(f"Plotting boxplots for SAE size {sae_size} and threshold {threshold}")   
            plot_boxplots(stats, output_dir, show_fliers=False)
            print(f"Plotted boxplots without outliers for SAE size {sae_size} and threshold {threshold}")
            plot_boxplots(stats, output_dir)
            print(f"Plotted boxplots for SAE size {sae_size} and threshold {threshold}")


            if os.path.exists(histogram_file):
                print(f"Loading histogram data for SAE size {sae_size} and threshold {threshold}")
                histogram_data = load_histogram_data(histogram_file)
                print(f"Loaded histogram data for SAE size {sae_size} and threshold {threshold}")
                print(f"Plotting histogram for SAE size {sae_size} and threshold {threshold}")
                plot_histogram(histogram_data, stats, output_dir)
                print(f"Plotted histogram for SAE size {sae_size} and threshold {threshold}")
            else:
                print(f"Histogram data file not found: {histogram_file}")

            # plot_ratio_boxplot(stats, output_dir)

            all_stats.append(stats)

    # Plot combined boxplots for all SAE sizes
    if len(all_stats) > 1:
        print("Plotting combined boxplots for all SAE sizes")
        plot_combined_boxplots(all_stats, output_dir)
        print("Plotted combined boxplots for all SAE sizes")
        print("Plotting combined boxplots without outliers for all SAE sizes")
        plot_combined_boxplots(all_stats, output_dir, show_fliers=False)
        print("Plotted combined boxplots without outliers for all SAE sizes")
    print("Boxplot and histogram generation complete.")

    # After calling calculate_and_save_feature_activation_stats
    # plot_feature_activation_stats(input_dir, output_dir)

    print("Feature activation statistics plot saved to " + output_dir)


if __name__ == "__main__":
    main()

