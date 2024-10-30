import os
import pickle
from os.path import join as pj

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sae_lens import SAE, ActivationsStore
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer

from sae_cooccurrence.normalised_cooc_functions import get_sae_release
from sae_cooccurrence.utils.saving_loading import notify, set_device
from sae_cooccurrence.utils.set_paths import get_git_root


def get_sae_size(sae_id: str, model_name: str, sae_release_short: str) -> int | None:
    if model_name == "gpt2-small":
        return int(sae_id.split("_")[-1])
    elif model_name == "gemma-2-2b":
        if sae_release_short == "gemma-scope-2b-pt-res-canonical":
            # Extract the number from 'width_XXk' format
            width = sae_id.split("/")[1]  # gets 'width_16k'
            return int(width.split("_")[1].replace("k", "000"))
        elif sae_release_short == "gemma-scope-2b-pt-res":
            # "layer_12/width_16k/average_l0_22",
            l0 = sae_id.split("average_l0_")[1]
            return int(l0)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def calculate_firing_stats(sae, activation_store, n_batches, threshold):
    all_fractions = []
    all_raw_numbers = []

    for _ in tqdm(range(n_batches), desc="Processing batches", leave=False):
        activations_batch = activation_store.next_batch()
        feature_acts = sae.encode(activations_batch).squeeze()

        firings = (feature_acts > threshold).float()
        fraction_fired = firings.mean(dim=1)  # Mean across features for each token
        raw_number_fired = firings.sum(dim=1)  # Sum across features for each token

        all_fractions.extend(fraction_fired.tolist())
        all_raw_numbers.extend(raw_number_fired.tolist())

    return {
        "fraction": {"mean": np.mean(all_fractions), "std": np.std(all_fractions)},
        "raw_number": {
            "mean": np.mean(all_raw_numbers),
            "std": np.std(all_raw_numbers),
        },
    }


def save_results(results, summary_stats, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    full_results = {"detailed_results": results, "summary_stats": summary_stats}
    pickle_path = pj(output_dir, "sae_firing_stats_full.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(full_results, f)
    print(f"Full results saved as pickle: {pickle_path}")

    np_results = {
        "sae_sizes": np.array(list(results.keys())),
        "thresholds": np.array(list(results[list(results.keys())[0]].keys())),
        "fractions": np.array(
            [
                [
                    results[size][threshold]["fraction"]["mean"]
                    for threshold in results[size]
                ]
                for size in results
            ]
        ),
        "fractions_std": np.array(
            [
                [
                    results[size][threshold]["fraction"]["std"]
                    for threshold in results[size]
                ]
                for size in results
            ]
        ),
        "raw_numbers": np.array(
            [
                [
                    results[size][threshold]["raw_number"]["mean"]
                    for threshold in results[size]
                ]
                for size in results
            ]
        ),
        "raw_numbers_std": np.array(
            [
                [
                    results[size][threshold]["raw_number"]["std"]
                    for threshold in results[size]
                ]
                for size in results
            ]
        ),
    }
    np_path = pj(output_dir, "sae_firing_stats.npy")
    np.save(np_path, np_results)  # type: ignore
    print(f"Numpy arrays saved as: {np_path}")

    txt_path = pj(output_dir, "sae_firing_stats_results.txt")
    with open(txt_path, "w") as f:
        f.write("Results:\n")
        f.write(
            "SAE Size | Threshold | Fraction Fired (mean ± std) | Avg Features Fired (mean ± std)\n"
        )
        f.write("-" * 80 + "\n")
        for sae_size in sorted(results.keys()):
            for threshold in results[sae_size]:
                fraction_mean = results[sae_size][threshold]["fraction"]["mean"]
                fraction_std = results[sae_size][threshold]["fraction"]["std"]
                raw_mean = results[sae_size][threshold]["raw_number"]["mean"]
                raw_std = results[sae_size][threshold]["raw_number"]["std"]
                f.write(
                    f"{sae_size:8d} | {threshold:9.2f} | {fraction_mean:10.6f} ± {fraction_std:10.6f} | {raw_mean:10.2f} ± {raw_std:10.2f}\n"
                )

        f.write("\nSummary Statistics:\n")
        for threshold in summary_stats:
            f.write(f"\nThreshold: {threshold}\n")
            for stat_type in ["fraction", "raw_number"]:
                f.write(f"  {stat_type.capitalize()}:\n")
                for stat, value in summary_stats[threshold][stat_type].items():
                    f.write(f"    {stat}: {value:.6f}\n")
    print(f"Printed results saved as: {txt_path}")


def plot_results(
    results,
    output_dir,
    sae_release_short: str,
    joint_plot_threshold=0.0,
    plot_errors=True,
):
    sns.set_theme()
    sae_sizes = sorted(results.keys())
    thresholds = sorted(results[sae_sizes[0]].keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    for threshold in thresholds:
        fractions = [results[size][threshold]["fraction"]["mean"] for size in sae_sizes]
        fractions_std = [
            results[size][threshold]["fraction"]["std"] for size in sae_sizes
        ]
        raw_numbers = [
            results[size][threshold]["raw_number"]["mean"] for size in sae_sizes
        ]
        raw_numbers_std = [
            results[size][threshold]["raw_number"]["std"] for size in sae_sizes
        ]

        if plot_errors:
            ax1.errorbar(
                sae_sizes,
                fractions,
                yerr=fractions_std,
                fmt="o-",
                capsize=5,
                label=f"Threshold {threshold}",
            )
            ax2.errorbar(
                sae_sizes,
                raw_numbers,
                yerr=raw_numbers_std,
                fmt="o-",
                capsize=5,
                label=f"Threshold {threshold}",
            )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("SAE Size")
    ax1.set_ylabel("Fraction of Features Fired")
    ax1.set_title("Fraction of SAE Features Fired vs SAE Size")
    ax1.legend()
    ax1.grid(True)

    ax2.set_xscale("log")
    ax2.set_xlabel("SAE Size")
    ax2.set_ylabel("Average Number of Features Fired")
    ax2.set_title("Average Number of SAE Features Fired vs SAE Size")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plot_path = pj(output_dir, "sae_firing_stats_plot.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved as: {plot_path}")
    plt.close()

    # New plot for threshold 0.0
    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax2 = ax1.twinx()

    threshold = joint_plot_threshold
    fractions = [results[size][threshold]["fraction"]["mean"] for size in sae_sizes]
    raw_numbers = [results[size][threshold]["raw_number"]["mean"] for size in sae_sizes]

    ax1.plot(sae_sizes, fractions, "b-", marker="o", label="Fraction Fired")
    ax2.plot(sae_sizes, raw_numbers, "r-", marker="s", label="Number Fired")

    ax1.set_xscale("log")
    if sae_release_short == "gemma-scope-2b-pt-res":
        ax1.set_xlabel("L0")
    else:
        ax1.set_xlabel("SAE Size")
    ax1.set_ylabel("Fraction of Features Fired", color="b")
    ax2.set_ylabel("Number of Features Fired", color="r")

    ax1.tick_params(axis="y", labelcolor="b")
    ax2.tick_params(axis="y", labelcolor="r")

    plt.title("Fraction and Number of SAE Features Fired vs SAE Size (Threshold 0.0)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    combined_plot_path = pj(output_dir, "sae_firing_stats_combined_plot.png")
    plt.savefig(combined_plot_path, dpi=300)
    print(f"Combined plot saved as: {combined_plot_path}")
    plt.close()


def load_existing_results(output_dir):
    pickle_path = pj(output_dir, "sae_firing_stats_full.pkl")
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            return pickle.load(f)
    return None


def process_single_sae(
    sae_id: str,
    model_name: str,
    sae_release_short: str,
    model: HookedTransformer,
    activation_thresholds: list[float],
    n_batches: int,
    n_batches_in_buffer: int,
    device: str,
) -> tuple[int | None, dict]:
    """Process a single SAE and calculate firing statistics.

    Args:
        sae_id: ID of the SAE to process
        model_name: Name of the model
        sae_release_short: Short name of the SAE release
        model: The transformer model
        activation_thresholds: List of thresholds for activation
        n_batches: Number of batches to process
        n_batches_in_buffer: Number of batches to keep in buffer
        device: Device to use for computation

    Returns:
        Tuple of (sae_size, results_dict)
    """
    sae_size = get_sae_size(sae_id, model_name, sae_release_short)
    sae_release = get_sae_release(model_name, sae_release_short)
    sae, _, _ = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=device,
    )

    activation_store = ActivationsStore.from_sae(
        model=model,
        sae=sae,
        streaming=True,
        store_batch_size_prompts=8,
        train_batch_size_tokens=4096,
        n_batches_in_buffer=n_batches_in_buffer,
        device=device,
    )

    results_dict = {}
    for threshold in activation_thresholds:
        results_dict[threshold] = calculate_firing_stats(
            sae,
            activation_store,
            n_batches,
            threshold,
        )

    return sae_size, results_dict


def calculate_summary_statistics(results: dict, threshold: float) -> dict:
    """Calculate summary statistics for a given threshold across all SAE sizes.

    Args:
        results: Dictionary of results for each SAE size
        threshold: Activation threshold value

    Returns:
        Dictionary containing summary statistics for fractions and raw numbers
    """
    return {
        "fraction": {
            "mean": np.mean(
                [results[size][threshold]["fraction"]["mean"] for size in results]
            ),
            "median": np.median(
                [results[size][threshold]["fraction"]["mean"] for size in results]
            ),
            "min": np.min(
                [results[size][threshold]["fraction"]["mean"] for size in results]
            ),
            "max": np.max(
                [results[size][threshold]["fraction"]["mean"] for size in results]
            ),
        },
        "raw_number": {
            "mean": np.mean(
                [results[size][threshold]["raw_number"]["mean"] for size in results]
            ),
            "median": np.median(
                [results[size][threshold]["raw_number"]["mean"] for size in results]
            ),
            "min": np.min(
                [results[size][threshold]["raw_number"]["mean"] for size in results]
            ),
            "max": np.max(
                [results[size][threshold]["raw_number"]["mean"] for size in results]
            ),
        },
    }


def process_model_sae_stats(
    model_name: str,
    sae_release_short: str,
    sae_ids: list[str],
    activation_thresholds: list[float] = [0.0, 0.1, 0.5, 1.5],
    joint_plot_threshold: float = 0.0,
    n_batches: int = 10,
    n_batches_in_buffer: int = 4,
) -> None:
    """Process SAE statistics for a given model and SAE configuration.

    Args:
        model_name: Name of the model (e.g., "gpt2-small", "gemma-2b")
        sae_release_short: Short name of the SAE release
        sae_ids: List of SAE IDs to process
        activation_thresholds: Thresholds for feature activation
        joint_plot_threshold: Threshold for joint plot
        n_batches: Number of batches to process
        n_batches_in_buffer: Number of batches to keep in buffer
    """
    device = set_device()
    git_root = get_git_root()

    output_dir = pj(
        git_root,
        "results",
        "cooc",
        "size_effects",
        model_name,
        sae_release_short,
        "sae_firing_stats",
    )

    # Check for existing results
    existing_results = load_existing_results(output_dir)
    if existing_results:
        user_input = input(
            "Existing results found. Do you want to load and plot them? (y/n): "
        )
        if user_input.lower() == "y":
            results = existing_results["detailed_results"]
            summary_stats = existing_results["summary_stats"]
            plot_results(
                results,
                output_dir,
                joint_plot_threshold=joint_plot_threshold,
                sae_release_short=sae_release_short,
            )
            print("Existing results loaded and plotted.")
            return

    # If no existing results or user chose to recalculate
    model = HookedTransformer.from_pretrained(model_name, device=device)
    results = {}

    for sae_id in tqdm(sae_ids, desc="Processing SAE IDs"):
        sae_size, sae_results = process_single_sae(
            sae_id=sae_id,
            model_name=model_name,
            sae_release_short=sae_release_short,
            model=model,
            activation_thresholds=activation_thresholds,
            n_batches=n_batches,
            n_batches_in_buffer=n_batches_in_buffer,
            device=device,
        )
        results[sae_size] = sae_results

    # Calculate summary statistics
    summary_stats = {}
    for threshold in activation_thresholds:
        summary_stats[threshold] = calculate_summary_statistics(results, threshold)

    # Save and plot results
    save_results(results, summary_stats, output_dir)
    plot_results(
        results,
        output_dir,
        joint_plot_threshold=joint_plot_threshold,
        sae_release_short=sae_release_short,
    )


def main():
    torch.set_grad_enabled(False)

    # Process GPT-2
    process_model_sae_stats(
        model_name="gpt2-small",
        sae_release_short="res-jb-feature-splitting",
        sae_ids=[
            "blocks.8.hook_resid_pre_768",
            "blocks.8.hook_resid_pre_1536",
            "blocks.8.hook_resid_pre_3072",
            "blocks.8.hook_resid_pre_6144",
            "blocks.8.hook_resid_pre_12288",
            "blocks.8.hook_resid_pre_24576",
            "blocks.8.hook_resid_pre_49152",
            "blocks.8.hook_resid_pre_98304",
        ],
    )

    # Process Gemma
    process_model_sae_stats(
        model_name="gemma-2-2b",
        sae_release_short="gemma-scope-2b-pt-res-canonical",
        sae_ids=[
            "layer_12/width_16k/canonical",
            "layer_12/width_32k/canonical",
            "layer_12/width_65k/canonical",
        ],
    )

    process_model_sae_stats(
        model_name="gemma-2-2b",
        sae_release_short="gemma-scope-2b-pt-res",
        sae_ids=[
            "layer_12/width_16k/average_l0_22",
            "layer_12/width_16k/average_l0_41",
            "layer_12/width_16k/average_l0_82",
            "layer_12/width_16k/average_l0_176",
            "layer_12/width_16k/average_l0_445",
        ],
    )

    notify("SAE firing statistics complete")


if __name__ == "__main__":
    main()
    notify("SAE firing statistics complete")
