import os
import warnings
import zipfile
from os.path import join as pj

import h5py
import numpy as np
import pandas as pd
import torch
from sae_lens import SAE, ActivationsStore
from scipy import stats
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer

from sae_cooccurrence.normalised_cooc_functions import get_sae_release
from sae_cooccurrence.utils.saving_loading import notify, set_device
from sae_cooccurrence.utils.set_paths import get_git_root


def save_tensor_to_h5(
    tensor: torch.Tensor, filename: str, dataset_name: str, compression: str = "gzip"
) -> None:
    """
    Save a PyTorch tensor to an HDF5 file with proper error handling and compression.

    Args:
    tensor (torch.Tensor): The tensor to save.
    filename (str): The name of the HDF5 file to create or update.
    dataset_name (str): The name of the dataset within the HDF5 file.
    compression (str): The compression filter to use (default: "gzip").
    """
    try:
        with h5py.File(filename, "w") as f:
            # Convert tensor to numpy and save as a dataset
            f.create_dataset(
                dataset_name, data=tensor.cpu().numpy(), compression=compression
            )
        print(f"Successfully saved {dataset_name} to {filename}")
    except Exception as e:
        print(f"Error saving tensor to HDF5: {e}")


def save_tensor_to_npz(tensor: torch.Tensor, filename: str) -> None:
    """
    Save a PyTorch tensor to a compressed numpy npz file.

    Args:
    tensor (torch.Tensor): The tensor to save.
    filename (str): The name of the npz file to create.
    """
    try:
        np.savez_compressed(filename, data=tensor.cpu().numpy())
        print(f"Successfully saved tensor to {filename}")
    except Exception as e:
        print(f"Error saving tensor to npz: {e}")


def calculate_feature_occurrences(
    sae: SAE,
    activation_store: ActivationsStore,
    n_batches: int,
    device: str,
    activation_threshold: float,
    chunk_size: int = 1000,
) -> tuple[torch.Tensor, int]:
    feature_occurrences = torch.zeros((sae.cfg.d_sae, sae.cfg.d_sae), device=device)
    total_tokens = 0

    for _ in tqdm(range(n_batches), desc="Processing batches", leave=False):
        activations_batch = activation_store.next_batch()
        feature_acts = sae.encode(activations_batch).squeeze()

        feature_activations = (feature_acts > activation_threshold).float()

        # Process in chunks
        for i in range(0, feature_activations.shape[1], chunk_size):
            chunk = feature_activations[:, i : i + chunk_size]
            chunk_occurrences = torch.matmul(chunk.T, feature_activations)
            feature_occurrences[i : i + chunk_size] += chunk_occurrences

        total_tokens += feature_acts.shape[0]

    return feature_occurrences, total_tokens


def calculate_expected_cooccurrences(
    feature_occurrences: torch.Tensor, total_tokens: int, chunk_size: int = 1000
) -> torch.Tensor:
    feature_probabilities = feature_occurrences.diag() / total_tokens
    n = feature_probabilities.size(0)
    expected_cooccurrences = torch.zeros_like(feature_occurrences)

    for i in tqdm(range(0, n, chunk_size), desc="Calculating expected co-occurrences", leave=False):
        end = min(i + chunk_size, n)
        chunk = feature_probabilities[i:end]
        expected_cooccurrences[i:end] = torch.outer(chunk, feature_probabilities)

    expected_cooccurrences *= total_tokens
    return expected_cooccurrences


def calculate_boxplot_stats(tensor: torch.Tensor, chunk_size: int = 1_000_000) -> dict:
    """Calculate summary statistics for a boxplot using chunked processing."""
    flattened = tensor.flatten()
    n = flattened.numel()
    
    # Initialize variables to store quantiles
    q1, median, q3 = 0.0, 0.0, 0.0
    
    # Process in chunks
    for i in tqdm(range(0, n, chunk_size), desc="Calculating quantiles", leave=False):
        chunk = flattened[i:i+chunk_size]
        q1 += torch.quantile(chunk, 0.25).item()
        median += torch.quantile(chunk, 0.5).item()
        q3 += torch.quantile(chunk, 0.75).item()
    
    # Calculate average quantiles
    num_chunks = (n + chunk_size - 1) // chunk_size
    q1 /= num_chunks
    median /= num_chunks
    q3 /= num_chunks
    
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    
    # Calculate outliers in chunks
    outliers = []
    for i in tqdm(range(0, n, chunk_size), desc="Calculating outliers", leave=False):
        chunk = flattened[i:i+chunk_size]
        chunk_outliers = chunk[(chunk < lower_fence) | (chunk > upper_fence)]
        outliers.append(chunk_outliers)
    
    outliers = torch.cat(outliers).cpu().numpy()

    return {
        "median": median,
        "q1": q1,
        "q3": q3,
        "whiskers": (lower_fence, upper_fence),
        "outliers": outliers,
    }


def generate_data_tensors(
    model_name,
    sae_release_short,
    sae_sizes,
    activation_thresholds,
    n_batches,
    n_batches_in_buffer,
    device,
    output_dir,
    save_npz: bool = False,
    save_h5: bool = True,  # New parameter
):
    for sae_size in tqdm(sae_sizes, desc="Generating data tensors"):
        
        if model_name == "gpt2-small":
            sae_id = f"blocks.8.hook_resid_pre_{sae_size}"
        elif sae_release_short == "gemma-scope-2b-pt-res":
            sae_id = f"layer_12/width_16k/average_l0_{sae_size}"
        elif sae_release_short == "gemma-scope-2b-pt-res-canonical":
            sae_id = f"layer_12/width_{sae_size}k/canonical"
        else:
            raise ValueError(f"Unknown model or SAE release: {model_name} {sae_release_short}")

        # Load model and SAE
        model = HookedTransformer.from_pretrained(model_name, device=device)
        release = get_sae_release(model_name, sae_release_short)

        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release=release, sae_id=sae_id, device=device
        )

        # Set up the activations store
        activation_store = ActivationsStore.from_sae(
            model=model,
            sae=sae,
            streaming=True,
            store_batch_size_prompts=8,
            train_batch_size_tokens=4096,
            n_batches_in_buffer=n_batches_in_buffer,
            device=device,
        )

        for threshold in activation_thresholds:
            safe_threshold = str(threshold).replace(".", "_")

            # Calculate feature occurrences
            print(f"Calculating feature occurrences for SAE size {sae_size} and threshold {threshold}")
            feature_occurrences, total_tokens = calculate_feature_occurrences(
                sae, activation_store, n_batches, device, threshold
            )

            # Save feature occurrences file as HDF5 if requested
            if save_h5:
                feature_occurrences_file = pj(
                    output_dir, f"feature_occurrences_{sae_size}_{safe_threshold}.h5"
                )
                save_tensor_to_h5(
                    feature_occurrences, feature_occurrences_file, "feature_occurrences"
                )

                # Save total_tokens as a separate dataset in the same file
                with h5py.File(feature_occurrences_file, "a") as f:
                    f.create_dataset("total_tokens", data=total_tokens)

            # Save as npz if requested
            if save_npz:
                npz_file = pj(
                    output_dir, f"feature_occurrences_{sae_size}_{safe_threshold}.npz"
                )
                save_tensor_to_npz(feature_occurrences, npz_file)

            # Calculate expected co-occurrences
            print(f"Calculating expected co-occurrences for SAE size {sae_size} and threshold {threshold}")
            expected_cooccurrences = calculate_expected_cooccurrences(
                feature_occurrences, total_tokens
            )

            # Save expected co-occurrences file as HDF5 if requested
            if save_h5:
                expected_cooccurrences_file = pj(
                    output_dir, f"expected_cooccurrences_{sae_size}_{safe_threshold}.h5"
                )
                save_tensor_to_h5(
                    expected_cooccurrences,
                    expected_cooccurrences_file,
                    "expected_cooccurrences",
                )

            # Save expected co-occurrences as npz if requested
            if save_npz:
                print(f"Saving expected co-occurrences as npz for SAE size {sae_size} and threshold {threshold} (can be very slow)")
                npz_file = pj(
                    output_dir, f"expected_cooccurrences_{sae_size}_{safe_threshold}.npz"
                )
                save_tensor_to_npz(expected_cooccurrences, npz_file)

        # Clear memory
        del model, sae, activation_store
        feature_occurrences = expected_cooccurrences = None
        torch.cuda.empty_cache()


def process_data_tensors(sae_sizes, activation_thresholds, output_dir):
    for sae_size in tqdm(sae_sizes, desc="Processing data tensors"):
        for threshold in activation_thresholds:
            safe_threshold = str(threshold).replace(".", "_")

            # Try loading from NPZ first, then H5
            feature_occurrences_npz = pj(
                output_dir, f"feature_occurrences_{sae_size}_{safe_threshold}.npz"
            )
            feature_occurrences_h5 = pj(
                output_dir, f"feature_occurrences_{sae_size}_{safe_threshold}.h5"
            )

            if os.path.exists(feature_occurrences_npz):
                with np.load(feature_occurrences_npz) as data:
                    feature_occurrences = torch.tensor(data['data'])
                # Load total_tokens from a separate file or recalculate it
                total_tokens = feature_occurrences.sum().item()
            elif os.path.exists(feature_occurrences_h5):
                with h5py.File(feature_occurrences_h5, "r") as f:
                    feature_occurrences = torch.tensor(np.array(f["feature_occurrences"]))
                    total_tokens = f["total_tokens"][()]  # type: ignore
            else:
                raise FileNotFoundError(f"No data file found for SAE size {sae_size} and threshold {threshold}")

            # Similar process for expected co-occurrences
            expected_cooccurrences_npz = pj(
                output_dir, f"expected_cooccurrences_{sae_size}_{safe_threshold}.npz"
            )
            expected_cooccurrences_h5 = pj(
                output_dir, f"expected_cooccurrences_{sae_size}_{safe_threshold}.h5"
            )

            # Initialize expected_cooccurrences
            expected_cooccurrences = None

            # Load expected co-occurrences
            if os.path.exists(expected_cooccurrences_npz):
                if zipfile.is_zipfile(expected_cooccurrences_npz):
                    with np.load(expected_cooccurrences_npz) as data:
                        expected_cooccurrences = torch.tensor(data['data'])
                else:
                    warnings.warn("NPZ file is not a valid zip archive")
            elif os.path.exists(expected_cooccurrences_h5):
                with h5py.File(expected_cooccurrences_h5, "r") as f:
                    expected_cooccurrences = torch.tensor(np.array(f["expected_cooccurrences"]))

            if expected_cooccurrences is None:
                raise FileNotFoundError(f"No expected co-occurrences file found for SAE size {sae_size} and threshold {threshold}")

            # Calculate summary statistics
            observed_stats = calculate_boxplot_stats(feature_occurrences)
            expected_stats = calculate_boxplot_stats(expected_cooccurrences)

            # Normalize by total tokens
            for summary_stats in [observed_stats, expected_stats]:
                for key in ["median", "q1", "q3", "whiskers"]:
                    if isinstance(summary_stats[key], tuple):
                        summary_stats[key] = tuple(v / total_tokens for v in summary_stats[key])
                    else:
                        summary_stats[key] /= total_tokens
                summary_stats["outliers"] /= total_tokens

            # Save summary statistics as h5 file
            stats_file = pj(output_dir, f"boxplot_stats_{sae_size}_{safe_threshold}.h5")
            with h5py.File(stats_file, "w") as f:
                observed_group = f.create_group("observed")
                expected_group = f.create_group("expected")

                for group, summary_stats in [
                    (observed_group, observed_stats),
                    (expected_group, expected_stats),
                ]:
                    for key, value in summary_stats.items():
                        if key == "whiskers":
                            group.create_dataset(key, data=np.array(value))
                        elif key == "outliers":
                            group.create_dataset(key, data=value, compression="gzip")
                        else:
                            group.create_dataset(key, data=value)

                f.create_dataset("total_tokens", data=total_tokens)
                f.create_dataset("sae_size", data=sae_size)
                f.create_dataset("activation_threshold", data=threshold)

            print(f"Boxplot statistics saved to {stats_file}")

            # Calculate histogram data for observed and expected co-occurrences
            (
                observed_bin_edges,
                observed_density,
                observed_log_bin_edges,
                observed_log_density,
            ) = generate_histogram_data(feature_occurrences)
            (
                expected_bin_edges,
                expected_density,
                expected_log_bin_edges,
                expected_log_density,
            ) = generate_histogram_data(expected_cooccurrences)

            # Save histogram data
            histogram_file = pj(
                output_dir, f"histogram_data_{sae_size}_{safe_threshold}.h5"
            )
            save_histogram_data_npz(
                observed_bin_edges,
                observed_density,
                observed_log_bin_edges,
                observed_log_density,
                expected_bin_edges,
                expected_density,
                expected_log_bin_edges,
                expected_log_density,
                histogram_file,
            )
            print(f"Histogram data saved to {histogram_file}")

        # Clear memory
        feature_occurrences = expected_cooccurrences = None
        torch.cuda.empty_cache()


def generate_histogram_data(
    tensor: torch.Tensor, num_bins: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Flatten the tensor
    values = tensor.flatten()

    # Calculate histogram for non-log data
    hist, bin_edges = torch.histogram(values, bins=num_bins)

    # Calculate histogram for log data
    log_values = torch.log10(values)
    log_values = log_values[torch.isfinite(log_values)]
    log_hist, log_bin_edges = torch.histogram(log_values, bins=num_bins)

    # Convert to numpy and calculate densities
    hist = hist.cpu().numpy()
    bin_edges = bin_edges.cpu().numpy()
    density = hist / hist.sum()

    log_hist = log_hist.cpu().numpy()
    log_bin_edges = log_bin_edges.cpu().numpy()
    log_density = log_hist / log_hist.sum()

    return bin_edges[:-1], density, log_bin_edges[:-1], log_density

def save_histogram_data_npz(
    observed_bin_edges: np.ndarray,
    observed_density: np.ndarray,
    observed_log_bin_edges: np.ndarray,
    observed_log_density: np.ndarray,
    expected_bin_edges: np.ndarray,
    expected_density: np.ndarray,
    expected_log_bin_edges: np.ndarray,
    expected_log_density: np.ndarray,
    filename: str,
) -> None:
    np.savez_compressed(
        filename,
        observed_bin_edges=observed_bin_edges,
        observed_density=observed_density,
        observed_log_bin_edges=observed_log_bin_edges,
        observed_log_density=observed_log_density,
        expected_bin_edges=expected_bin_edges,
        expected_density=expected_density,
        expected_log_bin_edges=expected_log_bin_edges,
        expected_log_density=expected_log_density
    )

def save_histogram_data_h5(
    observed_bin_edges: np.ndarray,
    observed_density: np.ndarray,
    observed_log_bin_edges: np.ndarray,
    observed_log_density: np.ndarray,
    expected_bin_edges: np.ndarray,
    expected_density: np.ndarray,
    expected_log_bin_edges: np.ndarray,
    expected_log_density: np.ndarray,
    filename: str,
) -> None:
    with h5py.File(filename, "w") as f:
        observed_group = f.create_group("observed")
        observed_group.create_dataset("bin_edges", data=observed_bin_edges)
        observed_group.create_dataset("density", data=observed_density)
        observed_group.create_dataset("log_bin_edges", data=observed_log_bin_edges)
        observed_group.create_dataset("log_density", data=observed_log_density)

        expected_group = f.create_group("expected")
        expected_group.create_dataset("bin_edges", data=expected_bin_edges)
        expected_group.create_dataset("density", data=expected_density)
        expected_group.create_dataset("log_bin_edges", data=expected_log_bin_edges)
        expected_group.create_dataset("log_density", data=expected_log_density)


def calculate_and_save_feature_activation_stats(
    sae_sizes, activation_thresholds, output_dir
):
    raise NotImplementedError(
        "This needs to be updated to match old script, currently reports mean of 1 for everything"
    )
    results = []

    for sae_size in sae_sizes:
        for threshold in activation_thresholds:
            safe_threshold = str(threshold).replace(".", "_")
            feature_occurrences_file = pj(
                output_dir, f"feature_occurrences_{sae_size}_{safe_threshold}.h5"
            )

            with h5py.File(feature_occurrences_file, "r") as f:
                feature_occurrences = torch.tensor(np.array(f["feature_occurrences"]))
                total_tokens = f["total_tokens"][()]

            # Calculate the number of active features for each token
            # TODO: This needs to be updated to use the adjusted activation threshold for Gemma
            active_features = (feature_occurrences.diag() > 0).float()

            # Calculate mean and standard error
            mean_active = active_features.mean().item() / total_tokens
            se_active = stats.sem(active_features.cpu().numpy())

            # Calculate mean fraction and its standard error
            mean_fraction = mean_active / sae_size
            se_fraction = se_active / sae_size

            results.append(
                {
                    "sae_size": sae_size,
                    "activation_threshold": threshold,
                    "mean_active_features": mean_active,
                    "se_active_features": se_active,
                    "mean_fraction_active": mean_fraction,
                    "se_fraction_active": se_fraction,
                    "total_tokens": total_tokens,
                }
            )

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(results)
    csv_file = pj(output_dir, "feature_activation_stats.csv")
    df.to_csv(csv_file, index=False)
    print(f"Feature activation statistics saved to {csv_file}")


def main() -> None:
    device = set_device()
    git_root = get_git_root()

    # model_name = "gpt2-small"
    # sae_release_short = "res-jb-feature-splitting"
    # # sae_sizes = [98304]
    # # sae_sizes = [49152, 98304]
    # sae_sizes = [98304]
    # activation_thresholds: list[float] = [0.0]
    # n_batches = 1000
    
    model_name = "gemma-2-2b"
    sae_release_short = "gemma-scope-2b-pt-res"
    # sae_sizes = [768, 1536, 3072, 6144, 12288, 24576]
    sae_sizes: list[int] = [176, 22, 41, 445, 82]
    # sae_sizes = [24576, 49152, 98304]
    # sae_sizes: list[int] = [768]
    # sae_sizes = [768]
    activation_thresholds: list[float] = [0.0]
    n_batches = 10
    
    

    
    
    n_batches_in_buffer = 4    
    regen_data = False
    save_npz = True
    save_h5 = False # this only controls the overall feature cooccurrence and expected cooccurrence files as they are much larger as h5

    # Create output directory
    output_dir = pj(
        git_root, f"results/cooc/cooccurrence_analysis/{model_name}/{sae_release_short}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Generate all data tensors first
    if regen_data:
        print("Generating data tensors")
        generate_data_tensors(
            model_name,
            sae_release_short,
            sae_sizes,
            activation_thresholds,
            n_batches,
            n_batches_in_buffer,
            device,
            output_dir,
            save_npz,
            save_h5,
        )
        print("Generated data tensors")
    # Process data tensors and create summary statistics
    print("Processing data tensors")
    process_data_tensors(sae_sizes, activation_thresholds, output_dir)
    print("Processed data tensors")
    # calculate_and_save_feature_activation_stats(sae_sizes, activation_thresholds, output_dir)

    notify("Co-occurrence analysis and summary statistics generation complete.")


if __name__ == "__main__":
    main()
