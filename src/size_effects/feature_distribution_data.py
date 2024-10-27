import os
from os.path import join as pj

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sae_lens import SAE, ActivationsStore
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer

from sae_cooccurrence.normalised_cooc_functions import get_sae_release
from sae_cooccurrence.utils.saving_loading import notify, set_device
from sae_cooccurrence.utils.set_paths import get_git_root


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


def create_cooccurrence_table(
    observed: torch.Tensor,
    expected: torch.Tensor,
    sae_size: int,
    threshold: float,
    total_tokens: int,
    chunk_size: int = 10000,
) -> pa.Table:
    total_size = observed.shape[0]
    table_list: list[pa.Table] = []

    for i in tqdm(range(0, total_size, chunk_size), desc="Creating Table"):
        end = min(i + chunk_size, total_size)

        # Get upper triangle indices for this chunk
        indices = torch.triu_indices(end - i, total_size, offset=1)
        indices[0] += i  # Adjust row indices

        # Extract values using these indices
        observed_values = observed[indices[0], indices[1]].cpu().numpy()
        expected_values = expected[indices[0], indices[1]].cpu().numpy()

        # Calculate per-token values
        observed_per_token = observed_values / total_tokens
        expected_per_token = expected_values / total_tokens
        total_tokens = total_tokens

        # Create PyArrow Table for this chunk
        table_chunk = pa.Table.from_arrays(
            [
                pa.array([sae_size] * len(observed_values)),
                pa.array([threshold] * len(observed_values)),
                pa.array(indices[0].cpu().numpy()),
                pa.array(indices[1].cpu().numpy()),
                pa.array(observed_values),
                pa.array(expected_values),
                pa.array(observed_per_token),
                pa.array(expected_per_token),
                pa.array([total_tokens] * len(observed_values)),
            ],
            names=[
                "SAE_Size",
                "Activation_Threshold",
                "Feature_1",
                "Feature_2",
                "Observed",
                "Expected",
                "Observed_Per_Token",
                "Expected_Per_Token",
                "Total_Tokens",
            ],
        )

        table_list.append(table_chunk)

    # Concatenate all chunks
    return pa.concat_tables(table_list)


def calculate_expected_cooccurrences(
    feature_occurrences: torch.Tensor, total_tokens: int, chunk_size: int = 1000
) -> torch.Tensor:
    feature_probabilities = feature_occurrences.diag() / total_tokens
    n = feature_probabilities.size(0)
    expected_cooccurrences = torch.zeros_like(feature_occurrences)

    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        chunk = feature_probabilities[i:end]
        expected_cooccurrences[i:end] = torch.outer(chunk, feature_probabilities)

    expected_cooccurrences *= total_tokens
    return expected_cooccurrences


def save_results_as_parquet(table: pa.Table, output_dir: str, sae_size: int, n_batches: int) -> None:
    # Create filename with SAE size
    parquet_filename = f"cooccurrence_analysis_results_sae_{sae_size}_nbatches_{n_batches}.parquet"
    parquet_path = pj(output_dir, parquet_filename)

    # Save as Parquet file
    pq.write_table(table, parquet_path)
    print(f"Results for SAE size {sae_size} saved as Parquet: {parquet_path}")


def main() -> None:
    device = set_device()
    git_root = get_git_root()

    # model_name = "gpt2-small"
    # sae_release_short = "res-jb-feature-splitting"
    # # sae_sizes = [768, 1536, 3072, 6144, 12288, 24576]
    # sae_sizes: list[int] = [768, 1536, 3072, 6144, 12288]
    # # sae_sizes = [24576, 49152, 98304]
    # # sae_sizes: list[int] = [768]
    # # sae_sizes = [768]
    # activation_thresholds: list[float] = [0.0]
    # n_batches = 5000
    
    model_name = "gemma-2-2b"
    sae_release_short = "gemma-scope-2b-pt-res"
    # sae_sizes = [768, 1536, 3072, 6144, 12288, 24576]
    sae_sizes: list[int] = [176, 22, 41, 445, 82]
    # sae_sizes = [24576, 49152, 98304]
    # sae_sizes: list[int] = [768]
    # sae_sizes = [768]
    activation_thresholds: list[float] = [0.0]
    n_batches = 10

    # Create output directory
    output_dir = pj(
        git_root, f"results/cooc/cooccurrence_analysis/{model_name}/{sae_release_short}"
    )
    os.makedirs(output_dir, exist_ok=True)

    for sae_size in tqdm(sae_sizes, desc="Processing SAE sizes"):
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
            n_batches_in_buffer=32,
            device=device,
        )

        sae_results: list[pa.Table] = []

        for threshold in activation_thresholds:
            # Calculate feature occurrences
            feature_occurrences, total_tokens = calculate_feature_occurrences(
                sae, activation_store, n_batches, device, threshold
            )

            # Calculate expected co-occurrences
            expected_cooccurrences = calculate_expected_cooccurrences(
                feature_occurrences, total_tokens
            )

            # Create PyArrow Table for this SAE size and threshold
            table = create_cooccurrence_table(
                feature_occurrences,
                expected_cooccurrences,
                sae_size,
                threshold,
                total_tokens,
            )
            sae_results.append(table)

        # Combine results for all thresholds of this SAE size
        sae_table = pa.concat_tables(sae_results)

        # Save results for this SAE size
        save_results_as_parquet(sae_table, output_dir, sae_size, n_batches)

        # Clear memory
        del model, sae, activation_store, sae_table
        feature_occurrences = expected_cooccurrences = None
        torch.cuda.empty_cache()

    notify("Co-occurrence analysis for multiple SAE sizes and thresholds complete.")


if __name__ == "__main__":
    main()
