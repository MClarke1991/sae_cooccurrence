import logging
import os
from os.path import join as pj

import numpy as np
import toml
import torch
from scipy import sparse
from tqdm.auto import tqdm

from sae_cooccurrence.graph_generation import (
    find_threshold,
    remove_low_weight_edges,
    remove_self_loops_inplace,
)
from sae_cooccurrence.normalised_cooc_functions import (
    get_sae_release,
    neat_sae_id,
    setup_logging,
)
from sae_cooccurrence.utils.saving_loading import (
    load_npz_files,
    log_config_variables,
    notify,
)
from sae_cooccurrence.utils.set_paths import get_git_root


def process_sae_for_graph(sae_id: str, config: dict) -> None:
    """
    Processes the normalised feature co-occurrence matrices for a given SAE ID.
    Only generates and saves thresholded matrices.
    """
    sae_id_neat = neat_sae_id(sae_id)
    sae_release = get_sae_release(
        config["generation"]["model_name"], config["generation"]["sae_release_short"]
    )
    results_dir = (
        f"results/{config['generation']['model_name']}/{sae_release}/{sae_id_neat}"
    )
    results_path = pj(get_git_root(), results_dir)

    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"The results path '{results_path}' does not exist. Check correctly specified/that data generation has been run."
        )

    setup_logging(
        results_path,
        config["generation"]["model_name"],
        config["generation"]["sae_release_short"],
        sae_id_neat,
        context="graph_gen",
    )
    log_start_info(config, sae_id)

    matrices = load_npz_files(results_path, "feature_acts_cooc_jaccard")
    remove_self_loops_inplace(matrices)

    edge_thresholds = {
        key: find_threshold(
            matrix,
            min_size=config["analysis"]["min_subgraph_size"],
            max_size=config["analysis"]["max_subgraph_size"],
        )
        for key, matrix in matrices.items()
    }

    save_edge_thresholds(edge_thresholds, results_path)

    thresholded_matrices = {
        key: remove_low_weight_edges(matrix, edge_thresholds.get(key)[0])  # type: ignore
        for key, matrix in matrices.items()
    }
    save_thresholded_matrices(thresholded_matrices, results_path)

    logging.info("Thresholded matrices generation complete")


def log_start_info(config: dict, sae_id: str) -> None:
    logging.info("Script started running")
    logging.info(
        f"Variables - model_name: {config['generation']['model_name']}, sae_release_short: {config['generation']['sae_release_short']}, sae_id: {sae_id}"
    )
    logging.info(f"Random seed: {config['analysis']['random_seed']}")
    log_config_variables(config)


def save_thresholded_matrices(thresholded_matrices: dict, results_path: str) -> None:
    """Save thresholded matrices to compressed npz files and sparse matrices in a subdirectory."""
    thresholded_matrices_dir = os.path.join(results_path, "thresholded_matrices")
    sparse_matrices_dir = os.path.join(results_path, "sparse_matrices")
    os.makedirs(thresholded_matrices_dir, exist_ok=True)
    os.makedirs(sparse_matrices_dir, exist_ok=True)

    for threshold, matrix in tqdm(
        thresholded_matrices.items(), leave=False, desc="Saving matrices"
    ):
        filepath_safe_threshold = str(threshold).replace(".", "_")

        # Save dense matrix
        np.savez_compressed(
            os.path.join(
                thresholded_matrices_dir,
                f"thresholded_matrix_{filepath_safe_threshold}.npz",
            ),
            matrix,
        )

        # Convert to sparse and save
        sparse_matrix = sparse.csr_matrix(matrix)
        sparse.save_npz(
            os.path.join(
                sparse_matrices_dir,
                f"sparse_matrix_{filepath_safe_threshold}.npz",
            ),
            sparse_matrix,
        )

    logging.info(
        "Matrices saved in 'thresholded_matrices' and 'sparse_matrices' subdirectories."
    )


def save_edge_thresholds(edge_thresholds: dict, results_path: str) -> None:
    """Save edge thresholds to a CSV file."""
    import pandas as pd

    thresholds_data = [
        {"activation_threshold": k, "edge_threshold": v[0]}
        for k, v in edge_thresholds.items()
    ]
    thresholds_df = pd.DataFrame(thresholds_data)
    thresholds_df.to_csv(f"{results_path}/edge_thresholds.csv", index=False)
    logging.info("Edge thresholds saved.")


def main():
    torch.set_grad_enabled(False)
    git_root = get_git_root()

    config = toml.load(pj(git_root, "src", "config_gemma.toml"))

    for sae_id in tqdm(config["generation"]["sae_ids"], desc="Processing SAE IDs"):
        process_sae_for_graph(sae_id, config)

    notify("Thresholded matrices generation complete.")


if __name__ == "__main__":
    main()
