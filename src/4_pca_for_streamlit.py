import logging
import os
from os.path import join as pj

import h5py
import numpy as np
import pandas as pd
import torch
from sae_lens import ActivationsStore
from tqdm.autonotebook import tqdm

from sae_cooccurrence.normalised_cooc_functions import (
    create_results_dir,
    get_sae_release,
    neat_sae_id,
)
from sae_cooccurrence.pca import perform_pca_on_results, process_examples
from sae_cooccurrence.streamlit import load_streamlit_config
from sae_cooccurrence.utils.saving_loading import load_model_and_sae, set_device
from sae_cooccurrence.utils.set_paths import get_git_root


def process_graph_for_pca(
    model,
    sae,
    activation_store,
    fs_splitting_nodes,
    n_batches_reconstruction,
    remove_special_tokens,
    device,
    max_examples=5_000_000,
    trim_excess=False,
):
    # First attempt with original n_batches
    results = process_examples(
        activation_store,
        model,
        sae,
        fs_splitting_nodes,
        n_batches_reconstruction,
        remove_special_tokens,
        device=device,
        max_examples=max_examples,
        trim_excess=trim_excess,
    )
    pca_df, _ = perform_pca_on_results(results, n_components=3, method="auto")

    # If PCA failed, retry with double n_batches
    if pca_df is None:
        logging.warning(
            f"PCA failed, retrying with {n_batches_reconstruction * 2} batches"
        )
        results = process_examples(
            activation_store,
            model,
            sae,
            fs_splitting_nodes,
            n_batches_reconstruction * 2,
            remove_special_tokens,
            device=device,
            max_examples=max_examples,
            trim_excess=trim_excess,
        )
        pca_df, _ = perform_pca_on_results(results, n_components=3, method="auto")

    return results, pca_df


def save_results_to_hdf5(file_path, results_dict, save_options):
    with h5py.File(file_path, "w") as f:
        for subgraph_id, (results, pca_df) in results_dict.items():
            group = f.create_group(f"subgraph_{subgraph_id}")

            # Save each component based on config options
            if save_options["fired_tokens"]:
                group.create_dataset(
                    "all_fired_tokens",
                    data=np.array(
                        [token.encode("utf-8") for token in results.all_fired_tokens]
                    ),
                )

            if save_options["top_3_tokens"]:
                group.create_dataset(
                    "top_3_tokens",
                    data=np.array(
                        [str(token).encode("utf-8") for token in results.top_3_tokens]
                    ),
                )

            if save_options["context"]:
                group.create_dataset(
                    "example_context",
                    data=np.array(
                        [
                            str(context).encode("utf-8")
                            for context in results.example_context
                        ]
                    ),
                )

            if save_options["reconstructions"]:
                group.create_dataset(
                    "all_reconstructions",
                    data=results.all_reconstructions.cpu().numpy(),
                )

            if save_options["graph_feature_acts"]:
                group.create_dataset(
                    "all_graph_feature_acts",
                    data=results.all_graph_feature_acts.cpu().numpy(),
                )

            if save_options["feature_acts"]:
                group.create_dataset(
                    "all_feature_acts", data=results.all_feature_acts.cpu().numpy()
                )

            if save_options["max_feature_info"]:
                group.create_dataset(
                    "all_max_feature_info",
                    data=results.all_max_feature_info.cpu().numpy(),
                )

            if save_options["examples_found"]:
                group.create_dataset(
                    "all_examples_found", data=results.all_examples_found
                )

            # Save all_token_dfs as a table
            if save_options["token_dfs"]:
                token_dfs_group = group.create_group("all_token_dfs")
                for column in results.all_token_dfs.columns:
                    if results.all_token_dfs[column].dtype == "object":
                        data = np.array(
                            [
                                str(item).encode("utf-8")
                                for item in results.all_token_dfs[column].values
                            ]
                        )
                        token_dfs_group.create_dataset(column, data=data)
                    else:
                        token_dfs_group.create_dataset(
                            column, data=results.all_token_dfs[column].values
                        )

            # Save pca_df as a table
            if save_options["pca"]:
                pca_df_group = group.create_group("pca_df")
                for column in pca_df.columns:
                    if pca_df[column].dtype == "object":
                        data = np.array(
                            [
                                str(item).encode("utf-8")
                                for item in pca_df[column].values
                            ]
                        )
                        pca_df_group.create_dataset(column, data=data)
                    else:
                        pca_df_group.create_dataset(column, data=pca_df[column].values)


def main():
    # Configuration
    torch.set_grad_enabled(False)
    device = set_device()
    git_root = get_git_root()

    # Load configuration from TOML
    config = load_streamlit_config("config_pca_streamlit_maxexamples.toml")

    model_name = config["model"]["name"]
    sae_release_short = config["model"]["sae_release_short"]
    sae_ids = config["model"]["sae_ids"]  # Change to list of sae_ids
    remove_special_tokens = config["processing"]["remove_special_tokens"]
    n_batches_reconstruction = config["processing"]["n_batches_reconstruction"]
    activation_threshold = config["processing"]["activation_threshold"]
    subgraph_sizes_to_plot = config["processing"]["subgraph_sizes_to_plot"]
    max_examples = config["processing"]["max_examples"]
    trim_excess = config["processing"]["trim_excess"]
    n_batches_generation = config["generation"]["n_batches_generation"]

    if model_name == "gemma-2-2b" and not remove_special_tokens:
        raise ValueError("Gemma requires removing special tokens")

    # Load save options from config
    save_options = config["processing"]["save_options"]

    # Iterate over each sae_id
    for sae_id in sae_ids:
        # Paths and logging setup
        sae_id_neat = neat_sae_id(sae_id)
        results_dir = create_results_dir(
            model_name, sae_release_short, sae_id_neat, n_batches_generation
        )
        results_path = pj(git_root, results_dir)

        output_dir = pj(git_root, results_dir, f"{sae_id_neat}_pca_for_streamlit")
        os.makedirs(output_dir, exist_ok=True)

        # Load model and SAE
        sae_release = get_sae_release(model_name, sae_release_short)
        model, sae = load_model_and_sae(model_name, sae_release, sae_id, device)

        # Set up activation store
        activation_store = ActivationsStore.from_sae(
            model=model,
            sae=sae,
            streaming=True,
            store_batch_size_prompts=8,
            train_batch_size_tokens=4096,
            n_batches_in_buffer=32,
            device=device,
        )

        # Load node_df
        activation_threshold_safe = str(activation_threshold).replace(".", "_")
        node_df = pd.read_csv(
            pj(results_path, f"dataframes/node_info_df_{activation_threshold_safe}.csv")
        )

        # Process graphs for each subgraph size
        for subgraph_size in subgraph_sizes_to_plot:
            results_dict = {}
            subgraphs_to_process = pd.Series(
                node_df[node_df["subgraph_size"] == subgraph_size]["subgraph_id"]
            ).unique()

            for subgraph_id in tqdm(
                subgraphs_to_process,
                desc=f"Processing subgraphs of size {subgraph_size}",
            ):
                fs_splitting_nodes = node_df[node_df["subgraph_id"] == subgraph_id][
                    "node_id"
                ].tolist()
                results, pca_df = process_graph_for_pca(
                    model,
                    sae,
                    activation_store,
                    fs_splitting_nodes,
                    n_batches_reconstruction,
                    remove_special_tokens,
                    device=device,
                    max_examples=max_examples,
                    trim_excess=trim_excess,
                )

                # Skip this subgraph if PCA still failed after retry
                if pca_df is None:
                    logging.warning(
                        f"Skipping subgraph {subgraph_id} due to PCA failure"
                    )
                    continue

                results_dict[subgraph_id] = (results, pca_df)

            # Save results for this subgraph size
            output_file = pj(
                output_dir,
                f"{max_examples}cap_graph_analysis_results_size_{subgraph_size}_nbatch_{n_batches_reconstruction}.h5",
            )
            save_results_to_hdf5(output_file, results_dict, save_options=save_options)

            logging.info(
                f"Analysis completed for subgraph size {subgraph_size}. Results saved to {output_file}"
            )

        logging.info(f"Processing completed for SAE ID: {sae_id}")


if __name__ == "__main__":
    main()
