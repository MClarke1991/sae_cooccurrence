import logging
import os
from os.path import join as pj

import h5py
import numpy as np
import pandas as pd
import torch
from sae_lens import ActivationsStore
from tqdm.autonotebook import tqdm

from sae_cooccurrence.pca import perform_pca_on_results, process_examples
from sae_cooccurrence.utils.saving_loading import load_model_and_sae, set_device
from sae_cooccurrence.utils.set_paths import get_git_root


def process_graph_for_pca(
    model, sae, activation_store, fs_splitting_nodes, n_batches_reconstruction, device
):
    results = process_examples(
        activation_store,
        model,
        sae,
        fs_splitting_nodes,
        n_batches_reconstruction,
        device,
    )
    pca_df, _ = perform_pca_on_results(results, n_components=3)
    return results, pca_df


def save_results_to_hdf5(file_path, results_dict, save_all_feature_acts=True):
    with h5py.File(file_path, "w") as f:
        for subgraph_id, (results, pca_df) in results_dict.items():
            group = f.create_group(f"subgraph_{subgraph_id}")

            # Save results
            group.create_dataset(
                "all_fired_tokens",
                data=np.array(
                    [token.encode("utf-8") for token in results.all_fired_tokens]
                ),
            )
            group.create_dataset(
                "all_reconstructions", data=results.all_reconstructions.cpu().numpy()
            )
            group.create_dataset(
                "all_graph_feature_acts",
                data=results.all_graph_feature_acts.cpu().numpy(),
            )
            if save_all_feature_acts:
                group.create_dataset(
                    "all_feature_acts", data=results.all_feature_acts.cpu().numpy()
                )
            group.create_dataset(
                "all_max_feature_info", data=results.all_max_feature_info.cpu().numpy()
            )
            group.create_dataset("all_examples_found", data=results.all_examples_found)

            # Save all_token_dfs as a table
            token_dfs_group = group.create_group("all_token_dfs")
            for column in results.all_token_dfs.columns:
                if results.all_token_dfs[column].dtype == "object":
                    # For object dtype (likely strings), encode to UTF-8
                    data = np.array(
                        [
                            str(item).encode("utf-8")
                            for item in results.all_token_dfs[column].values
                        ]
                    )
                    token_dfs_group.create_dataset(column, data=data)
                else:
                    # For numeric dtypes, save directly
                    token_dfs_group.create_dataset(
                        column, data=results.all_token_dfs[column].values
                    )

            # Save pca_df as a table
            pca_df_group = group.create_group("pca_df")
            for column in pca_df.columns:
                if pca_df[column].dtype == "object":
                    # For object dtype (likely strings), encode to UTF-8
                    data = np.array(
                        [str(item).encode("utf-8") for item in pca_df[column].values]
                    )
                    pca_df_group.create_dataset(column, data=data)
                else:
                    # For numeric dtypes, save directly
                    pca_df_group.create_dataset(column, data=pca_df[column].values)


def main():
    # Configuration
    torch.set_grad_enabled(False)
    device = set_device()
    git_root = get_git_root()

    model_name = "gpt2-small"
    sae_release_short = "res-jb-feature-splitting"
    sae_id = "blocks.8.hook_resid_pre_24576"
    n_batches_reconstruction = 100
    activation_threshold = 1.5
    subgraph_sizes_to_plot = [5, 6, 7]  # List of subgraph sizes to process
    save_all_feature_acts = False

    # Paths and logging setup
    sae_id_neat = sae_id.replace(".", "_").replace("/", "_")
    results_dir = f"results/cooc/{model_name}/{sae_release_short}/{sae_id_neat}"
    results_path = pj(git_root, results_dir)

    output_dir = pj(git_root, results_dir, f"{sae_id_neat}_pca_for_streamlit")
    os.makedirs(output_dir, exist_ok=True)

    # Load model and SAE
    sae_release = f"{model_name}-{sae_release_short}"
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
            subgraphs_to_process, desc=f"Processing subgraphs of size {subgraph_size}"
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
                device,
            )
            results_dict[subgraph_id] = (results, pca_df)

        # Save results for this subgraph size
        output_file = pj(output_dir, f"graph_analysis_results_size_{subgraph_size}.h5")
        save_results_to_hdf5(
            output_file, results_dict, save_all_feature_acts=save_all_feature_acts
        )

        logging.info(
            f"Analysis completed for subgraph size {subgraph_size}. Results saved to {output_file}"
        )


if __name__ == "__main__":
    main()
