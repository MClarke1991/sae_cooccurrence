import logging
import os
import pickle
from os.path import join as pj

import numpy as np
import pandas as pd
import toml
import torch
from sae_lens import SAE, ActivationsStore
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

from PIBBSS.pca import (
    calculate_pca_decoder,
    create_pca_plots_decoder,
    perform_pca_on_results,
    plot_pca_explanation_and_save,
    plot_pca_feature_strength,
    plot_pca_with_active_features,
    plot_pca_with_top_feature,
    plot_simple_scatter,
    plot_token_pca_and_save,
    process_examples,
)
from PIBBSS.utils.saving_loading import set_device
from PIBBSS.utils.set_paths import get_git_root


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_sae_id(
    sae_id,
    activation_threshold,
    candidate_size,
    candidates_per_size,
    pca_prefix="pca",
    recalculate_results=False,
):
    sae_id_neat = sae_id.replace(".", "_")
    results_dir = f"results/cooc/{model_name}/{sae_release_short}/{sae_id_neat}"
    results_path = pj(git_root, results_dir)
    activation_threshold_safe = str(activation_threshold).replace(".", "_")

    pca_root_dir = pj("pca", f"{pca_prefix}_{activation_threshold_safe}")
    pca_sub_dir = f"subgraph_size_{candidate_size}"
    pca_root_path = pj(results_path, pca_root_dir)
    create_directory(pca_root_path)
    pca_path = pj(pca_root_path, pca_sub_dir)
    create_directory(pca_path)

    node_df = pd.read_csv(
        pj(results_path, f"dataframes/node_info_df_{activation_threshold_safe}.csv")
    )

    candidate_list = (
        node_df.query("subgraph_size == @candidate_size")["subgraph_id"]
        .unique()
        .tolist()[:candidates_per_size]
    )

    # Load SAE
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=f"{model_name}-{sae_release_short}", sae_id=sae_id, device=device
    )

    # Normalise the decoder weights
    sae.W_dec.norm(dim=-1).mean()
    sae.fold_W_dec_norm()

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

    for fs_splitting_cluster in tqdm(candidate_list, leave=False):
        cluster_path = pj(pca_path, f"cluster_{fs_splitting_cluster}")
        create_directory(cluster_path)

        results_file = pj(cluster_path, f"results_cluster_{fs_splitting_cluster}.pkl")
        fs_splitting_nodes = node_df.query("subgraph_id == @fs_splitting_cluster")[
            "node_id"
        ].tolist()

        if os.path.exists(results_file) and not recalculate_results:
            # Load existing results
            with open(results_file, "rb") as f:
                results = pickle.load(f)
            print(f"Loaded existing results for cluster {fs_splitting_cluster}")
        else:
            # Calculate new results
            results = process_examples(
                activation_store,
                model,
                sae,
                fs_splitting_nodes,
                n_batches_reconstruction,
            )

            # Save results
            with open(results_file, "wb") as f:
                pickle.dump(results, f)
            # print(f"Calculated and saved new results for cluster {fs_splitting_cluster}")

        pca_df, pca = perform_pca_on_results(results, n_components=3)

        # Save pca_df as CSV
        pca_df_filename = f"pca_df_cluster_{fs_splitting_cluster}.csv"
        pca_df.to_csv(pj(cluster_path, pca_df_filename), index=False)

        # Create subdirectories for each plot type and generate plots
        token_pca_path = pj(cluster_path, "token_pca")
        create_directory(token_pca_path)
        plot_token_pca_and_save(
            pca_df, token_pca_path, fs_splitting_cluster, color_by="token", save=True
        )

        pca_explanation_path = pj(cluster_path, "pca_explanation")
        create_directory(pca_explanation_path)
        plot_pca_explanation_and_save(
            pca, pca_explanation_path, fs_splitting_cluster, save=True
        )

        scatter_path = pj(cluster_path, "scatter")
        create_directory(scatter_path)
        plot_simple_scatter(
            results, scatter_path, fs_splitting_cluster, fs_splitting_nodes, save=True
        )

        top_feature_path = pj(cluster_path, "top_feature")
        create_directory(top_feature_path)
        plot_pca_with_top_feature(
            pca_df,
            results,
            fs_splitting_nodes,
            fs_splitting_cluster,
            top_feature_path,
            save=True,
        )

        feature_strength_path = pj(cluster_path, "feature_strength")
        create_directory(feature_strength_path)
        plot_pca_feature_strength(
            pca_df,
            results,
            fs_splitting_nodes,
            fs_splitting_cluster,
            feature_strength_path,
            pc_x="PC1",
            pc_y="PC2",
            save=True,
        )
        plot_pca_feature_strength(
            pca_df,
            results,
            fs_splitting_nodes,
            fs_splitting_cluster,
            feature_strength_path,
            pc_x="PC1",
            pc_y="PC3",
            save=True,
        )
        plot_pca_feature_strength(
            pca_df,
            results,
            fs_splitting_nodes,
            fs_splitting_cluster,
            feature_strength_path,
            pc_x="PC2",
            pc_y="PC3",
            save=True,
        )

        active_features_path = pj(cluster_path, "active_features")
        create_directory(active_features_path)
        plot_pca_with_active_features(
            pca_df,
            results,
            fs_splitting_nodes,
            fs_splitting_cluster,
            active_features_path,
            activation_threshold=activation_threshold,
            save=True,
        )

        pca_decoder, pca_decoder_df = calculate_pca_decoder(sae, fs_splitting_nodes)

        # Save pca_decoder_df as CSV
        pca_decoder_df_filename = f"pca_decoder_df_cluster_{fs_splitting_cluster}.csv"
        pca_decoder_df.to_csv(pj(cluster_path, pca_decoder_df_filename), index=False)

        pca_decoder_path = pj(cluster_path, "pca_decoder")
        create_directory(pca_decoder_path)
        create_pca_plots_decoder(
            pca_decoder_df, fs_splitting_cluster, pca_decoder_path, save=True
        )


# Config -------------
torch.set_grad_enabled(False)
device = set_device()

git_root = get_git_root()
config = toml.load(pj(git_root, "src", "cooc", "config_feature_split.toml"))
model_name = config["generation"]["model_name"]
sae_release_short = config["generation"]["sae_release_short"]
sae_ids = config["generation"]["sae_ids"]
activation_thresholds = config["generation"]["activation_thresholds"]
random_seed = config["analysis"]["random_seed"]
min_subgraph_size = config["analysis"]["min_subgraph_size"]
max_subgraph_size = config["analysis"]["max_subgraph_size"]
candidate_sizes = config["pca"]["candidate_sizes"]
candidates_per_size = config["pca"]["candidates_per_size"]
n_batches_reconstruction = config["pca"]["n_batches_reconstruction"]
recalculate_results = config["pca"]["recalculate_results"]
hist_dir = "histograms"
df_dir = "dataframes"
logging.info(f"Random seed:{random_seed}")
np.random.seed(random_seed)

# Model ------
# Load model
model = HookedTransformer.from_pretrained(model_name, device=device)

# Main execution
profiler.start()
for sae_id in tqdm(sae_ids, desc="SAE"):
    for activation_threshold in tqdm(
        activation_thresholds, desc="Activation Thresholds", leave=False
    ):
        for candidate_size in tqdm(
            candidate_sizes, desc="Candidate Sizes", leave=False
        ):
            process_sae_id(
                sae_id,
                activation_threshold,
                candidate_size,
                candidates_per_size,
                recalculate_results=recalculate_results,
            )
# code you want to profile
profiler.stop()
profiler.print()
