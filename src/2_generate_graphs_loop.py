import cProfile
import io
import logging
import os
import pickle
import pstats
import time
from os.path import join as pj
from pstats import SortKey

import networkx as nx
import numpy as np
import toml
import torch
from sae_lens import SAE
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

from sae_cooccurrence.graph_generation import (
    calculate_token_factors_inds_efficient,
    create_graph_from_matrix,
    create_node_info_dataframe,
    create_subgraph_plot,
    find_threshold,
    get_subgraphs,
    plot_subgraph_size_density,
    remove_low_weight_edges,
    remove_self_loops_inplace,
)
from sae_cooccurrence.normalised_cooc_functions import setup_logging
from sae_cooccurrence.utils.saving_loading import (
    load_model_and_sae,
    load_npz_files,
    log_config_variables,
    notify,
    set_device,
)
from sae_cooccurrence.utils.set_paths import get_git_root


def process_sae_for_graph(sae_id: str, config: dict, device: str) -> None:
    """
    Processes the normalised feature co-occurrence matrices for a given SAE ID.

    This first: finds a threshold of edge weight such that we decompose the graph into connected components,
    that are below a threshold in size.
    Then: generates graphs from the normalised feature co-occurrence matrices and outputs dataframes of their nodes' information, as well as visualisations of their subgraphs, and pickle files of the subgraph objects for later revisualisation.

    Parameters:
    sae_id (str): The unique identifier for the SAE.
    config (dict): The configuration dictionary containing model and analysis settings.
    device (str): The device to use for computations (e.g., 'cpu', 'mps', 'cuda').
    """
    sae_id_neat = sae_id.replace(".", "_").replace("/", "_")
    sae_release = (
        "gemma-scope-2b-pt-res-canonical"
        if config["generation"]["model_name"] == "gemma-2-2b"
        else f"{config['generation']['model_name']}-{config['generation']['sae_release_short']}"
    )
    results_dir = f"results/cooc/{config['generation']['model_name']}/{config['generation']['sae_release_short']}/{sae_id_neat}"
    results_path = pj(get_git_root(), results_dir)

    setup_logging(
        results_path,
        config["generation"]["model_name"],
        config["generation"]["sae_release_short"],
        sae_id_neat,
        context="graph_gen",
    )
    log_start_info(config, sae_id)
    create_directories(results_path)

    model, sae = load_model_and_sae(
        config["generation"]["model_name"], sae_release, sae_id, device
    )
    token_factors_inds = calculate_token_factors_inds_efficient(model, sae)
    decode_tokens = create_decode_tokens_function(model)

    feature_activations, matrices = load_data(results_path)

    pr = cProfile.Profile()
    pr.enable()

    process_matrices(
        matrices,
        config,
        results_path,
        feature_activations,
        token_factors_inds,
        decode_tokens,
        sae,
    )

    log_execution_time()
    save_profiling_results(pr, results_path, sae_id_neat)


def log_start_info(config: dict, sae_id: str) -> None:
    logging.info("Script started running")
    logging.info(
        f"Variables - model_name: {config['generation']['model_name']}, sae_release_short: {config['generation']['sae_release_short']}, sae_id: {sae_id}"
    )
    logging.info(f"Random seed: {config['analysis']['random_seed']}")
    log_config_variables(config)


def create_directories(results_path: str) -> None:
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(pj(results_path, "histograms"), exist_ok=True)
    os.makedirs(pj(results_path, "dataframes"), exist_ok=True)


def create_decode_tokens_function(model: HookedTransformer) -> np.vectorize:
    def decode_tokens(idx: int) -> str:
        return "None" if idx is None else model.tokenizer.decode([idx])  # type: ignore

    return np.vectorize(decode_tokens)


def load_data(results_path: str) -> tuple:
    return (
        load_npz_files(results_path, "feature_acts_cooc_activations"),
        load_npz_files(results_path, "feature_acts_cooc_jaccard"),
    )


def process_matrices(
    matrices: dict,
    config: dict,
    results_path: str,
    feature_activations: np.ndarray,
    token_factors_inds: np.ndarray,
    decode_tokens: np.vectorize,
    sae: SAE,
) -> None:
    """
    Processes the given matrices to generate thresholded graphs and subgraphs, plots their densities, and further processes the subgraphs.

    Parameters:
    - matrices (dict): A dictionary of matrices to be processed.
    - config (dict): Configuration settings for the processing.
    - results_path (str): The path where the results will be saved.
    - feature_activations (np.ndarray): An array of feature activations.
    - token_factors_inds (np.ndarray): An array of token factor indices.
    - decode_tokens (np.vectorize): A vectorized function to decode token indices.
    - sae (SAE): The SAE model instance.
    """
    # Remove self-loops from the matrices in-place
    remove_self_loops_inplace(matrices)

    # Calculate edge thresholds for each matrix based on configuration
    edge_thresholds = calculate_edge_thresholds(matrices, config)

    # Create thresholded matrices by removing edges below the calculated thresholds
    thresholded_matrices = create_thresholded_matrices(matrices, edge_thresholds)

    # Convert thresholded matrices to graphs
    thresholded_graphs = create_thresholded_graphs(thresholded_matrices)

    # Find connected subgraphs within the thresholded graphs
    thresholded_subgraphs = create_thresholded_subgraphs(thresholded_graphs)

    # Plot the density of subgraphs for each threshold
    plot_subgraph_densities(thresholded_subgraphs, results_path, config)

    # Further process the subgraphs, including node information and metrics
    # TODO seperate out from this function to make more modular
    process_subgraphs(
        thresholded_matrices,
        feature_activations,
        token_factors_inds,
        decode_tokens,
        sae,
        config,
        results_path,
        edge_thresholds,
    )


def calculate_edge_thresholds(matrices, config):
    return {
        key: find_threshold(
            matrix,
            min_size=config["analysis"]["min_subgraph_size"],
            max_size=config["analysis"]["max_subgraph_size"],
        )
        for key, matrix in matrices.items()
    }


def create_thresholded_matrices(matrices, edge_thresholds):
    return {
        key: remove_low_weight_edges(matrix, edge_thresholds.get(key)[0])
        for key, matrix in matrices.items()
    }


def create_thresholded_graphs(thresholded_matrices):
    return {
        threshold: nx.from_numpy_array(matrix)
        for threshold, matrix in thresholded_matrices.items()
    }


def create_thresholded_subgraphs(thresholded_graphs):
    return {
        threshold: list(nx.connected_components(graph))
        for threshold, graph in thresholded_graphs.items()
    }


def plot_subgraph_densities(thresholded_subgraphs, results_path, config):
    for threshold, subgraphs in thresholded_subgraphs.items():
        safe_threshold = str(threshold).replace(".", "_")
        plot_subgraph_size_density(
            subgraphs,
            pj(results_path, "histograms"),
            f"subgraph_density_{safe_threshold}",
            config["analysis"]["min_subgraph_size"],
            config["analysis"]["max_subgraph_size"],
        )


def process_subgraphs(
    thresholded_matrices,
    feature_activations,
    token_factors_inds,
    decode_tokens,
    sae,
    config,
    results_path,
    edge_thresholds,
):
    graphs, subgraph_lists, node_info_dfs = {}, {}, {}

    for key, matrix in tqdm(thresholded_matrices.items(), leave=False):
        graph = create_graph_from_matrix(matrix)
        graphs[key] = graph
        subgraphs = get_subgraphs(graph)
        subgraph_lists[key] = subgraphs
        node_info_df = create_node_info_dataframe(
            subgraphs=subgraphs,
            activity_threshold=key,
            feature_activations=feature_activations.get(key),
            token_factors_inds=token_factors_inds,
            decode_tokens=decode_tokens,
            SAE=sae,
            include_metrics=config["analysis"].get("include_metrics", False),
        )
        node_info_dfs[key] = node_info_df

    save_node_info_dfs(node_info_dfs, results_path)

    if not config["analysis"].get("skip_subgraph_plots", False):
        create_subgraph_plots(
            subgraph_lists, node_info_dfs, edge_thresholds, config, results_path
        )
    else:
        logging.info("Skipping subgraph plot generation as per configuration.")

    if not config["analysis"].get("skip_subgraph_pickles", False):
        save_subgraph_pickles(subgraph_lists, results_path)
    else:
        logging.info("Skipping subgraph pickle saving as per configuration.")


def save_node_info_dfs(node_info_dfs, results_path):
    for key, df in node_info_dfs.items():
        safe_key = str(key).replace(".", "_")
        df.to_csv(
            pj(results_path, "dataframes", f"node_info_df_{safe_key}.csv"), index=False
        )


def create_subgraph_plots(
    subgraph_lists, node_info_dfs, edge_thresholds, config, results_path
):
    for key, subgraphs in subgraph_lists.items():
        safe_key = str(key).replace(".", "_")
        subgraph_dir = os.path.join(results_path, f"graphs_{safe_key}")
        os.makedirs(subgraph_dir, exist_ok=True)

        node_info_df = node_info_dfs[key]
        edge_threshold = edge_thresholds[key][0]

        large_subgraphs = [
            subgraph
            for subgraph in subgraphs
            if len(subgraph) >= config["analysis"]["min_subgraph_size_to_plot"]
        ]

        for subgraph in large_subgraphs:
            plot, subgraph_id = create_subgraph_plot(
                subgraph, node_info_df, edge_threshold
            )
            plot_filename = os.path.join(subgraph_dir, f"subgraph_{subgraph_id}.html")
            out = plot.generate_html()
            with open(plot_filename, "w", encoding="utf-8") as f:
                f.write(out)


def save_subgraph_pickles(subgraph_lists, results_path):
    for key, subgraphs in subgraph_lists.items():
        safe_key = str(key).replace(".", "_")
        pickle_dir = os.path.join(
            results_path, f"subgraph_objects/activation_{safe_key}"
        )
        os.makedirs(pickle_dir, exist_ok=True)

        for i, subgraph in enumerate(subgraphs):
            pickle_filename = os.path.join(pickle_dir, f"subgraph_{i}.pkl")
            with open(pickle_filename, "wb") as f:
                pickle.dump(subgraph, f)


def log_execution_time():
    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(
        f"Script finished running. Execution time: {execution_time:.2f} seconds"
    )


def save_profiling_results(pr, results_path, sae_id_neat):
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats()

    with open(pj(results_path, f"profiling_results_{sae_id_neat}.txt"), "w") as f:
        f.write(s.getvalue())


def main():
    torch.set_grad_enabled(False)
    device = set_device()
    git_root = get_git_root()
    global start_time
    start_time = time.time()

    config = toml.load(pj(git_root, "src", "config_gemma.toml"))

    for sae_id in tqdm(config["generation"]["sae_ids"], desc="Processing SAE IDs"):
        process_sae_for_graph(sae_id, config, device)

    notify("Graph generation and analysis complete.")


if __name__ == "__main__":
    main()
