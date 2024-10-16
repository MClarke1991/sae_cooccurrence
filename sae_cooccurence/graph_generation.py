import ast
import pickle
import warnings
from functools import partial
from math import ceil
from os.path import exists as path_exists
from os.path import join as pj

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from pyvis.network import Network
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import euclidean
from torch import topk as torch_topk
from tqdm.autonotebook import tqdm

from sae_cooccurence.utils.mc_neuronpedia import (
    get_neuronpedia_feature_dashboard_no_open,
    get_neuronpedia_quick_list_no_open,
)


def calculate_token_factors_inds(model, sae):
    # Make labels
    token_factors = model.W_E @ sae.W_dec.T
    print(token_factors.shape)
    # Needs to be on CPU to prevent underflow errors, see Issue #7
    vals, inds = torch_topk(token_factors.T.cpu(), k=10)
    print(inds.shape)
    token_factors_inds = inds.cpu().numpy()
    detect_zero_token_factors_inds(token_factors_inds)
    return token_factors_inds


def calculate_token_factors_inds_efficient(model, sae, batch_size=1000, k=10):
    vocab_size, d_model = model.W_E.shape
    d_sae = sae.W_dec.shape[0]

    # Initialize the result array
    token_factors_inds = np.zeros((d_sae, k), dtype=np.int64)

    # Process in batches
    for i in range(0, d_sae, batch_size):
        end = min(i + batch_size, d_sae)

        # Compute token factors for this batch
        with torch.no_grad():
            batch_token_factors = torch.mm(model.W_E, sae.W_dec[i:end].T)

        # Find top k indices
        _, batch_inds = torch.topk(batch_token_factors, k=k, dim=0)

        # Store results
        token_factors_inds[i:end] = batch_inds.T.cpu().numpy()

        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    detect_zero_token_factors_inds_efficient(token_factors_inds)
    return token_factors_inds


def detect_zero_token_factors_inds_efficient(token_factors_inds):
    if np.all(token_factors_inds == 0):
        print(
            "Warning: All token factor indices are zero. May need to increase precision."
        )


def detect_zero_token_factors_inds(token_factors_inds: np.ndarray) -> None:
    """
    Detects if there are any zero token factor indices in the given array.

    (Note to self): For large SAE, I found that I was getting token factors indices of all zeros.
    Looking into it, it seemed like the actual token_factors were generated without issue, but taking the torch.topk
    resulted in a result that was all zeroes if run on GPU (macOS MPS). This might be todo with MPS being
    restricted to float32. Switching to calculating on CPU resolves the issues.

    Parameters:
    token_factors_inds (np.ndarray): The array of token factor indices.

    Returns:
    None
    """
    min_tfi = token_factors_inds.min()
    max_tfi = token_factors_inds.max()
    n_non_zeroes = token_factors_inds.nonzero()[0].shape[0]

    if (min_tfi == 0 and max_tfi == 0) | n_non_zeroes == 0:
        warnings.warn(
            "All token factor indices are zero. May need to increase precision?"
        )
    return None


def remove_self_loops_inplace(matrices: dict[float, np.ndarray]) -> None:
    """
    Remove self-loops from matrices by setting diagonal elements to zero.
    This function modifies the input matrices in place.

    Args:
    matrices (dict[float, np.ndarray]): dictionary of matrices to modify.

    Returns:
    None: The input matrices are modified in place.
    """
    for threshold, matrix in matrices.items():
        np.fill_diagonal(matrix, 0)


def remove_low_weight_edges(matrix: np.ndarray, edge_threshold: float) -> np.ndarray:
    """
    Remove edges from a matrix that are below a certain threshold.

    Args:
    matrix (np.ndarray): The matrix to threshold.
    threshold (float): The threshold below which to remove edges.

    Returns:
    np.ndarray: The thresholded matrix.
    """
    # Create a copy of the matrix to avoid modifying the original data
    matrix_copy = np.copy(matrix)

    # Set edges below the threshold to 0
    matrix_copy[matrix_copy < edge_threshold] = 0

    return matrix_copy


def largest_component_size(sparse_matrix, threshold):
    binary_matrix = sparse_matrix >= threshold
    n_components, labels = connected_components(
        binary_matrix, directed=False, connection="weak"
    )
    print(f"n_components: {n_components}, labels: {np.max(np.bincount(labels))}")
    return np.max(np.bincount(labels))


def find_threshold(matrix, min_size=150, max_size=200, tolerance=1e-3):
    sparse_matrix = csr_matrix(matrix)

    low, high = 0, 1
    best_threshold = None
    best_size = None

    while high - low > tolerance:
        print(f"low: {low}, high: {high}")
        mid = (low + high) / 2
        size = largest_component_size(sparse_matrix, mid)

        if min_size <= size <= max_size:
            return mid, size  # Early return if size is within desired range

        if size < min_size:
            high = mid
        else:  # size > max_size
            low = mid

        # Update best found so far
        if best_size is None or abs(size - (min_size + max_size) / 2) < abs(
            best_size - (min_size + max_size) / 2
        ):
            best_threshold = mid
            best_size = size

    # If we didn't find an exact match, return the best approximation
    return best_threshold, best_size


def plot_subgraph_size_density(subgraphs, hist_path, filename, min_size, max_size):
    # Extract subgraph sizes from the dictionary
    subgraph_sizes = [len(subgraph) for subgraph in subgraphs]

    # Create a density plot
    plt.figure(figsize=(8, 6))
    plt.hist(subgraph_sizes, bins=30, density=True, color="skyblue")

    # Add dotted vertical lines at min and max sizes
    plt.axvline(min_size, color="r", linestyle="--", label=f"Min Size: {min_size}")
    plt.axvline(max_size, color="g", linestyle="--", label=f"Max Size: {max_size}")

    plt.xlabel("Subgraph Size")
    plt.ylabel("Density")
    plt.title(
        f"Density Plot of Subgraph Sizes\nMin Size: {min_size}, Max Size: {max_size}"
    )
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.yscale("log")
    plt.legend()

    # Save the plot as an image file
    plt.savefig(pj(hist_path, f"{filename}.png"), dpi=300)
    plt.close()


def create_graph_from_matrix(matrix):
    # Create a graph from the matrix
    graph = nx.from_numpy_array(matrix)
    return graph


def get_subgraphs(graph):
    # Find connected components (subgraphs)
    subgraphs = [graph.subgraph(c) for c in nx.connected_components(graph)]
    return subgraphs


# def create_node_info_dataframe(subgraphs,
#                                activity_threshold: float,
#                                feature_activations,
#                                token_factors_inds,
#                                tokenizer,
#                                sae_id,
#                                model_name,
#                                sae_release_short):
#     # Create a DataFrame with node information

#     sae_layer = re.search(r'blocks\.(\d+)', sae_id).group(1) # type: ignore
#     node_info_data = []
#     for i, subgraph in tqdm(enumerate(subgraphs)):
#         for node in subgraph.nodes():
#             # Get feature activations for the node
#             node_activations = feature_activations[node]

#             # Get top 10 token indices for the node
#             top_10_token_indices = token_factors_inds[node][:10]

#             # Decode token indices to actual tokens
#             top_10_tokens = [tokenizer.decode([idx]) for idx in top_10_token_indices]

#             if sae_release_short == "res-jb-feature-splitting":
#                 sae_release_short_quicklist = "res-jb-fs"
#             else:
#                 sae_release_short_quicklist = sae_release_short

#             node_info_data.append({
#                 'node_id': node,
#                 'activity_threshold': activity_threshold,
#                 'subgraph_id': i,
#                 'subgraph_size': len(subgraph),
#                 'feature_activations': node_activations,
#                 'top_10_tokens': top_10_tokens,
#                 'neuronpedia_link': mc_neuronpedia_link(node, int(sae_layer), model_name, sae_release_short_quicklist)
#             })
#     node_info_df = pd.DataFrame(node_info_data)
#     return node_info_df


def create_node_info_dataframe(
    subgraphs,
    activity_threshold: float,
    feature_activations,
    token_factors_inds,
    decode_tokens,
    SAE,
    include_metrics: bool = False,
):
    # sae_layer = get_layer_from_id(sae_id)

    # Vectorize token decoding
    # decode_tokens = np.vectorize(lambda idx: tokenizer.decode([idx]))

    # Partial function for neuronpedia link
    neuronpedia_link_partial = partial(
        # mc_neuronpedia_link, sae_id=sae_id, model=model_name, dataset=sae_release_short
        get_neuronpedia_feature_dashboard_no_open,
        sae=SAE,
    )

    node_info_data = []
    subgraph_nodes = {}

    for i, subgraph in enumerate(tqdm(subgraphs)):
        nodes = list(subgraph.nodes())
        subgraph_nodes[i] = nodes
        subgraph_size = len(subgraph)

        # Vectorized operations
        node_activations = feature_activations[nodes]
        top_10_token_indices = token_factors_inds[nodes, :10]
        top_10_tokens = [
            decode_tokens(indices).tolist() for indices in top_10_token_indices
        ]
        neuronpedia_links = [neuronpedia_link_partial(index=node) for node in nodes]

        if include_metrics:
            metrics = calculate_key_graph_metrics(subgraph)
            struc_score = calculate_structure_scores(metrics)
        else:
            metrics, struc_score = None, None

        for node, act, tokens, link in zip(
            nodes, node_activations, top_10_tokens, neuronpedia_links
        ):
            node_info = {
                "node_id": node,
                "activity_threshold": activity_threshold,
                "subgraph_id": i,
                "subgraph_size": subgraph_size,
                "feature_activations": act,
                "top_10_tokens": tokens,
                "neuronpedia_link": link,
            }

            if include_metrics and metrics is not None and struc_score is not None:
                node_info.update(
                    {
                        "density": metrics["density"],
                        "max_avg_degree_ratio": metrics["max_avg_degree_ratio"],
                        "avg_clustering": metrics["avg_clustering"],
                        "diameter": metrics["diameter"],
                        "single_node_score": struc_score["single_node_score"],
                        "hub_spoke_score": struc_score["hub_spoke_score"],
                        "strongly_connected_score": struc_score[
                            "strongly_connected_score"
                        ],
                        "linear_score": struc_score["linear_score"],
                    }
                )

            node_info_data.append(node_info)

    # Create quicklist links
    quicklist_links = {}
    for subgraph_id, nodes in subgraph_nodes.items():
        num_batches = ceil(len(nodes) / 10)
        for batch_num in range(num_batches):
            start = batch_num * 10
            end = min((batch_num + 1) * 10, len(nodes))
            batch_nodes = nodes[start:end]
            # quicklist_link = mc_quicklist(
            #     features=batch_nodes,
            #     sae_id=sae_id,
            #     model=model_name,
            #     dataset=sae_release_short,
            #     name=f"Subgraph_{subgraph_id}_Batch_{batch_num}",
            # )
            quicklist_link = get_neuronpedia_quick_list_no_open(
                sae=SAE, features=batch_nodes, open=False
            )
            for node in batch_nodes:
                quicklist_links[(subgraph_id, node)] = quicklist_link

    # Add quicklist links to node_info_data
    for node_info in node_info_data:
        node_info["quicklist_link"] = quicklist_links.get(
            (node_info["subgraph_id"], node_info["node_id"]), ""
        )

    return pd.DataFrame(node_info_data)


def create_subgraph_plot(subgraph, node_info_df, edge_threshold):
    # Create a new Plotly graph
    plot = Network(
        notebook=True, cdn_resources="in_line", height="1000px", width="1000px"
    )

    # Normalize feature activations within the subgraph
    subgraph_node_indices = node_info_df["node_id"].isin(subgraph.nodes())
    subgraph_node_info_df = node_info_df.loc[subgraph_node_indices]
    subgraph_id = node_info_df["subgraph_id"].loc[subgraph_node_indices].iloc[0]
    min_activation = subgraph_node_info_df["feature_activations"].min()
    max_activation = subgraph_node_info_df["feature_activations"].max()
    node_info_df.loc[subgraph_node_indices, "normalized_activations"] = (
        subgraph_node_info_df["feature_activations"] - min_activation
    ) / (max_activation - min_activation)

    # Add nodes to the plot
    for node in subgraph.nodes():
        node_info = node_info_df[node_info_df["node_id"] == node].iloc[0]
        node_id = str(node)
        node_label = f"{node_info['top_10_tokens'][0]} (ID: {node_id}, Act: {node_info['feature_activations']:.2f})"
        node_color = plt.cm.viridis(node_info["normalized_activations"])  # type: ignore
        node_title = "<br>".join(node_info["top_10_tokens"])
        plot.add_node(
            node_id,
            label=node_label,
            color=matplotlib_to_hex(node_color),
            title=node_title,
        )

    # Normalize edge weights within the subgraph
    edge_weights = [data["weight"] for _, _, data in subgraph.edges(data=True)]
    min_weight = min(edge_weights)
    max_weight = max(edge_weights)
    normalized_edge_weights = [
        (weight - min_weight) / (max_weight - min_weight + 1e6)
        for weight in edge_weights
    ]

    # Add edges to the plot
    for (u, v, data), normalized_weight in zip(
        subgraph.edges(data=True), normalized_edge_weights
    ):
        edge_weight = data["weight"]
        if edge_weight >= edge_threshold:
            edge_color = plt.cm.plasma(normalized_weight)  # type: ignore
            edge_title = str(edge_weight)
            plot.add_edge(
                str(u),
                str(v),
                value=edge_weight,
                color=matplotlib_to_hex(edge_color),
                title=edge_title,
            )

    # Set the plot's title to include the subgraph ID
    # plot.title = f"Subgraph {subgraph_id}"

    return plot, subgraph_id


def matplotlib_to_hex(color):
    return (
        f"#{int(color[0] * 255):02x}{int(color[1] * 255):02x}{int(color[2] * 255):02x}"
    )


# Graph structure


def calculate_key_graph_metrics(G):
    # Number of nodes
    n_nodes = G.number_of_nodes()

    if n_nodes == 1:
        return {
            "density": 0,
            "max_avg_degree_ratio": 0,
            "avg_clustering": 0,
            "diameter": 0,
            "is_single_node": True,
        }

    # Density
    density = nx.density(G)

    # Degree statistics
    degrees = [d for n, d in G.degree()]
    avg_degree = sum(degrees) / n_nodes
    max_degree = max(degrees)
    max_avg_degree_ratio = max_degree / avg_degree if avg_degree > 0 else n_nodes - 1

    # Average clustering coefficient
    avg_clustering = nx.average_clustering(G)

    # Diameter (use n_nodes-1 for disconnected graphs as an upper bound)
    diameter = nx.diameter(G) if nx.is_connected(G) else n_nodes - 1

    return {
        "n_nodes": n_nodes,
        "density": density,
        "max_avg_degree_ratio": max_avg_degree_ratio,
        "avg_clustering": avg_clustering,
        "diameter": diameter,
        "is_single_node": False,
    }


def calculate_structure_scores(metrics):
    if metrics.get("is_single_node", False):
        return {
            "single_node_score": 1.0,
            "hub_spoke_score": 0.0,
            "strongly_connected_score": 0.0,
            "linear_score": 0.0,
        }

    n_nodes = metrics["n_nodes"]

    # Idealized metrics for each structure, adjusted for graph size
    ideal_hub_spoke = np.array([2 / n_nodes, n_nodes - 1, 0, 2])
    ideal_strongly_connected = np.array([1, 1, 1, 1])
    ideal_linear = np.array([2 / (n_nodes - 1), 2, 0, n_nodes - 1])

    # Normalize metrics
    normalized_metrics = np.array(
        [
            metrics["density"],
            min(metrics["max_avg_degree_ratio"], n_nodes - 1),
            metrics["avg_clustering"],
            metrics["diameter"],
        ]
    )

    # Calculate distances to ideal structures
    hub_spoke_distance = euclidean(normalized_metrics, ideal_hub_spoke)
    strongly_connected_distance = euclidean(
        normalized_metrics, ideal_strongly_connected
    )
    linear_distance = euclidean(normalized_metrics, ideal_linear)

    # Convert distances to scores (closer = higher score)
    total_distance = hub_spoke_distance + strongly_connected_distance + linear_distance
    if total_distance == 0:
        # If all distances are 0, assign equal scores
        hub_spoke_score = strongly_connected_score = linear_score = 1 / 3
    else:
        hub_spoke_score = 1 - (hub_spoke_distance / total_distance)
        strongly_connected_score = 1 - (strongly_connected_distance / total_distance)
        linear_score = 1 - (linear_distance / total_distance)

    return {
        "single_node_score": 0.0,
        "hub_spoke_score": hub_spoke_score,
        "strongly_connected_score": strongly_connected_score,
        "linear_score": linear_score,
    }


# Static plotting


def plot_subgraph_static(
    subgraph,
    node_info_df,
    output_path,
    activation_array,
    save_figs=False,
    normalize_globally=True,
) -> None:
    # Create a new figure
    plt.figure(figsize=(7, 7))

    # Create a layout for our nodes
    pos = nx.spring_layout(subgraph, k=0.5, iterations=50, seed=1234)

    # Extract activations for nodes in this subgraph
    subgraph_activations = [activation_array[node] for node in subgraph.nodes()]

    # Determine normalization range
    if normalize_globally:
        min_activation = min(activation_array)
        max_activation = max(activation_array)
    else:
        min_activation = min(subgraph_activations)
        max_activation = max(subgraph_activations)

    activation_range = max_activation - min_activation

    # Prepare node labels and colors
    labels = {}
    node_colors = []
    for node in subgraph.nodes():
        node_info = node_info_df[node_info_df["node_id"] == node].iloc[0]
        node_id = node_info["node_id"]
        # Safely evaluate the string representation of the list
        top_tokens = ast.literal_eval(node_info["top_10_tokens"])
        top_token = top_tokens[0]
        labels[node] = f"ID: {node_id}\n{top_token}"

        # Set fill to white if activation is 0, otherwise use the color map
        if activation_array[node] == 0:
            node_colors.append("white")
        else:
            # Normalize the node's activation
            if activation_range != 0:
                normalized_activation = (
                    activation_array[node] - min_activation
                ) / activation_range
            else:
                normalized_activation = (
                    0.5  # If all activations are the same, use middle value
                )

            node_colors.append(plt.cm.Blues(normalized_activation))  # type: ignore

    # Get edge weights
    edge_weights = [subgraph[u][v]["weight"] for u, v in subgraph.edges()]

    # Normalize edge weights for thickness
    max_weight = max(edge_weights)
    min_weight = min(edge_weights)
    normalized_weights = [
        (w - min_weight) / (max_weight - min_weight) for w in edge_weights
    ]

    # Scale the weights to a reasonable thickness range (e.g., 0.5 to 5)
    edge_thickness = [0.5 + 4.5 * w for w in normalized_weights]

    # Draw the graph with weighted edges and black outline for nodes
    nx.draw(
        subgraph,
        pos,
        with_labels=False,
        node_size=1000,
        node_color=node_colors,
        # edgecolors=(0.467, 0.282, 0.702, 1.0), # node edge color
        edgecolors="black",  # node edge color
        linewidths=3,  # node edge thickness
        # edge_color=(0.5, 0.5, 0.5, 0.5),
        edge_color=(0.5, 0.5, 0.5, 0.75),
        # edge_color="black", # edge color
        # edge_color= (0.467, 0.282, 0.702, 0.502), # edge color
        # edge_color = (0.282, 0.129, 0.6, 0.659),
        width=edge_thickness,  # edge thickness
        arrows=True,
    )
    # note that the edge_colour is 3 rgb values and then the alpha

    # Add node labels outside the nodes
    label_pos = {
        k: (v[0], v[1] - 0.12) for k, v in pos.items()
    }  # Offset labels above nodes
    nx.draw_networkx_labels(subgraph, label_pos, labels, font_size=8)

    # Set title
    # plt.title(f"Subgraph {node_info_df['subgraph_id'].iloc[0]}", fontsize=16)

    # Remove axis
    plt.axis("off")

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    if save_figs:
        plt.savefig(f"{output_path}.png", format="png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{output_path}.pdf", format="pdf", dpi=300, bbox_inches="tight")
        plt.savefig(f"{output_path}.svg", format="svg", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# Usage example:
# plot_subgraph_png(subgraph, node_info_df, edge_threshold, 'subgraph_plot.png')


def load_subgraph(
    results_path: str, activation_threshold: float, subgraph_id: int
) -> nx.Graph:
    """
    Load a subgraph from a pickle file.

    Args:
    results_path (str): Path to the results directory
    activation_threshold (float): Activation threshold used for the subgraph
    subgraph_id (int or str): ID of the subgraph to load

    Returns:
    networkx.Graph: The loaded subgraph
    """
    # Convert activation threshold to a string safe for filenames
    activation_threshold_safe = str(activation_threshold).replace(".", "_")

    # Construct the full path to the subgraph pickle file
    subgraph_path = pj(
        results_path,
        "subgraph_objects",
        f"activation_{activation_threshold_safe}",
        f"subgraph_{subgraph_id}.pkl",
    )
    # Check if the file exists
    if not path_exists(subgraph_path):
        raise FileNotFoundError(f"Subgraph file not found: {subgraph_path}")
    else:
        # Load and return the subgraph
        with open(subgraph_path, "rb") as f:
            subgraph = pickle.load(f)

    return subgraph
