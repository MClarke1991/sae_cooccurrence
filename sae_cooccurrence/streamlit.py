import glob
import logging
import os
import re
from os.path import join as pj

import h5py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
import streamlit as st
import toml
from scipy import sparse

from sae_cooccurrence.utils.set_paths import get_git_root

#### Data loading ####


def load_streamlit_config(filename: str) -> dict:
    """Load the streamlit config toml file

    Args:
        filename: Name of the toml file to load default: "src/config_pca_streamlit_maxexamples.toml"

    Returns:
        Dictionary containing the config
    """
    config_path = pj(get_git_root(), "src", filename)
    with open(config_path) as f:
        return toml.load(f)


def load_dataset(dataset: h5py.Dataset) -> np.ndarray | bytes:
    """Load a dataset from an h5py file

    Args:
        dataset: The dataset to load

    Returns:
        The loaded dataset
    """
    if dataset.shape == ():  # Scalar dataset
        return dataset[()]
    else:  # Array dataset
        return dataset[:]


def decode_if_bytes(data):
    """Decode bytes if necessary

    E.g. if the token strings are stored as bytes in the h5py file, we decode them to strings.
    """
    if isinstance(data, bytes):
        return data.decode("utf-8")
    elif isinstance(data, np.ndarray) and data.dtype.char == "S":
        return np.char.decode(data, "utf-8")
    return data


def load_subgraph_data(
    file_path: str, subgraph_id: int, load_options: dict
) -> tuple[dict, pd.DataFrame]:
    """Load the data for a given subgraph

    Args:
        file_path: Path to the h5 file containing the subgraph
        subgraph_id: The ID of the subgraph to load
        load_options: A dictionary of options for loading the subgraph data

    Returns:
        A tuple containing the subgraph data and the PCA dataframe
    """
    log_memory_usage("start of load_subgraph_data")

    base_path = os.path.splitext(file_path)[0]
    chunk_pattern = f"{base_path}_chunk*.h5"
    chunk_files = sorted(glob.glob(chunk_pattern))

    # Find which file contains our subgraph
    target_file = file_path
    if chunk_files:
        for chunk_file in chunk_files:
            with h5py.File(chunk_file, "r") as f:
                if f"subgraph_{subgraph_id}" in f:
                    target_file = chunk_file
                    break

    # Load data from the correct file
    with h5py.File(target_file, "r") as f:
        group = f[f"subgraph_{subgraph_id}"]
        results = {}

        # Conditionally load each component based on config
        if load_options["fired_tokens"]:
            results["all_fired_tokens"] = decode_if_bytes(
                load_dataset(group["all_fired_tokens"])  # type: ignore
            )

        if load_options["reconstructions"]:
            results["all_reconstructions"] = load_dataset(group["all_reconstructions"])  # type: ignore

        if load_options["graph_feature_acts"]:
            results["all_graph_feature_acts"] = load_dataset(
                group["all_graph_feature_acts"]  # type: ignore
            )  # type: ignore

        if load_options["feature_acts"]:
            results["all_feature_acts"] = load_dataset(group["all_feature_acts"])  # type: ignore

        if load_options["max_feature_info"]:
            results["all_max_feature_info"] = load_dataset(
                group["all_max_feature_info"]  # type: ignore
            )  # type: ignore

        if load_options["examples_found"]:
            results["all_examples_found"] = load_dataset(group["all_examples_found"])  # type: ignore

        if load_options["token_dfs"]:
            results["token_dfs"] = load_dataset(group["token_dfs"])  # type: ignore

        # Load pca_df
        pca_df_group = group["pca_df"]  # type: ignore
        pca_df_data = {}
        for column in pca_df_group.keys():  # type: ignore
            pca_df_data[column] = decode_if_bytes(
                load_dataset(pca_df_group[column])  # type: ignore
            )
        pca_df = pd.DataFrame(pca_df_data)

    log_memory_usage("end of load_subgraph_data")
    return results, pca_df


@st.cache_data
def load_subgraph_metadata(file_path: str, subgraph_id: int):
    """Load the metadata for a given subgraph

    Args:
        file_path: Path to the h5 file containing the subgraph
        subgraph_id: The ID of the subgraph to load

    Returns:
        A tuple containing the top 3 tokens and the example context
    """
    base_path = os.path.splitext(file_path)[0]
    chunk_pattern = f"{base_path}_chunk*.h5"
    chunk_files = sorted(glob.glob(chunk_pattern))

    # Find which file contains our subgraph
    target_file = file_path
    if chunk_files:
        for chunk_file in chunk_files:
            with h5py.File(chunk_file, "r") as f:
                if f"subgraph_{subgraph_id}" in f:
                    target_file = chunk_file
                    break

    # Load data from the correct file
    top_3_tokens = []
    example_context = ""
    with h5py.File(target_file, "r") as f:
        group = f[f"subgraph_{subgraph_id}"]
        top_3_tokens = decode_if_bytes(load_dataset(group["top_3_tokens"]))  # type: ignore
        example_context = decode_if_bytes(load_dataset(group["example_context"]))  # type: ignore
        example_context = "".join(example_context)  # type: ignore
    return top_3_tokens, example_context


@st.cache_data
def load_available_subgraphs(file_path: str) -> list[int]:
    """Load the available subgraphs for a given file

    Args:
        file_path: Path to the h5 file containing the subgraphs

    Returns:
        A list of subgraph IDs
    """
    base_path = os.path.splitext(file_path)[0]

    # Check if chunked files exist
    chunk_pattern = f"{base_path}_chunk*.h5"
    chunk_files = sorted(glob.glob(chunk_pattern))

    subgraphs = []
    if chunk_files:
        # Load subgraphs from all chunks
        for chunk_file in chunk_files:
            with h5py.File(chunk_file, "r") as f:
                subgraphs.extend(
                    int(key.split("_")[1])
                    for key in f.keys()
                    if key.startswith("subgraph_")
                )
    else:
        # Load from single file
        with h5py.File(file_path, "r") as f:
            subgraphs = [
                int(key.split("_")[1])
                for key in f.keys()
                if key.startswith("subgraph_")
            ]

    return sorted(subgraphs)


@st.cache_data
def load_thresholded_matrix(file_path: str) -> np.ndarray:
    """Load the thresholded matrix from an NPZ file (deprecated as these files are too large for use in Streamlit)

    Args:
        file_path: Path to the NPZ file containing the thresholded matrix

    Returns:
        The loaded thresholded matrix
    """
    with np.load(file_path) as data:
        # Assuming there's a single array in the NPZ file
        # If there are multiple arrays, you'll need to specify the key
        return data[data.files[0]]


@st.cache_data
def load_sparse_thresholded_matrix(file_path: str) -> sparse.csr_matrix:
    """Load the thresholded matrix from an NPZ file (deprecated as these files are too large for use in Streamlit)

    Args:
        file_path: Path to the NPZ file containing the thresholded matrix

    Returns:
        The loaded thresholded matrix
    """
    return sparse.load_npz(file_path)


@st.cache_data
def load_data(
    file_path: str, subgraph_id: int, config: dict
) -> tuple[dict, pd.DataFrame]:
    """Load the subgraph data and PCA dataframe for a specific subgraph
    (wrapper around load_subgraph_data)

    Args:
        file_path: Path to the h5 file containing the subgraph
        subgraph_id: The ID of the subgraph to load
        config: The streamlit config

    Returns:
        A tuple containing the subgraph data and the PCA dataframe
    """
    log_memory_usage("start of load_data")
    results, pca_df = load_subgraph_data(file_path, subgraph_id, config)
    log_memory_usage("end of load_data")
    return results, pca_df


@st.cache_data
def get_available_sizes(
    results_root: str,
    sae_id_neat: str,
    n_batches_reconstruction: int,
    max_examples: str,
) -> list[int]:
    """Get all available subgraph sizes from the directory"""
    base_path = pj(results_root, f"{sae_id_neat}_pca_for_streamlit")

    # Find both regular and chunked files
    pattern = pj(
        base_path,
        f"{max_examples}graph_analysis_results_size_*_nbatch_{n_batches_reconstruction}*.h5",
    )
    files = glob.glob(pattern)

    # Extract sizes, handling both regular and chunked files
    sizes = set()
    for f in files:
        # Extract size from filename, ignoring potential _chunk suffix
        match = re.search(r"size_(\d+)_nbatch_", f)
        if match:
            sizes.add(int(match.group(1)))

    return sorted(list(sizes))


#### Neuronpedia interface ####


def get_neuronpedia_embed_url(
    model: str, sae_release: str, feature_idx: int, sae_id: str
) -> str:
    """Generate the correct Neuronpedia embed URL based on model and SAE release"""
    base_url = "https://neuronpedia.org"
    path = None
    if model == "gpt2-small":
        if sae_release == "res-jb":
            # Extract layer number from sae_id (e.g., "blocks_7_hook_resid_pre" -> 7)
            layer = re.search(r"blocks_(\d+)_", sae_id).group(1)  # type: ignore
            path = f"{model}/{layer}-res-jb/{feature_idx}"
        elif sae_release == "res-jb-feature-splitting":
            # Extract layer and width (e.g., "blocks_8_hook_resid_pre_24576" -> layer=8, width=24576)
            layer = re.search(r"blocks_(\d+)_", sae_id).group(1)  # type: ignore
            width = re.search(r"_(\d+)$", sae_id).group(1)  # type: ignore
            path = f"{model}/{layer}-res_fs{width}-jb/{feature_idx}"
    elif model == "gemma-2-2b":
        # Extract layer and width (e.g., "layer_20/width_16k/canonical" -> layer=20, width=16k)
        layer = re.search(r"layer_(\d+)", sae_id).group(1)  # type: ignore
        width = re.search(r"width_(\d+k)", sae_id).group(1)  # type: ignore
        path = f"{model}/{layer}-gemmascope-res-{width}/{feature_idx}"
    else:
        raise ValueError(f"Invalid model: {model}")
    embed_params = "?embed=true&embedtest=true&embedexplanation=false&height=300"
    return f"{base_url}/{path}{embed_params}"


#### Plotting Functions ####


def simplify_token_display(tokens: list, remove_counts: bool = True) -> list:
    """Clean up token display by optionally removing count numbers and parentheses.

    Args:
        tokens: List of token strings
        remove_counts: If True, removes count numbers from tokens

    Returns:
        List of cleaned token strings
    """
    if not remove_counts:
        return tokens

    cleaned = [str(token) for token in tokens]
    # Remove count numbers and clean up formatting
    cleaned = [re.sub(r'(?<=", )\d+(?=\))', "", token) for token in cleaned]
    cleaned = [re.sub(r"(?<=', )\d+(?=\))", "", token) for token in cleaned]
    cleaned = [re.sub(r"^\(", "", token) for token in cleaned]
    cleaned = [re.sub(r"\)$", "", token) for token in cleaned]

    return list(cleaned)


def update_url_params(key, value):
    """Update URL parameters without triggering a reload"""
    current_params = st.query_params.to_dict()
    current_params[key] = value
    st.query_params.update(current_params)


def generate_color_palette(n_colors: int) -> list[str]:
    # Start with qualitative color scales
    colors = (
        px.colors.qualitative.Plotly
        + px.colors.qualitative.D3
        + px.colors.qualitative.G10
        + px.colors.qualitative.T10
        + px.colors.qualitative.Alphabet
    )

    # If we need more colors, interpolate between existing ones
    if n_colors > len(colors) and n_colors is not None:
        colors_list = px.colors.n_colors(
            "rgb(5, 200, 200)", "rgb(200, 10, 10)", max(n_colors, len(colors))
        )
        return list(colors_list)[:n_colors] if colors_list else colors[:n_colors]
    else:
        return colors[:n_colors]


def plot_pca_2d(pca_df, max_feature_info, fs_splitting_nodes, pc_x="PC2", pc_y="PC3"):
    # Extract max feature indices and whether they're in the graph
    max_feature_indices = max_feature_info[:, 1].astype(int)
    max_feature_in_graph = max_feature_info[:, 2].astype(bool)

    # Create a color map for fs_splitting_nodes
    unique_features = np.unique(fs_splitting_nodes)
    n_unique = len(unique_features)

    color_palette = generate_color_palette(n_unique)
    color_map = dict(zip(unique_features, color_palette))

    # Create figure with points grouped by feature for legend
    fig = go.Figure()

    # Add grey points for features not in graph
    grey_points = ~max_feature_in_graph
    if any(grey_points):
        fig.add_trace(
            go.Scatter(
                x=pca_df.loc[grey_points, pc_x],
                y=pca_df.loc[grey_points, pc_y],
                mode="markers",
                marker=dict(color="grey"),
                name="Not in graph",
                hovertemplate=(
                    "Token: %{customdata[0]}<br>"
                    "Context: %{customdata[1]}<br>"
                    "Feature: %{customdata[2]}<br>"
                    "<extra></extra>"
                ),
                customdata=np.column_stack(
                    (
                        pca_df.loc[grey_points, "tokens"],
                        pca_df.loc[grey_points, "context"],
                        max_feature_indices[grey_points],
                    )
                ),
            )
        )

    # Add points for each feature in fs_splitting_nodes
    for feature in unique_features:
        feature_points = (max_feature_indices == feature) & max_feature_in_graph
        if any(feature_points):
            fig.add_trace(
                go.Scatter(
                    x=pca_df.loc[feature_points, pc_x],
                    y=pca_df.loc[feature_points, pc_y],
                    mode="markers",
                    marker=dict(color=color_map[feature]),
                    name=f"Feature {feature}",
                    hovertemplate=(
                        "Token: %{customdata[0]}<br>"
                        "Context: %{customdata[1]}<br>"
                        "Feature: %{customdata[2]}<br>"
                        "<extra></extra>"
                    ),
                    customdata=np.column_stack(
                        (
                            pca_df.loc[feature_points, "tokens"],
                            pca_df.loc[feature_points, "context"],
                            max_feature_indices[feature_points],
                        )
                    ),
                )
            )

    fig.update_layout(
        xaxis_title=pc_x,
        yaxis_title=pc_y,
        hovermode="closest",
        hoverdistance=5,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            font=dict(size=10),
        ),
        margin=dict(l=40, r=40, t=40, b=60),
        autosize=True,
    )

    return fig, color_map


def plot_legend(color_map: dict[str, str]) -> go.Figure:
    fig = go.Figure()

    for feature, color in color_map.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=color),
                name=f"Feature {feature}",
                legendgroup=f"Feature {feature}",
                showlegend=True,
            )
        )

    fig.update_layout(
        height=400,
        width=300,
        title="Legend",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=0),
    )

    return fig


def plot_feature_activations(
    all_graph_feature_acts, point_index, fs_splitting_nodes, context=None
):
    if point_index is None:
        fig = go.Figure()
        fig.update_layout(
            autosize=True,
            # height=600,
            # width=800,
            title="Instructions",
            annotations=[
                dict(
                    text="Click on a point in the PCA plot to see its feature activations",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                )
            ],
        )
        return fig

    activations = all_graph_feature_acts[point_index]
    trace = go.Bar(
        x=[f"Feature {i}" for i in fs_splitting_nodes],
        y=activations,
        marker_color="blue",
    )

    fig = go.Figure(data=[trace])
    fig.update_layout(
        height=600,
        width=800,
        title=f'Context: "{context}"' if context else "Feature Activations",
        xaxis_title="Feature",
        yaxis_title="Activation",
        hovermode=False,
    )
    return fig


#### Streamlit controls ####


def load_recommended_views(config: dict) -> dict:
    """Load recommended views from config"""
    return config.get("recommended_views", {})


def apply_recommended_view(view_config: dict) -> None:
    """Update URL parameters for a recommended view and trigger rerun"""
    new_params = {
        "model": view_config["model"],
        "sae_release": view_config["sae_release"],
        "sae_id": view_config["sae_id"],
        "size": str(view_config["size"]),
        "subgraph": str(view_config["subgraph"]),
    }
    st.query_params.update(new_params)
    st.rerun()


#### Dev and Utils ####


def log_memory_usage(location: str) -> None:
    """Log current memory usage"""
    process = psutil.Process()
    memory_gb = process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB
    logging.info(f"Memory usage at {location}: {memory_gb:.2f} GB")


def split_large_h5_files(directory: str, max_size_mb: int = 100) -> None:
    """Split HDF5 files larger than max_size_mb into smaller chunks.

    Args:
        directory: Path to directory containing h5 files
        max_size_mb: Maximum file size in MB (default: 100)
    """
    for filename in os.listdir(directory):
        if not filename.endswith(".h5"):
            continue

        file_path = os.path.join(directory, filename)
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

        if file_size_mb <= max_size_mb:
            continue

        # Calculate number of chunks needed
        n_chunks = int(np.ceil(file_size_mb / max_size_mb))

        with h5py.File(file_path, "r") as source:
            # Get all subgraph keys
            subgraph_keys = [k for k in source.keys() if k.startswith("subgraph_")]
            chunk_size = len(subgraph_keys) // n_chunks

            # Split into chunks
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = (
                    start_idx + chunk_size if i < n_chunks - 1 else len(subgraph_keys)
                )
                chunk_keys = subgraph_keys[start_idx:end_idx]

                # Create new file for chunk
                chunk_filename = f"{os.path.splitext(filename)[0]}_chunk{i+1}.h5"
                chunk_path = os.path.join(directory, chunk_filename)

                with h5py.File(chunk_path, "w") as target:
                    # Copy selected subgraphs to new file
                    for key in chunk_keys:
                        source.copy(source[key], target, key)

        logging.info(f"Split {filename} into {n_chunks} chunks")
