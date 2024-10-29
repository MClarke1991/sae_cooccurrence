import glob
import os
import re
import tempfile
from os.path import join as pj

import gdown
import h5py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit_plotly_events as spe

from sae_cooccurrence.normalised_cooc_functions import neat_sae_id
from sae_cooccurrence.utils.set_paths import get_git_root


def download_from_gdrive(url):
    """Download file from Google Drive directly into memory using gdown."""
    temp_dir = tempfile.mkdtemp()
    try:
        gdown.download_folder(url, output=temp_dir, quiet=False)
        # Find the .h5 file in the downloaded folder
        h5_files = glob.glob(os.path.join(temp_dir, "**/*.h5"), recursive=True)
        if not h5_files:
            raise FileNotFoundError("No .h5 files found in downloaded data")
        return h5_files[0]
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return None


def generate_color_palette(n_colors):
    # Start with qualitative color scales
    colors = (
        px.colors.qualitative.Plotly
        + px.colors.qualitative.D3
        + px.colors.qualitative.G10
        + px.colors.qualitative.T10
        + px.colors.qualitative.Alphabet
    )

    # If we need more colors, interpolate between existing ones
    if n_colors > len(colors):
        return n_colors(
            "rgb(5, 200, 200)", "rgb(200, 10, 10)", max(n_colors, len(colors))
        )[:n_colors]
    else:
        return colors[:n_colors]


def load_dataset(dataset):
    if dataset.shape == ():  # Scalar dataset
        return dataset[()]
    else:  # Array dataset
        return dataset[:]


def decode_if_bytes(data):
    if isinstance(data, bytes):
        return data.decode("utf-8")
    elif isinstance(data, np.ndarray) and data.dtype.char == "S":
        return np.char.decode(data, "utf-8")
    return data


def load_subgraph_data(file_path, subgraph_id):
    with h5py.File(file_path, "r") as f:
        group = f[f"subgraph_{subgraph_id}"]

        # Load results
        results = {
            "all_fired_tokens": decode_if_bytes(
                load_dataset(group["all_fired_tokens"])  # type: ignore
            ),  # type: ignores
            "all_reconstructions": load_dataset(group["all_reconstructions"]),  # type: ignore
            "all_graph_feature_acts": load_dataset(group["all_graph_feature_acts"]),  # type: ignore
            # 'all_feature_acts': load_dataset(group['all_feature_acts']),
            "all_max_feature_info": load_dataset(group["all_max_feature_info"]),  # type: ignore
            "all_examples_found": load_dataset(group["all_examples_found"]),  # type: ignore
        }

        # Load all_token_dfs
        token_dfs_group = group["all_token_dfs"]  # type: ignore
        all_token_dfs_data = {}
        for column in token_dfs_group.keys():  # type: ignore
            all_token_dfs_data[column] = decode_if_bytes(
                load_dataset(token_dfs_group[column])  # type: ignore
            )  # type: ignore
        results["all_token_dfs"] = pd.DataFrame(all_token_dfs_data)

        # Load pca_df
        pca_df_group = group["pca_df"]  # type: ignore
        pca_df_data = {}
        for column in pca_df_group.keys():  # type: ignore
            pca_df_data[column] = decode_if_bytes(load_dataset(pca_df_group[column]))  # type: ignore
        pca_df = pd.DataFrame(pca_df_data)

    return results, pca_df


@st.cache_data
def load_data(file_path, subgraph_id):
    results, pca_df = load_subgraph_data(file_path, subgraph_id)
    return results, pca_df


def plot_pca_2d(pca_df, max_feature_info, fs_splitting_nodes):
    # Extract max feature indices and whether they're in the graph
    max_feature_indices = max_feature_info[:, 1].astype(int)
    max_feature_in_graph = max_feature_info[:, 2].astype(bool)

    # Create a color map for fs_splitting_nodes
    unique_features = np.unique(fs_splitting_nodes)
    n_unique = len(unique_features)

    color_palette = generate_color_palette(n_unique)
    color_map = dict(zip(unique_features, color_palette))

    # Assign colors based on whether the max feature is in fs_splitting_nodes
    colors = [
        "grey" if not in_graph else color_map.get(feature, "grey")
        for feature, in_graph in zip(max_feature_indices, max_feature_in_graph)
    ]

    # Add max_feature_indices to pca_df for hover data
    pca_df["max_feature_index"] = max_feature_indices

    fig = px.scatter(
        pca_df,
        x="PC2",
        y="PC3",
        hover_data=["tokens", "context", "max_feature_index"],
        labels={"max_feature_index": "Most Active Feature"},
        custom_data=["max_feature_index"],
    )

    # Update marker colors
    fig.update_traces(marker=dict(color=colors))

    fig.update_layout(
        height=600,
        width=800,
        title="PCA Plot (PC2 vs PC3)",
        xaxis_title="PC2",
        yaxis_title="PC3",
        showlegend=False,
    )  # Hide the legend in the main plot

    return fig, color_map


def plot_legend(color_map):
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
        # Create an empty figure with instructions
        fig = go.Figure()
        fig.update_layout(
            height=600,
            width=800,
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

    # Original plotting code
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
    )
    return fig


@st.cache_data
def load_available_subgraphs(file_path):
    with h5py.File(file_path, "r") as f:
        return sorted(
            [int(key.split("_")[1]) for key in f.keys() if key.startswith("subgraph_")]
        )


@st.cache_data
def load_subgraph_metadata(file_path, subgraph_id):
    top_3_tokens = []
    example_context = ""
    with h5py.File(file_path, "r") as f:
        group = f[f"subgraph_{subgraph_id}"]
        top_3_tokens = decode_if_bytes(load_dataset(group["top_3_tokens"]))  # type: ignore
        example_context = decode_if_bytes(load_dataset(group["example_context"]))  # type: ignore
        example_context = "".join(example_context)  # type: ignore
    return top_3_tokens, example_context


@st.cache_data
def get_available_sizes(results_root, sae_id_neat, n_batches_reconstruction):
    """Get all available subgraph sizes from the directory"""
    base_path = pj(results_root, f"{sae_id_neat}_pca_for_streamlit")
    files = glob.glob(
        pj(
            base_path,
            f"graph_analysis_results_size_*_nbatch_{n_batches_reconstruction}.h5",
        )
    )
    sizes = [int(re.search(r"size_(\d+)_nbatch_", f).group(1)) for f in files]  # type: ignore
    return sorted(sizes)


def main():
    use_gdrive = True

    st.set_page_config(layout="wide")
    st.title("PCA Visualization with Feature Activations")

    git_root = get_git_root()
    MODEL_DATA_URLS = pd.read_csv(pj(git_root, "data", "gdown_lookup.csv"))

    model_to_batch_size = {
        "gpt2-small": 100,
        "gemma-2-2b": 10,
    }

    model = st.selectbox(
        "Select model",
        ["gpt2-small", "gemma-2-2b"],
        format_func=lambda x: f"{x} (batch size: {model_to_batch_size[x]})",
    )

    model_to_releases = {
        "gpt2-small": ["res-jb", "res-jb-feature-splitting"],
        "gemma-2-2b": [
            "gemma-scope-2b-pt-res-canonical",
            "gemma-scope-2b-pt-res",
        ],
    }

    sae_release_to_ids = {
        "res-jb": ["blocks.0.hook_resid_pre"],
        "res-jb-feature-splitting": [
            "blocks.8.hook_resid_pre_24576",
        ],
        "gemma-scope-2b-pt-res-canonical": [
            "layer_0/width_16k/canonical",
            "layer_12/width_16k/canonical",
            "layer_12/width_32k/canonical",
            "layer_12/width_65k/canonical",
            "layer_18/width_16k/canonical",
            "layer_21/width_16k/canonical",
        ],
        "gemma-scope-2b-pt-res": [
            "layer_12/width_16k/average_l0_22",
            "layer_12/width_16k/average_l0_41",
            "layer_12/width_16k/average_l0_82",
            "layer_12/width_16k/average_l0_176",
            "layer_12/width_16k/average_l0_445",
        ],
    }

    # Get batch size for selected model
    n_batches_reconstruction = model_to_batch_size[model]

    available_sae_releases = model_to_releases[model]
    sae_release = st.selectbox("Select SAE release", available_sae_releases)
    available_sae_ids = sae_release_to_ids[sae_release]
    sae_id = st.selectbox(
        "Select SAE ID", [neat_sae_id(id) for id in available_sae_ids]
    )
    results_root = pj(
        git_root,
        f"results/{model}/{sae_release}/{sae_id}",
    )

    # Add size selection before loading subgraphs
    available_sizes = get_available_sizes(
        results_root, sae_id, n_batches_reconstruction
    )
    selected_size = st.selectbox(
        "Select subgraph size",
        options=available_sizes,
        format_func=lambda x: f"Size {x}",
        key="size_selector",
    )

    if use_gdrive:
        matching_urls = MODEL_DATA_URLS.query(
            "model == @model and release == @sae_release and sae_id == @sae_id"
        )
        if matching_urls.empty:
            st.error(
                f"No data found for model={model}, release={sae_release}, sae_id={sae_id}"
            )
            return
        url = matching_urls["url"].iloc[0]
        pca_results_path = download_from_gdrive(url)
        if pca_results_path is None:
            st.error("Failed to download data. Please try again or use local data.")
            return
    else:
        results_root = pj(
            git_root,
            f"results/{model}/{sae_release}/{sae_id}",
        )
        pca_results_path = pj(
            results_root,
            f"{sae_id}_pca_for_streamlit",
            f"graph_analysis_results_size_{selected_size}_nbatch_{n_batches_reconstruction}.h5",
        )

    # Load available subgraphs
    available_subgraphs = load_available_subgraphs(pca_results_path)

    # Load metadata for all subgraphs
    subgraph_options = []
    for sg_id in available_subgraphs:
        top_3_tokens, example_context = load_subgraph_metadata(pca_results_path, sg_id)
        label = f"Subgraph {sg_id} - Top tokens: {', '.join(top_3_tokens)} | Example: {example_context}"  # type: ignore
        subgraph_options.append({"label": label, "value": sg_id})

    # Dropdown for subgraph selection
    selected_subgraph = st.selectbox(
        "Select a subgraph",
        options=[opt["value"] for opt in subgraph_options],
        format_func=lambda x: next(
            opt["label"] for opt in subgraph_options if opt["value"] == x
        ),
        key="subgraph_selector",
    )

    activation_threshold = 1.5
    activation_threshold_safe = str(activation_threshold).replace(".", "_")
    node_df = pd.read_csv(
        pj(results_root, f"dataframes/node_info_df_{activation_threshold_safe}.csv")
    )
    fs_splitting_nodes = node_df.query("subgraph_id == @selected_subgraph")[
        "node_id"
    ].tolist()

    results, pca_df = load_data(pca_results_path, selected_subgraph)
    # feature_activations = results['all_feature_acts']

    col1, col2, col3 = st.columns([3, 1, 2])

    with col1:
        st.write("## PCA Plot (PC2 vs PC3)")
        st.write(
            "Color represents character count. Click on a point to see its feature activations."
        )
        pca_plot, color_map = plot_pca_2d(
            pca_df=pca_df,
            max_feature_info=results["all_max_feature_info"],
            fs_splitting_nodes=fs_splitting_nodes,
        )

        # Use a unique key for plotly_events based on the selected subgraph
        selected_points = spe.plotly_events(
            pca_plot,
            click_event=True,
            override_height=600,
            key=f"pca_plot_{selected_subgraph}",
        )

    with col2:
        st.write("## Legend")
        legend_fig = plot_legend(color_map)
        st.plotly_chart(legend_fig, use_container_width=True)

    with col3:
        st.write("## Feature Activations (Only for fs_splitting_nodes)")

        if not selected_points:
            # Show empty plot with instructions when no point is selected
            feature_plot = plot_feature_activations(
                results["all_graph_feature_acts"],
                point_index=None,
                fs_splitting_nodes=fs_splitting_nodes,
            )
            st.plotly_chart(feature_plot, use_container_width=True)
        else:
            # Rest of the existing code for when a point is selected
            selected_x = selected_points[0]["x"]
            selected_y = selected_points[0]["y"]
            matching_points = pca_df[
                (pca_df["PC2"] == selected_x) & (pca_df["PC3"] == selected_y)
            ]

            if not matching_points.empty:
                point_index = matching_points.index[0]

                st.write("## Selected Point Info")
                st.write(f"Token: {pca_df.loc[point_index, 'tokens']}")
                st.write(f"Context: {pca_df.loc[point_index, 'context']}")

                context = pca_df.loc[point_index, "context"]
                feature_plot = plot_feature_activations(
                    results["all_graph_feature_acts"],
                    point_index,
                    fs_splitting_nodes,
                    context,
                )
                st.plotly_chart(feature_plot, use_container_width=True)
            else:
                st.write(
                    "The selected point is not in the current dataset. Please select a new point."
                )


if __name__ == "__main__":
    main()
