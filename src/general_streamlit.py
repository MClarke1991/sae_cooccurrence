import glob
import re
from os.path import join as pj

import h5py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit_plotly_events as spe

from sae_cooccurrence.normalised_cooc_functions import neat_sae_id
from sae_cooccurrence.streamlit import load_streamlit_config
from sae_cooccurrence.utils.set_paths import get_git_root


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


def load_subgraph_data(file_path, subgraph_id, load_options):
    with h5py.File(file_path, "r") as f:
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

        return results, pca_df


@st.cache_data
def load_data(file_path, subgraph_id, config):
    results, pca_df = load_subgraph_data(file_path, subgraph_id, config)
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
        hovermode="closest",
        hoverdistance=5,  # Reduce hover sensitivity
    )

    # Add this to make hover more responsive
    fig.update_traces(
        hovertemplate=(
            "Token: %{customdata[0]}<br>"
            "Context: %{customdata[1]}<br>"
            "Feature: %{customdata[2]}<br>"
            "<extra></extra>"  # This removes the secondary box
        )
    )

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
        hovermode=False,
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
    # Add query parameter handling at the start of main()
    query_params = st.query_params

    git_root = get_git_root()
    config = load_streamlit_config(pj(git_root, "src", "config_pca_streamlit.toml"))
    load_options = config["processing"]["load_options"]
    model_to_batch_size = config["models"]["batch_sizes"]

    st.set_page_config(layout="wide")
    st.title("PCA Visualization with Feature Activations")

    # Fix query parameter handling for model selection
    default_model_idx = 0
    if "model" in query_params:
        try:
            model_param = query_params["model"]
            # Handle both list and string cases
            model_value = (
                model_param[0] if isinstance(model_param, list) else model_param
            )
            default_model_idx = ["gpt2-small", "gemma-2-2b"].index(model_value)
        except (ValueError, IndexError):
            # If the model parameter is invalid, use default
            default_model_idx = 0

    model = st.selectbox(
        "Select model",
        ["gpt2-small", "gemma-2-2b"],
        index=default_model_idx,
        format_func=lambda x: f"{x} (batch size: {model_to_batch_size[x]})",
    )

    # Load model configurations from config
    # Get batch size for selected model
    n_batches_reconstruction = model_to_batch_size[model]
    model_to_releases = config["models"]["releases"]
    sae_release_to_ids = config["models"]["sae_ids"]

    available_sae_releases = model_to_releases[model]
    default_release_idx = 0
    if "sae_release" in query_params:
        try:
            release_param = query_params["sae_release"]
            release_value = (
                release_param[0] if isinstance(release_param, list) else release_param
            )
            default_release_idx = available_sae_releases.index(release_value)
        except (ValueError, IndexError):
            # If the release parameter is invalid, use default
            default_release_idx = 0

    sae_release = st.selectbox(
        "Select SAE release", available_sae_releases, index=default_release_idx
    )

    # Similarly for SAE ID selection
    available_sae_ids = sae_release_to_ids[sae_release]
    default_sae_idx = 0
    if "sae_id" in query_params:
        try:
            sae_param = query_params["sae_id"]
            sae_value = sae_param[0] if isinstance(sae_param, list) else sae_param
            neat_ids = [neat_sae_id(id) for id in available_sae_ids]
            default_sae_idx = neat_ids.index(sae_value)
        except (ValueError, IndexError):
            # If the SAE ID parameter is invalid, use default
            default_sae_idx = 0

    sae_id = st.selectbox(
        "Select SAE ID",
        [neat_sae_id(id) for id in available_sae_ids],
        index=default_sae_idx,
    )
    results_root = pj(
        git_root,
        f"results/{model}/{sae_release}/{sae_id}",
    )

    # Add size selection before loading subgraphs
    available_sizes = get_available_sizes(
        results_root, sae_id, n_batches_reconstruction
    )
    default_size_idx = 0
    if "size" in query_params:
        try:
            size_param = query_params["size"]
            size_value = size_param[0] if isinstance(size_param, list) else size_param
            default_size_idx = available_sizes.index(int(size_value))
        except (ValueError, IndexError):
            # If the size parameter is invalid, use default
            default_size_idx = 0

    selected_size = st.selectbox(
        "Select subgraph size",
        options=available_sizes,
        index=default_size_idx,
        format_func=lambda x: f"Size {x}",
        key="size_selector",
    )

    # Update pca_results_path to use selected size
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
    default_subgraph_idx = 0
    if "subgraph" in query_params:
        try:
            subgraph_param = query_params["subgraph"]
            subgraph_value = (
                subgraph_param[0]
                if isinstance(subgraph_param, list)
                else subgraph_param
            )
            default_subgraph_idx = available_subgraphs.index(int(subgraph_value))
        except (ValueError, IndexError):
            # If the subgraph parameter is invalid, use default
            default_subgraph_idx = 0

    selected_subgraph = st.selectbox(
        "Select a subgraph",
        options=[opt["value"] for opt in subgraph_options],
        index=default_subgraph_idx,
        format_func=lambda x: next(
            opt["label"] for opt in subgraph_options if opt["value"] == x
        ),
        key="subgraph_selector",
    )

    # Add a section to display the shareable link
    current_params = {
        "model": model,
        "sae_release": sae_release,
        "sae_id": sae_id,
        "size": str(selected_size),
        "subgraph": str(selected_subgraph),
    }

    query_string = "&".join([f"{k}={v}" for k, v in current_params.items()])
    base_url = "https://saecoocpocapp-6ict2wobwrxf52wrrugm8u.streamlit.app/"
    st.write("### Shareable Link")
    st.text_input(
        "Copy this link to share current view:",
        f"{base_url}?{query_string}",
        key="share_link",
    )

    activation_threshold = 1.5
    activation_threshold_safe = str(activation_threshold).replace(".", "_")
    node_df = pd.read_csv(
        pj(results_root, f"dataframes/node_info_df_{activation_threshold_safe}.csv")
    )
    fs_splitting_nodes = node_df.query("subgraph_id == @selected_subgraph")[
        "node_id"
    ].tolist()

    results, pca_df = load_data(pca_results_path, selected_subgraph, load_options)
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
