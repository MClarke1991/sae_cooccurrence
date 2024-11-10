import glob
import logging
import re
from html import escape
from os.path import join as pj

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import streamlit_plotly_events as spe
from scipy import sparse

from sae_cooccurrence.normalised_cooc_functions import (
    create_results_dir,
    neat_sae_id,
)
from sae_cooccurrence.pca import (
    generate_subgraph_plot_data_sparse,
    plot_pca_feature_strength_streamlit,
    plot_subgraph_interactive_from_nx,
)
from sae_cooccurrence.streamlit import (
    decode_if_bytes,
    load_dataset,
    load_streamlit_config,
    load_subgraph_data,
    log_memory_usage,
    plot_feature_activations,
    plot_pca_2d,
)
from sae_cooccurrence.utils.set_paths import get_git_root


@st.cache_data
def load_data(file_path, subgraph_id, config):
    log_memory_usage("start of load_data")
    results, pca_df = load_subgraph_data(file_path, subgraph_id, config)
    log_memory_usage("end of load_data")
    return results, pca_df


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


@st.cache_data
def load_available_subgraphs(file_path):
    with h5py.File(file_path, "r") as f:
        return sorted(
            [int(key.split("_")[1]) for key in f.keys() if key.startswith("subgraph_")]
        )


@st.cache_data
def load_thresholded_matrix(file_path: str) -> np.ndarray:
    # Load and return the actual array data from the NPZ file
    with np.load(file_path) as data:
        # Assuming there's a single array in the NPZ file
        # If there are multiple arrays, you'll need to specify the key
        return data[data.files[0]]


@st.cache_data
def load_sparse_thresholded_matrix(file_path: str) -> sparse.csr_matrix:
    return sparse.load_npz(file_path)


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
def get_available_sizes(
    results_root, sae_id_neat, n_batches_reconstruction, max_examples
):
    """Get all available subgraph sizes from the directory"""
    base_path = pj(results_root, f"{sae_id_neat}_pca_for_streamlit")
    files = glob.glob(
        pj(
            base_path,
            f"{max_examples}graph_analysis_results_size_*_nbatch_{n_batches_reconstruction}.h5",
        )
    )
    sizes = [int(re.search(r"size_(\d+)_nbatch_", f).group(1)) for f in files]  # type: ignore
    return sorted(sizes)


def get_neuronpedia_embed_url(model, sae_release, feature_idx, sae_id):
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


def update_url_params(key, value):
    """Update URL parameters without triggering a reload"""
    current_params = st.query_params.to_dict()
    current_params[key] = value
    st.query_params.update(current_params)


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


def main():
    logging.basicConfig(level=logging.INFO)
    log_memory_usage("start of main")

    query_params = st.query_params

    # if "point_x" in st.query_params and "point_y" in st.query_params:
    #     initial_point_x = float(st.query_params["point_x"])
    #     initial_point_y = float(st.query_params["point_y"])

    st.set_page_config(
        page_title="Feature Cooccurrence Explorer",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .title-text {
            font-size: 42px;
            font-weight: 600;
            color: #1E1E1E;
            padding-bottom: 20px;
        }
        .subtitle-text {
            font-size: 24px;
            font-weight: 500;
            color: #4A4A4A;
            padding-bottom: 10px;
        }
        .section-text {
            font-size: 20px;
            font-weight: 500;
            color: #2C3E50;
            padding-bottom: 10px;
        }
        .stSelectbox {
            padding-bottom: 15px;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="title-text">SAE Latent Cooccurrence Explorer</p>',
        unsafe_allow_html=True,
    )
    with st.expander("What is SAE Latent Co-occurrence?"):
        st.markdown("""
            SAE (Sparse Autoencoder) latent co-occurrence analysis helps us understand how different features learned by 
            the autoencoder tend to activate together. When two features frequently activate at the same time across many examples,
            we say they "co-occur". This tool visualizes these co-occurrence patterns to help understand the relationships
            between different learned SAE latents, and demonstrates how this co-occurrence maps interpretable subspaces. 
            
            For a given cluster of SAE latents we:

            - Search through the training data for examples of prompts that activate these latents
            - Use a PCA to represent the vectors made up of the same of feature activations from only the SAE latents in that cluster
            - This is to explore if these latents are more explicable as a group
            
            For a cluster of co-occurring latents we show this PCA plot, and the corresponding co-occurrence graph. 
            Click on any point in the PCA plot to see the relative strength of activations for that token and context, and 
            how these separate across the PCA dimensions. We also show the Neuronpedia links for the SAE latents in the cluster to show their general properties. 
        """)

    git_root = get_git_root()
    config = load_streamlit_config(
        pj(git_root, "src", "config_pca_streamlit_maxexamples.toml")
    )
    load_options = config["processing"]["load_options"]
    models = config["streamlit"]["models"]
    model_to_batch_size = config["models"]["pca_batch_sizes"]
    use_max_examples = config["processing"]["load_options"]["use_max_examples"]
    show_max_examples = config["streamlit"]["dev"]["show_max_examples"]
    show_batch_size = config["streamlit"]["dev"]["show_batch_size"]
    model_to_max_examples = config["models"]["max_examples"]

    with st.sidebar:
        st.markdown(
            '<p class="subtitle-text">Configuration</p>', unsafe_allow_html=True
        )
        default_model_idx = 0
        if "model" in query_params:
            try:
                model_param = query_params["model"]
                # Handle both list and string cases
                model_value = (
                    model_param[0] if isinstance(model_param, list) else model_param
                )
                default_model_idx = list(models.values()).index(model_value)
            except (ValueError, IndexError):
                default_model_idx = 0

        model = st.selectbox(
            "Model",
            list(models.values()),
            index=default_model_idx,
            key="model_selector",
            on_change=lambda: update_url_params(
                "model", st.session_state.model_selector
            ),
        )

        # Load model configurations from config
        model_to_releases = config["models"]["releases"]
        sae_release_to_ids = config["models"]["sae_ids"]

        available_sae_releases = model_to_releases[model]

        n_batches_reconstruction = model_to_batch_size[model]

        if use_max_examples:
            max_examples = str(model_to_max_examples[model]) + "cap_"
        else:
            max_examples = ""

        default_release_idx = 0
        if "sae_release" in query_params:
            try:
                release_param = query_params["sae_release"]
                release_value = (
                    release_param[0]
                    if isinstance(release_param, list)
                    else release_param
                )
                default_release_idx = available_sae_releases.index(release_value)
            except (ValueError, IndexError):
                default_release_idx = 0

        sae_release = st.selectbox(
            "SAE Release",
            available_sae_releases,
            index=default_release_idx,
            key="sae_release_selector",
            on_change=lambda: update_url_params(
                "sae_release", st.session_state.sae_release_selector
            ),
        )

        available_sae_ids = sae_release_to_ids[sae_release]

        default_sae_idx = 0
        if "sae_id" in query_params:
            try:
                sae_param = query_params["sae_id"]
                sae_value = sae_param[0] if isinstance(sae_param, list) else sae_param
                neat_ids = [neat_sae_id(id) for id in available_sae_ids]
                default_sae_idx = neat_ids.index(sae_value)
            except (ValueError, IndexError):
                default_sae_idx = 0

        sae_id = st.selectbox(
            "SAE ID",
            [neat_sae_id(id) for id in available_sae_ids],
            index=default_sae_idx,
            key="sae_id_selector",
            on_change=lambda: update_url_params(
                "sae_id", st.session_state.sae_id_selector
            ),
        )

        release_to_generation_batch_size = config["releases"]["generation_batch_sizes"]
        n_batches_generation = release_to_generation_batch_size[sae_release]

        results_root = create_results_dir(
            model, sae_release, sae_id, n_batches_generation
        )

        # st.markdown('<p class="section-text">Size Settings</p>', unsafe_allow_html=True)
        available_sizes = get_available_sizes(
            results_root, sae_id, n_batches_reconstruction, max_examples
        )

        default_size_idx = 0
        if "size" in query_params:
            try:
                size_param = query_params["size"]
                size_value = (
                    size_param[0] if isinstance(size_param, list) else size_param
                )
                default_size_idx = available_sizes.index(int(size_value))
            except (ValueError, IndexError):
                default_size_idx = 0

        selected_size = st.selectbox(
            "Cluster Size",
            options=available_sizes,
            index=default_size_idx,
            format_func=lambda x: f"Size {x}",
            key="size_selector",
            on_change=lambda: update_url_params("size", st.session_state.size_selector),
        )

    # Update pca_results_path
    pca_results_path = pj(
        results_root,
        f"{sae_id}_pca_for_streamlit",
        f"{max_examples}graph_analysis_results_size_{selected_size}_nbatch_{n_batches_reconstruction}.h5",
    )

    # Load available subgraphs
    available_subgraphs = load_available_subgraphs(pca_results_path)

    # Load metadata for all subgraphs
    subgraph_options = []
    remove_token_counts = config["streamlit"].get("remove_token_counts", True)
    for sg_id in available_subgraphs:
        top_3_tokens, example_context = load_subgraph_metadata(pca_results_path, sg_id)
        top_3_tokens_simple = simplify_token_display(top_3_tokens, remove_token_counts)  # type: ignore
        label = f"Subgraph {sg_id} - Top tokens: {' '.join(top_3_tokens_simple)} | Example: '{example_context}'"
        subgraph_options.append({"label": label, "value": sg_id})

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
            default_subgraph_idx = 0

    selected_subgraph = st.selectbox(
        "Choose a subgraph to visualize",
        options=[opt["value"] for opt in subgraph_options],
        index=default_subgraph_idx,
        format_func=lambda x: next(
            opt["label"] for opt in subgraph_options if opt["value"] == x
        ),
        key="subgraph_selector",
        on_change=lambda: update_url_params(
            "subgraph", st.session_state.subgraph_selector
        ),
    )

    activation_threshold = 1.5
    activation_threshold_safe = str(activation_threshold).replace(".", "_")
    log_memory_usage("before loading node_df")
    node_df = pd.read_csv(
        pj(results_root, f"dataframes/node_info_df_{activation_threshold_safe}.csv")
    )
    log_memory_usage("after loading node_df")
    thresholded_matrix = load_sparse_thresholded_matrix(
        pj(
            results_root,
            f"thresholded_matrices/sparse_thresholded_matrix_{activation_threshold_safe}.npz",
        )
    )
    log_memory_usage("after loading thresholded_matrix")
    fs_splitting_nodes = node_df.query("subgraph_id == @selected_subgraph")[
        "node_id"
    ].tolist()

    log_memory_usage("before loading data")
    results, pca_df = load_data(pca_results_path, selected_subgraph, load_options)
    log_memory_usage("after loading data")

    # Create 2x2 grid layout
    top_left, top_right = st.columns(2)
    bottom_left, bottom_right = st.columns(2)

    # Initialize or get current activations from session state
    if "current_activations" not in st.session_state:
        st.session_state.current_activations = None

    with top_left:
        st.markdown('<p class="section-text">PCA</p>', unsafe_allow_html=True)
        st.markdown(
            """
            The plot below shows the PCA projection of SAE latent activations. 
            Colours represent the most active latent in the cluster. Click on any point to see detailed activations.
            """
        )

        # Add PCA dimension selector
        pca_dims = st.selectbox(
            "Select PCA dimensions to plot:",
            options=["PC1 vs PC2", "PC2 vs PC3", "PC1 vs PC3"],
            index=1,  # Default to PC2 vs PC3
        )

        # Map selection to column names
        dim_mapping = {
            "PC1 vs PC2": ("PC1", "PC2"),
            "PC2 vs PC3": ("PC2", "PC3"),
            "PC1 vs PC3": ("PC1", "PC3"),
        }
        x_dim, y_dim = dim_mapping[pca_dims]

        log_memory_usage("before PCA plot")
        pca_plot, color_map = plot_pca_2d(
            pca_df=pca_df,
            max_feature_info=results["all_max_feature_info"],
            fs_splitting_nodes=fs_splitting_nodes,
            pc_x=x_dim,
            pc_y=y_dim,
        )
        log_memory_usage("after PCA plot")

        selected_points = spe.plotly_events(
            pca_plot,
            click_event=True,
            key=f"pca_plot_{selected_subgraph}",
        )

        # Update activations immediately when point is selected
        if selected_points:
            selected_x = selected_points[0]["x"]
            selected_y = selected_points[0]["y"]
            matching_points = pca_df[
                (pca_df[x_dim] == selected_x) & (pca_df[y_dim] == selected_y)
            ]
            if not matching_points.empty:
                point_index = matching_points.index[0]
                st.session_state.current_activations = results[
                    "all_graph_feature_acts"
                ][point_index]
        else:
            st.session_state.current_activations = None

    with top_right:
        st.markdown(
            '<p class="section-text">SAE Latent Co-occurrence Graph</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            Graph of the co-occurrence relations between latents in this cluster. Node size represents latent density, edge weight represents co-occurrence strength. 
            Click on a point in the PCA plot to see relative strength of latent activations for that token and context. 
            """
        )
        subgraph, subgraph_df = generate_subgraph_plot_data_sparse(
            thresholded_matrix, node_df, selected_subgraph
        )

        _, html = plot_subgraph_interactive_from_nx(
            subgraph=subgraph,
            subgraph_df=subgraph_df,
            node_info_df=node_df,
            plot_token_factors=False,
            height="400px",
            colour_when_inactive=False,
            activation_array=st.session_state.current_activations,
        )
        components.html(html, height=400)

    with bottom_left:
        if selected_points:
            matching_points = pca_df[
                (pca_df[x_dim] == selected_points[0]["x"])
                & (pca_df[y_dim] == selected_points[0]["y"])
            ]
            if not matching_points.empty:
                point_index = matching_points.index[0]
                # Create two columns for token and context display
                st.markdown(f"**Token:** {pca_df.loc[point_index, 'tokens']}")
                # Sanitize the context
                sanitized_context = escape(
                    str(pca_df.loc[point_index, "context"])
                ).replace("\n", "\\n")
                st.markdown(f"**Context:** {sanitized_context}")
        else:
            st.info(
                " 👆 Click on a point in the PCA plot to see token and context details."
            )

        st.markdown(
            '<p class="section-text">All Latent Activations at point</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            The plot below shows the relative strength of SAE latent activations for all latents in the cluster at a particular point in the PCA. 
            """
        )
        # st.write("#")
        if not selected_points:
            # st.info(
            #     "👆 Click on any point in the PCA plot to see its SAE latent activations in the cluster."
            # )
            feature_plot = plot_feature_activations(
                results["all_graph_feature_acts"],
                point_index=None,
                fs_splitting_nodes=fs_splitting_nodes,
            )
            st.plotly_chart(feature_plot, use_container_width=True)
        else:
            update_url_params("point_x", str(selected_points[0]["x"]))
            update_url_params("point_y", str(selected_points[0]["y"]))
            matching_points = pca_df[
                (pca_df[x_dim] == selected_points[0]["x"])
                & (pca_df[y_dim] == selected_points[0]["y"])
            ]
            if not matching_points.empty:
                point_index = matching_points.index[0]
                context = pca_df.loc[point_index, "context"]
                feature_plot = plot_feature_activations(
                    results["all_graph_feature_acts"],
                    point_index,
                    fs_splitting_nodes,
                    context,
                )
                st.plotly_chart(feature_plot, use_container_width=True)

    # with bottom_right:
    # st.markdown('<p class="section-text">Token and Context</p>', unsafe_allow_html=True)
    # if selected_points:
    #     matching_points = pca_df[
    #         (pca_df["PC2"] == selected_points[0]["x"])
    #         & (pca_df["PC3"] == selected_points[0]["y"])
    #     ]
    #     if not matching_points.empty:
    #         point_index = matching_points.index[0]
    #         st.markdown(f"**Token:** {pca_df.loc[point_index, 'tokens']}")
    #         st.markdown(f"**Context:** {pca_df.loc[point_index, 'context']}")
    # else:
    #     st.info("Click on a point in the PCA plot to see token and context details.")
    with bottom_right:
        st.write("#")
        st.markdown(
            '<p class="section-text">SAE Latent Activation Landscape</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            Activation of an SAE latent for all points in the PCA. 
            """
        )
        # Add feature selector dropdown with sorted options
        sorted_features = sorted(fs_splitting_nodes)
        selected_feature = st.selectbox(
            "Select latent to visualize activation strength:",
            options=sorted_features,
            format_func=lambda x: f"SAE Latent {x}",
        )

        # Get index of selected feature in fs_splitting_nodes
        feature_idx = fs_splitting_nodes.index(selected_feature)

        # Get activations for selected feature
        feature_activations = results["all_graph_feature_acts"][:, feature_idx]

        # Create and display the plot
        feature_strength_plot = plot_pca_feature_strength_streamlit(
            pca_df=pca_df,
            feature_activations=feature_activations,
            feature_idx=selected_feature,
        )
        st.plotly_chart(feature_strength_plot, use_container_width=True)

    # Move Neuronpedia section below the quadrants
    st.markdown(
        '<p class="subtitle-text">SAE latent dashboards from Neuronpedia</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Showing visualizations for up to 10 latents from the current graph, sorted by latent index."
    )
    sorted_features = sorted(fs_splitting_nodes)[:10]

    # Create rows of 2 embeds each
    for i in range(0, len(sorted_features), 2):
        col1, col2 = st.columns(2)

        with col1:
            feature_idx = sorted_features[i]
            embed_url = get_neuronpedia_embed_url(
                model, sae_release, feature_idx, sae_id
            )
            st.markdown(f"#### SAE Latent {feature_idx}")
            st.markdown(
                f'<iframe src="{embed_url}" '
                'title="Neuronpedia" '
                'style="height: 300px; width: 100%; border: none;"></iframe>',
                unsafe_allow_html=True,
            )

        with col2:
            if i + 1 < len(sorted_features):
                feature_idx = sorted_features[i + 1]
                embed_url = get_neuronpedia_embed_url(
                    model, sae_release, feature_idx, sae_id
                )
                st.markdown(f"#### SAE Latent {feature_idx}")
                st.markdown(
                    f'<iframe src="{embed_url}" '
                    'title="Neuronpedia" '
                    'style="height: 300px; width: 100%; border: none;"></iframe>',
                    unsafe_allow_html=True,
                )

        # Add shareable link section
    with st.sidebar:
        st.markdown("### Share This View")
        current_params = {
            "model": model,
            "sae_release": sae_release,
            "sae_id": sae_id,
            "size": str(selected_size),
            "subgraph": str(selected_subgraph),
        }

        query_string = "&".join([f"{k}={v}" for k, v in current_params.items()])
        base_url = "https://feature-cooccurrence.streamlit.app/"
        shareable_link = f"{base_url}?{query_string}"

        st.text_input(
            "Copy this link to share current view:", shareable_link, key="share_link"
        )

        st.sidebar.markdown("#### Dev Info")
        if use_max_examples and show_max_examples:
            st.sidebar.info(f"Max number of examples: {model_to_max_examples[model]}")
        elif show_batch_size:
            st.sidebar.info(f"Batch size {model_to_batch_size[model]}")

    log_memory_usage("end of main")


if __name__ == "__main__":
    main()
