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

    # Create figure with points grouped by feature for legend
    fig = go.Figure()

    # Add grey points for features not in graph
    grey_points = ~max_feature_in_graph
    if any(grey_points):
        fig.add_trace(
            go.Scatter(
                x=pca_df.loc[grey_points, "PC2"],
                y=pca_df.loc[grey_points, "PC3"],
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
                    x=pca_df.loc[feature_points, "PC2"],
                    y=pca_df.loc[feature_points, "PC3"],
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
        xaxis_title="PC2",
        yaxis_title="PC3",
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
        # height = 600,
        # width=800,  # You can adjust this value
        # height=800,  # Make height equal to width
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


def main():
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
        '<p class="title-text">Feature Cooccurrence Explorer</p>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    The plot below shows the PCA projection of feature activations. 
    Colors represent different features. Click on any point to see detailed activations.
    """)
    git_root = get_git_root()
    config = load_streamlit_config(
        pj(git_root, "src", "config_pca_streamlit_maxexamples.toml")
    )
    load_options = config["processing"]["load_options"]
    models = config["streamlit"]["models"]
    model_to_batch_size = config["models"]["batch_sizes"]
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
        results_root = pj(
            git_root,
            f"results/{model}/{sae_release}/{sae_id}",
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
            "Subgraph Size",
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
    for sg_id in available_subgraphs:
        top_3_tokens, example_context = load_subgraph_metadata(pca_results_path, sg_id)
        label = f"Subgraph {sg_id} - Top tokens: {', '.join(top_3_tokens)} | Example: {example_context}"  # type: ignore
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

    # Add a section to display the shareable link
    current_params = {
        "model": model,
        "sae_release": sae_release,
        "sae_id": sae_id,
        "size": str(selected_size),
        "subgraph": str(selected_subgraph),
    }

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

    # Main visualization area

    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.markdown('<p class="section-text">PCA</p>', unsafe_allow_html=True)

        pca_plot, color_map = plot_pca_2d(
            pca_df=pca_df,
            max_feature_info=results["all_max_feature_info"],
            fs_splitting_nodes=fs_splitting_nodes,
        )

        selected_points = spe.plotly_events(
            pca_plot,
            click_event=True,
            key=f"pca_plot_{selected_subgraph}",
        )

    with col2:
        st.markdown(
            '<p class="section-text">Feature Activation</p>', unsafe_allow_html=True
        )

        if not selected_points:
            st.info(
                "👆 Click on any point in the PCA plot to see its feature activations."
            )
            feature_plot = plot_feature_activations(
                results["all_graph_feature_acts"],
                point_index=None,
                fs_splitting_nodes=fs_splitting_nodes,
            )
            st.plotly_chart(feature_plot, use_container_width=True)
        else:
            # Add the new URL parameter update code here
            selected_x = selected_points[0]["x"]
            selected_y = selected_points[0]["y"]
            update_url_params("point_x", str(selected_x))
            update_url_params("point_y", str(selected_y))

            # Continue with your existing code
            matching_points = pca_df[
                (pca_df["PC2"] == selected_x) & (pca_df["PC3"] == selected_y)
            ]

            if not matching_points.empty:
                point_index = matching_points.index[0]

                with st.expander("View token and context", expanded=True):
                    st.markdown(f"**Token:** {pca_df.loc[point_index, 'tokens']}")
                    st.markdown(f"**Context:** {pca_df.loc[point_index, 'context']}")

                context = pca_df.loc[point_index, "context"]
                feature_plot = plot_feature_activations(
                    results["all_graph_feature_acts"],
                    point_index,
                    fs_splitting_nodes,
                    context,
                )
                st.plotly_chart(feature_plot, use_container_width=True)
            else:
                st.error(
                    "The selected point is not in the current dataset. Please select a different point."
                )

    st.markdown(
        '<p class="subtitle-text">Feature dashboards from Neuronpedia</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Showing visualizations for up to 10 features from the current graph, sorted by feature index."
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
            st.markdown(f"#### Feature {feature_idx}")
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
                st.markdown(f"#### Feature {feature_idx}")
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


if __name__ == "__main__":
    main()
