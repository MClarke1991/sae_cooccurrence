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


def load_subgraph_data(file_path, subgraph_id):
    with h5py.File(file_path, "r") as f:
        group = f[f"subgraph_{subgraph_id}"]

        # # Load results
   # Load results
        results = {
            # "all_fired_tokens": decode_if_bytes(
            #     load_dataset(group["all_fired_tokens"])  # type: ignore
            # ),  # type: ignores
            # "all_reconstructions": load_dataset(group["all_reconstructions"]),  # type: ignore
            "all_graph_feature_acts": load_dataset(group["all_graph_feature_acts"]),  # type: ignore
            # 'all_feature_acts': load_dataset(group['all_feature_acts']),
            "all_max_feature_info": load_dataset(group["all_max_feature_info"]),  # type: ignore
            # "all_examples_found": load_dataset(group["all_examples_found"]),  # type: ignore
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

    # Create figure with points grouped by feature for legend
    fig = go.Figure()

    # Add grey points for features not in graph
    grey_points = ~max_feature_in_graph
    if any(grey_points):
        fig.add_trace(
            go.Scatter(
                x=pca_df.loc[grey_points, 'PC2'],
                y=pca_df.loc[grey_points, 'PC3'],
                mode='markers',
                marker=dict(color='grey'),
                name='Not in graph',
                hovertemplate=(
                    "Token: %{customdata[0]}<br>"
                    "Context: %{customdata[1]}<br>"
                    "Feature: %{customdata[2]}<br>"
                    "<extra></extra>"
                ),
                customdata=np.column_stack((
                    pca_df.loc[grey_points, 'tokens'],
                    pca_df.loc[grey_points, 'context'],
                    max_feature_indices[grey_points]
                ))
            )
        )

    # Add points for each feature in fs_splitting_nodes
    for feature in unique_features:
        feature_points = (max_feature_indices == feature) & max_feature_in_graph
        if any(feature_points):
            fig.add_trace(
                go.Scatter(
                    x=pca_df.loc[feature_points, 'PC2'],
                    y=pca_df.loc[feature_points, 'PC3'],
                    mode='markers',
                    marker=dict(color=color_map[feature]),
                    name=f'Feature {feature}',
                    hovertemplate=(
                        "Token: %{customdata[0]}<br>"
                        "Context: %{customdata[1]}<br>"
                        "Feature: %{customdata[2]}<br>"
                        "<extra></extra>"
                    ),
                    customdata=np.column_stack((
                        pca_df.loc[feature_points, 'tokens'],
                        pca_df.loc[feature_points, 'context'],
                        max_feature_indices[feature_points]
                    ))
                )
            )

    fig.update_layout(
        height=500,  # Reduced from 600
        width=650,   # Reduced from 800
        title="PCA Plot (PC2 vs PC3)",
        xaxis_title="PC2",
        yaxis_title="PC3",
        hovermode="closest",
        hoverdistance=5,
        # Adjusted legend positioning and style
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            font=dict(size=10)  # Smaller font size for legend
        ),
        # Add margins to ensure nothing is cut off
        margin=dict(l=50, r=50, t=50, b=50)
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
    st.set_page_config(
        page_title="Feature Cooccurrence Explorer",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better styling
    st.markdown("""
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
    """, unsafe_allow_html=True)

    # Main title
    st.markdown('<p class="title-text">Feature Cooccurrence Explorer</p>', unsafe_allow_html=True)

    git_root = get_git_root()
    model_to_batch_size = {
        "gpt2-small": 100,
        "gemma-2-2b": 10,
    }

    with st.sidebar:
        st.markdown('<p class="subtitle-text">Configuration Settings</p>', unsafe_allow_html=True)
        
        st.markdown('<p class="section-text">Model Selection</p>', unsafe_allow_html=True)
        model = st.selectbox(
            "Choose Model",
            ["gpt2-small", "gemma-2-2b"],
            format_func=lambda x: f"{x} (batch size: {model_to_batch_size[x]})",
        )

        model_to_releases = {
            "gpt2-small": ["res-jb", "res-jb-feature-splitting"],
            "gemma-2-2b": ["gemma-scope-2b-pt-res-canonical"],
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
        }

        n_batches_reconstruction = model_to_batch_size[model]

        st.markdown('<p class="section-text">Feature Configuration</p>', unsafe_allow_html=True)
        available_sae_releases = model_to_releases[model]
        sae_release = st.selectbox("SAE Release", available_sae_releases)
        
        available_sae_ids = sae_release_to_ids[sae_release]
        sae_id = st.selectbox(
            "SAE ID", 
            [neat_sae_id(id) for id in available_sae_ids]
        )

        results_root = pj(
            git_root,
            f"results/{model}/{sae_release}/{sae_id}",
        )

        st.markdown('<p class="section-text">Size Settings</p>', unsafe_allow_html=True)
        available_sizes = get_available_sizes(
            results_root, sae_id, n_batches_reconstruction
        )
        selected_size = st.selectbox(
            "Subgraph Size",
            options=available_sizes,
            format_func=lambda x: f"Size {x}",
            key="size_selector",
        )

    # Update pca_results_path
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
        label = f"Subgraph {sg_id} - Top tokens: {', '.join(top_3_tokens)} | Example: {example_context}"
        subgraph_options.append({"label": label, "value": sg_id})

    # st.markdown('<p class="section-text">Subgraph Selection</p>', unsafe_allow_html=True)
    selected_subgraph = st.selectbox(
        "Choose a subgraph to visualize",
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

    # Main visualization area
    st.markdown('<p class="subtitle-text">Visualization</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1.5])  # Changed ratio from [3, 2]

    with col1:
        st.markdown('<p class="section-text">PCA Visualization</p>', unsafe_allow_html=True)
        st.markdown("""
            The plot below shows the PCA projection of feature activations. 
            Colors represent different features. Click on any point to see detailed activations.
        """)
        pca_plot, color_map = plot_pca_2d(
            pca_df=pca_df,
            max_feature_info=results["all_max_feature_info"],
            fs_splitting_nodes=fs_splitting_nodes,
        )

        selected_points = spe.plotly_events(
            pca_plot,
            click_event=True,
            override_height=500,
            key=f"pca_plot_{selected_subgraph}",
        )

    with col2:
        st.markdown('<p class="section-text">Feature Activation Analysis</p>', unsafe_allow_html=True)

        if not selected_points:
            st.info("👆 Click on any point in the PCA plot to see its feature activations.")
            feature_plot = plot_feature_activations(
                results["all_graph_feature_acts"],
                point_index=None,
                fs_splitting_nodes=fs_splitting_nodes,
            )
            st.plotly_chart(feature_plot, use_container_width=True)
        else:
            selected_x = selected_points[0]["x"]
            selected_y = selected_points[0]["y"]
            matching_points = pca_df[
                (pca_df["PC2"] == selected_x) & (pca_df["PC3"] == selected_y)
            ]

            if not matching_points.empty:
                point_index = matching_points.index[0]

                st.markdown('<p class="section-text">Selected Point Details</p>', unsafe_allow_html=True)
                
                # Create an expander for point details
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
                st.error("The selected point is not in the current dataset. Please select a different point.")

if __name__ == "__main__":
    main()