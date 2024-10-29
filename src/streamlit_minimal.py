from os.path import join as pj

import h5py
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sae_cooccurrence.utils.set_paths import get_git_root


def decode_if_bytes(data):
    if isinstance(data, bytes):
        return data.decode("utf-8")
    elif isinstance(data, np.ndarray) and data.dtype.char == "S":
        return np.char.decode(data, "utf-8")
    return data

def load_dataset(dataset):
    if dataset.shape == ():
        return dataset[()]
    else:
        return dataset[:]

def load_pca_data(file_path, subgraph_id):
    with h5py.File(file_path, "r") as f:
        group = f[f"subgraph_{subgraph_id}"]
        
        # Load PCA dataframe
        pca_df_group = group["pca_df"]
        pca_df_data = {}
        for column in pca_df_group.keys():
            pca_df_data[column] = decode_if_bytes(load_dataset(pca_df_group[column]))
        pca_df = pd.DataFrame(pca_df_data)
        
        # Load max feature info
        max_feature_info = load_dataset(group["all_max_feature_info"])
        
        return pca_df, max_feature_info

def generate_color_palette(n_colors):
    colors = (
        px.colors.qualitative.Plotly
        + px.colors.qualitative.D3
        + px.colors.qualitative.G10
        + px.colors.qualitative.T10
        + px.colors.qualitative.Alphabet
    )

    if n_colors > len(colors):
        return px.colors.sample_colorscale("Viridis", n_colors)
    else:
        return colors[:n_colors]

def plot_pca_2d(pca_df, max_feature_info, fs_splitting_nodes):
    max_feature_indices = max_feature_info[:, 1].astype(int)
    max_feature_in_graph = max_feature_info[:, 2].astype(bool)

    unique_features = np.unique(fs_splitting_nodes)
    n_unique = len(unique_features)
    color_palette = generate_color_palette(n_unique)
    color_map = dict(zip(unique_features, color_palette))

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
    )

    # Update marker colors and hover template
    fig.update_traces(
        marker=dict(color=colors),
        hovertemplate=(
            "tokens: %{customdata[0]}<br>"
            "context: %{customdata[1]}<br>"
            "max_feature_index: %{customdata[2]}"
            "<extra></extra>"
        ),
        hoverlabel=dict(
            bgcolor=colors,
            font=dict(color='white'), 
            bordercolor='black' 
        )
    )

    fig.update_layout(
        height=600,
        title="PCA Plot (PC2 vs PC3)",
        xaxis_title="PC2",
        yaxis_title="PC3",
        showlegend=False,
    )

    return fig

def main():
    st.set_page_config(page_title="PCA Visualization")
    st.title("PCA Visualization")

    git_root = get_git_root()
    results_root = pj(
        git_root,
        "results/cooc/gpt2-small/res-jb-feature-splitting/blocks_8_hook_resid_pre_24576/",
    )
    pca_results_path = pj(
        results_root, 
        "blocks_8_hook_resid_pre_24576_pca_for_streamlit", 
        "graph_analysis_results_size_51.h5"
    )
    
    subgraph_id = 829   
    activation_threshold = 1.5
    activation_threshold_safe = str(activation_threshold).replace(".", "_")

    try:
        # Load node information for coloring
        node_df = pd.read_csv(
            pj(results_root, f"dataframes/node_info_df_{activation_threshold_safe}.csv")
        )
        fs_splitting_nodes = node_df.query("subgraph_id == @subgraph_id")["node_id"].tolist()

        pca_df, max_feature_info = load_pca_data(pca_results_path, subgraph_id)
        fig = plot_pca_2d(pca_df, max_feature_info, fs_splitting_nodes)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

if __name__ == "__main__":
    main()