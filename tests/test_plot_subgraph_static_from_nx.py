import tempfile
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from sae_cooccurrence.pca import plot_subgraph_static_from_nx

# Mock plotly.io.kaleido before importing the module
with patch("plotly.io.kaleido.scope", create=True):
    from sae_cooccurrence.pca import plot_subgraph_static_from_nx


@pytest.fixture
def sample_graph():
    """Create a sample graph for testing."""
    G = nx.Graph()
    G.add_edges_from([(0, 1, {"weight": 0.5}), (1, 2, {"weight": 0.8})])
    return G


@pytest.fixture
def sample_subgraph_df():
    """Create a sample subgraph DataFrame."""
    return pd.DataFrame({"node_id": [0, 1, 2], "feature_activations": [0.1, 0.5, 0.3]})


@pytest.fixture
def sample_node_info_df():
    """Create a sample node info DataFrame."""
    return pd.DataFrame(
        {
            "node_id": [0, 1, 2],
            "top_10_tokens": [
                "['token1', 'token2']",
                "['token3', 'token4']",
                "['token5', 'token6']",
            ],
        }
    )


def test_basic_plot(sample_graph, sample_subgraph_df):
    """Test basic plotting functionality."""
    plot_subgraph_static_from_nx(sample_graph, sample_subgraph_df, show_plot=False)
    plt.close()


def test_plot_with_node_info(sample_graph, sample_subgraph_df, sample_node_info_df):
    """Test plotting with node info DataFrame."""
    plot_subgraph_static_from_nx(
        sample_graph,
        sample_subgraph_df,
        node_info_df=sample_node_info_df,
        show_plot=False,
    )
    plt.close()


def test_plot_with_activation_array(sample_graph, sample_subgraph_df):
    """Test plotting with custom activation array."""
    activation_array = np.array([0.2, 0.4, 0.6])
    plot_subgraph_static_from_nx(
        sample_graph,
        sample_subgraph_df,
        activation_array=activation_array,
        show_plot=False,
    )
    plt.close()


def test_save_figures(sample_graph, sample_subgraph_df):
    """Test saving figures to different formats."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_plot"
        plot_subgraph_static_from_nx(
            sample_graph,
            sample_subgraph_df,
            output_path=str(output_path),
            save_figs=True,
            show_plot=False,
        )

        assert (output_path.with_suffix(".png")).exists()
        assert (output_path.with_suffix(".pdf")).exists()
        assert (output_path.with_suffix(".svg")).exists()


def test_plot_with_token_factors(sample_graph, sample_subgraph_df, sample_node_info_df):
    """Test plotting with token factors enabled."""
    plot_subgraph_static_from_nx(
        sample_graph,
        sample_subgraph_df,
        node_info_df=sample_node_info_df,
        plot_token_factors=True,
        show_plot=False,
    )
    plt.close()


# def test_invalid_inputs():
#     """Test handling of invalid inputs."""
#     with pytest.raises(TypeError):
#         plot_subgraph_static_from_nx(None, pd.DataFrame())  # type: ignore

#     with pytest.raises(TypeError):
#         plot_subgraph_static_from_nx(nx.Graph(), None)  # type: ignore

#     with pytest.raises(ValueError):
#         plot_subgraph_static_from_nx(
#             nx.Graph(), pd.DataFrame(columns=["node_id", "feature_activations"])
#         )


def test_invalid_inputs():
    """Test handling of invalid inputs."""
    with pytest.raises(TypeError):
        plot_subgraph_static_from_nx(None, pd.DataFrame())  # type: ignore

    with pytest.raises(TypeError):
        plot_subgraph_static_from_nx(nx.Graph(), None)  # type: ignore

    # Fix: Create DataFrame with proper typing
    empty_df = pd.DataFrame({"node_id": [], "feature_activations": []})
    with pytest.raises(ValueError):
        plot_subgraph_static_from_nx(nx.Graph(), empty_df)


def test_different_node_sizes(sample_graph, sample_subgraph_df):
    """Test plotting with different base node sizes."""
    plot_subgraph_static_from_nx(
        sample_graph, sample_subgraph_df, base_node_size=2000, show_plot=False
    )
    plt.close()


def test_colour_when_inactive(sample_graph, sample_subgraph_df):
    """Test plotting with colour_when_inactive flag."""
    plot_subgraph_static_from_nx(
        sample_graph, sample_subgraph_df, colour_when_inactive=False, show_plot=False
    )
    plt.close()


def test_normalize_globally(sample_graph, sample_subgraph_df):
    """Test plotting with different normalization settings."""
    plot_subgraph_static_from_nx(
        sample_graph, sample_subgraph_df, normalize_globally=False, show_plot=False
    )
    plt.close()
