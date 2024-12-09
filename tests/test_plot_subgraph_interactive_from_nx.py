from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from pyvis.network import Network

from sae_cooccurrence.pca import plot_subgraph_interactive_from_nx

# Mock plotly.io.kaleido before importing the module
with patch("plotly.io.kaleido.scope", create=True):
    from sae_cooccurrence.pca import plot_subgraph_interactive_from_nx


@pytest.fixture
def sample_graph():
    """Create a sample graph for testing."""
    G = nx.Graph()
    G.add_edge(0, 1, weight=0.5)
    G.add_edge(1, 2, weight=0.3)
    return G


@pytest.fixture
def sample_subgraph_df():
    """Create a sample subgraph DataFrame."""
    return pd.DataFrame({"node_id": [0, 1, 2], "feature_activations": [0.1, 0.2, 0.3]})


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


def test_basic_functionality(sample_graph, sample_subgraph_df):
    """Test basic functionality with minimal inputs."""
    net, html = plot_subgraph_interactive_from_nx(sample_graph, sample_subgraph_df)

    assert isinstance(net, Network)
    assert isinstance(html, str)
    assert len(net.nodes) == 3
    assert len(net.edges) == 2


def test_with_node_info(sample_graph, sample_subgraph_df, sample_node_info_df):
    """Test with node info DataFrame provided."""
    net, html = plot_subgraph_interactive_from_nx(
        sample_graph,
        sample_subgraph_df,
        node_info_df=sample_node_info_df,
        plot_token_factors=True,
    )

    assert isinstance(net, Network)
    assert "token1" in html


def test_with_activation_array(sample_graph, sample_subgraph_df):
    """Test with custom activation array."""
    activation_array = np.array([0.5, 0.6, 0.7])
    net, html = plot_subgraph_interactive_from_nx(
        sample_graph, sample_subgraph_df, activation_array=activation_array
    )

    assert isinstance(net, Network)
    assert all(node["color"] != "#ffffff" for node in net.nodes)


def test_zero_activations(sample_graph, sample_subgraph_df):
    """Test with zero activations."""
    activation_array = np.zeros(3)
    net, html = plot_subgraph_interactive_from_nx(
        sample_graph, sample_subgraph_df, activation_array=activation_array
    )

    assert isinstance(net, Network)
    assert all(node["color"] == "#ffffff" for node in net.nodes)


def test_custom_height(sample_graph, sample_subgraph_df):
    """Test with custom height parameter."""
    net, html = plot_subgraph_interactive_from_nx(
        sample_graph, sample_subgraph_df, height="500px"
    )

    assert "height: 500px" in html


def test_input_validation():
    """Test input validation."""
    with pytest.raises(TypeError):
        plot_subgraph_interactive_from_nx(
            "not_a_graph",  # type: ignore
            pd.DataFrame(),
        )

    with pytest.raises(TypeError):
        plot_subgraph_interactive_from_nx(
            nx.Graph(),
            "not_a_dataframe",  # type: ignore
        )


def test_empty_graph():
    """Test with empty graph."""
    empty_graph = nx.Graph()
    empty_df = pd.DataFrame(columns=["node_id", "feature_activations"])  # type: ignore

    with pytest.raises(ValueError):
        plot_subgraph_interactive_from_nx(empty_graph, empty_df)


def test_single_node_graph():
    """Test with single node graph."""
    G = nx.Graph()
    G.add_node(0)
    df = pd.DataFrame({"node_id": [0], "feature_activations": [0.1]})

    net, html = plot_subgraph_interactive_from_nx(G, df)

    assert isinstance(net, Network)
    assert len(net.nodes) == 1
    assert len(net.edges) == 0


def test_mismatched_data():
    """Test with mismatched graph and DataFrame."""
    G = nx.Graph()
    G.add_node(0)
    df = pd.DataFrame(
        {
            "node_id": [0, 1],  # Extra node in DataFrame
            "feature_activations": [0.1, 0.2],
        }
    )

    with pytest.raises(IndexError):
        plot_subgraph_interactive_from_nx(G, df)
