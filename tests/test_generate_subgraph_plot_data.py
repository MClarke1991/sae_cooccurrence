import networkx as nx
import numpy as np
import pandas as pd
import pytest

from sae_cooccurrence.pca import generate_subgraph_plot_data


@pytest.fixture
def sample_data():
    # Create sample thresholded matrix (5x5)
    thresholded_matrix = np.array(
        [
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
        ]
    )

    # Create sample node DataFrame
    node_df = pd.DataFrame(
        {
            "node_id": [0, 1, 2, 3, 4],
            "feature_activations": [0.5, 0.6, 0.7, 0.8, 0.9],
            "subgraph_id": [1, 1, 1, 2, 2],
        }
    )

    return thresholded_matrix, node_df


def test_generate_subgraph_plot_data(sample_data):
    thresholded_matrix, node_df = sample_data

    # Test subgraph 1 (nodes 0, 1, 2)
    subgraph, subgraph_df = generate_subgraph_plot_data(thresholded_matrix, node_df, 1)

    # Check if subgraph is correct type
    assert isinstance(subgraph, nx.Graph)
    assert isinstance(subgraph_df, pd.DataFrame)

    # Check subgraph size
    assert len(list(subgraph.nodes())) == 3
    assert len(list(subgraph.edges())) == 2

    # Check if subgraph_df has correct columns and size
    assert list(subgraph_df.columns) == ["node_id", "feature_activations"]
    assert len(subgraph_df) == 3

    # Check if node IDs match
    assert set(subgraph_df["node_id"]) == {0, 1, 2}


def test_generate_subgraph_plot_data_empty_subgraph(sample_data):
    thresholded_matrix, node_df = sample_data

    # Test with non-existent subgraph ID
    subgraph, subgraph_df = generate_subgraph_plot_data(thresholded_matrix, node_df, 3)

    # Check if results are empty but valid
    assert isinstance(subgraph, nx.Graph)
    assert isinstance(subgraph_df, pd.DataFrame)
    assert len(list(subgraph.nodes())) == 0
    assert len(subgraph_df) == 0


def test_generate_subgraph_plot_data_invalid_input():
    # Test with invalid input types
    with pytest.raises(TypeError):
        generate_subgraph_plot_data(None, pd.DataFrame(), 1)  # type: ignore

    with pytest.raises(ValueError):
        generate_subgraph_plot_data(np.array([]), pd.DataFrame(), 1)
