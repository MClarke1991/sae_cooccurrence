from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from sae_lens import SAE

from sae_cooccurrence.graph_generation import create_node_info_dataframe


@pytest.fixture
def mock_sae():
    return MagicMock(spec=SAE)


@pytest.fixture
def mock_decode_tokens():
    return np.vectorize(lambda x: f"token_{x}")


@pytest.fixture
def sample_data():
    # Create two small test subgraphs
    g1 = nx.Graph()
    g1.add_edges_from([(0, 1), (1, 2)])
    g2 = nx.Graph()
    g2.add_edges_from([(3, 4)])

    subgraphs = [g1, g2]
    activity_threshold = 0.5
    feature_activations = np.array([0.6, 0.7, 0.8, 0.9, 1.0])
    token_factors_inds = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        ]
    )

    return {
        "subgraphs": subgraphs,
        "activity_threshold": activity_threshold,
        "feature_activations": feature_activations,
        "token_factors_inds": token_factors_inds,
    }


def test_create_node_info_dataframe_basic(mock_sae, mock_decode_tokens, sample_data):
    # Mock the neuronpedia functions using patch
    with patch(
        "sae_cooccurrence.graph_generation.get_neuronpedia_feature_dashboard_no_open",
        return_value="dashboard_link",
    ), patch(
        "sae_cooccurrence.graph_generation.get_neuronpedia_quick_list_no_open",
        return_value="quicklist_link",
    ):
        # Create dataframe
        df = create_node_info_dataframe(
            subgraphs=sample_data["subgraphs"],
            activity_threshold=sample_data["activity_threshold"],
            feature_activations=sample_data["feature_activations"],
            token_factors_inds=sample_data["token_factors_inds"],
            decode_tokens=mock_decode_tokens,
            SAE=mock_sae,
            include_metrics=False,
        )

        # Basic assertions
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5  # Total number of nodes in both subgraphs

        # Check columns
        expected_columns = {
            "node_id",
            "activity_threshold",
            "subgraph_id",
            "subgraph_size",
            "feature_activations",
            "top_10_tokens",
            "neuronpedia_link",
            "quicklist_link",
        }
        assert set(df.columns) == expected_columns

        # Check subgraph assignments
        assert len(df[df["subgraph_id"] == 0]) == 3  # First subgraph has 3 nodes
        assert len(df[df["subgraph_id"] == 1]) == 2  # Second subgraph has 2 nodes

        # Check values
        assert all(df["activity_threshold"] == sample_data["activity_threshold"])
        assert all(df["neuronpedia_link"] == "dashboard_link")
        assert all(df["quicklist_link"] == "quicklist_link")


def test_create_node_info_dataframe_with_metrics(
    mock_sae, mock_decode_tokens, sample_data
):
    with patch(
        "sae_cooccurrence.graph_generation.get_neuronpedia_feature_dashboard_no_open",
        return_value="dashboard_link",
    ), patch(
        "sae_cooccurrence.graph_generation.get_neuronpedia_quick_list_no_open",
        return_value="quicklist_link",
    ):
        # Create dataframe with metrics
        df = create_node_info_dataframe(
            subgraphs=sample_data["subgraphs"],
            activity_threshold=sample_data["activity_threshold"],
            feature_activations=sample_data["feature_activations"],
            token_factors_inds=sample_data["token_factors_inds"],
            decode_tokens=mock_decode_tokens,
            SAE=mock_sae,
            include_metrics=True,
        )

        # Check additional metric columns
        metric_columns = {
            "density",
            "max_avg_degree_ratio",
            "avg_clustering",
            "diameter",
            "single_node_score",
            "hub_spoke_score",
            "strongly_connected_score",
            "linear_score",
        }
        assert all(col in df.columns for col in metric_columns)

        # Check metric values are numeric and not null
        for col in metric_columns:
            series = pd.to_numeric(df[col], errors="coerce")
            assert not series.isna().any()  # Check no NaN values # type: ignore
            # Check series is not empty
            assert len(df[col]) > 0  # pyright: ignore


def test_create_node_info_dataframe_empty(mock_sae, mock_decode_tokens):
    with patch(
        "sae_cooccurrence.graph_generation.get_neuronpedia_feature_dashboard_no_open",
        return_value="dashboard_link",
    ), patch(
        "sae_cooccurrence.graph_generation.get_neuronpedia_quick_list_no_open",
        return_value="quicklist_link",
    ):
        # Test with empty input
        df = create_node_info_dataframe(
            subgraphs=[],
            activity_threshold=0.5,
            feature_activations=np.array([]),
            token_factors_inds=np.array([]).reshape(0, 10),
            decode_tokens=mock_decode_tokens,
            SAE=mock_sae,
            include_metrics=False,
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
