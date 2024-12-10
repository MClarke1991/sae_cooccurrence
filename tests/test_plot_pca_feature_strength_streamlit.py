from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from plotly.graph_objs import Figure, Scatter

from sae_cooccurrence.pca import plot_pca_feature_strength_streamlit

# Mock plotly.io.kaleido before importing the module
with patch("plotly.io.kaleido.scope", create=True):
    from sae_cooccurrence.pca import plot_pca_feature_strength_streamlit


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create sample PCA DataFrame
    pca_df = pd.DataFrame(
        {
            "PC1": [-1.0, 0.0, 1.0, 2.0],
            "PC2": [-1.0, 0.0, 1.0, 2.0],
            "PC3": [0.5, -0.5, 1.5, -1.5],
            "tokens": ["token1", "token2", "token3", "token4"],
            "context": ["ctx1", "ctx2", "ctx3", "ctx4"],
        }
    )

    # Create sample feature activations
    feature_activations = np.array([0.1, 0.5, 0.8, 0.2])

    return pca_df, feature_activations


def test_plot_pca_feature_strength_single_returns_figure(sample_data):
    """Test that function returns a Plotly Figure object."""
    pca_df, feature_activations = sample_data
    fig = plot_pca_feature_strength_streamlit(
        pca_df, feature_activations, feature_idx=0
    )
    assert isinstance(fig, Figure)


def test_plot_pca_feature_strength_single_trace_properties(sample_data):
    """Test properties of the scatter trace in the figure."""
    pca_df, feature_activations = sample_data
    fig = plot_pca_feature_strength_streamlit(
        pca_df, feature_activations, feature_idx=0, pc_x="PC2", pc_y="PC3"
    )

    # Check there is exactly one trace
    assert len(fig.data) == 1  # type: ignore
    trace = fig.data[0]

    # Check trace type and mode
    assert isinstance(trace, Scatter)
    assert trace.mode == "markers"

    # Check x and y data
    np.testing.assert_array_equal(trace.x, pca_df["PC2"])  # type: ignore
    np.testing.assert_array_equal(trace.y, pca_df["PC3"])  # type: ignore

    # Check marker properties
    assert trace.marker.size == 5  # type: ignore
    assert trace.marker.line.width == 1  # type: ignore
    assert trace.marker.line.color == "DarkSlateGrey"  # type: ignore
    assert trace.marker.cmin == 0.1  # activation_threshold default  # type: ignore
    assert trace.marker.cmax == np.max(feature_activations)  # type: ignore


def test_plot_pca_feature_strength_single_layout_properties(sample_data):
    """Test layout properties of the figure."""
    pca_df, feature_activations = sample_data
    feature_idx = 42
    fig = plot_pca_feature_strength_streamlit(
        pca_df, feature_activations, feature_idx=feature_idx, pc_x="PC2", pc_y="PC3"
    )

    # Check layout properties
    assert fig.layout.title.text == f"SAE Latent {feature_idx} Activation Strength"
    assert fig.layout.xaxis.title.text == "PC2"
    assert fig.layout.yaxis.title.text == "PC3"
    assert fig.layout.hovermode == "closest"
    assert fig.layout.height == 600


def test_plot_pca_feature_strength_single_custom_pc_axes(sample_data):
    """Test custom PC axis labels."""
    pca_df, feature_activations = sample_data
    fig = plot_pca_feature_strength_streamlit(
        pca_df, feature_activations, feature_idx=0, pc_x="PC1", pc_y="PC2"
    )

    assert fig.layout.xaxis.title.text == "PC1"
    assert fig.layout.yaxis.title.text == "PC2"


def test_plot_pca_feature_strength_single_custom_threshold(sample_data):
    """Test custom activation threshold."""
    pca_df, feature_activations = sample_data
    custom_threshold = 0.3
    fig = plot_pca_feature_strength_streamlit(
        pca_df,
        feature_activations,
        feature_idx=0,
        activation_threshold=custom_threshold,
    )

    assert fig.data[0].marker.cmin == custom_threshold  # type: ignore


def test_plot_pca_feature_strength_single_hover_template(sample_data):
    """Test hover template format."""
    pca_df, feature_activations = sample_data
    fig = plot_pca_feature_strength_streamlit(
        pca_df, feature_activations, feature_idx=0
    )

    expected_template = (
        "Token: %{customdata[0]}<br>"
        "Context: %{customdata[1]}<br>"
        "Activation: %{customdata[2]:.3f}<br>"
        "<extra></extra>"
    )
    assert fig.data[0].hovertemplate == expected_template  # type: ignore


def test_plot_pca_feature_strength_single_customdata(sample_data):
    """Test customdata content and shape."""
    pca_df, feature_activations = sample_data
    fig = plot_pca_feature_strength_streamlit(
        pca_df, feature_activations, feature_idx=0
    )

    customdata = fig.data[0].customdata  # type: ignore
    assert customdata.shape == (4, 3)  # 4 points, 3 columns
    np.testing.assert_array_equal(customdata[:, 0], pca_df["tokens"])
    np.testing.assert_array_equal(customdata[:, 1], pca_df["context"])
    np.testing.assert_array_equal(customdata[:, 2], feature_activations)
