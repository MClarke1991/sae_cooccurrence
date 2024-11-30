import numpy as np
import pytest
from scipy.sparse import csr_matrix

from sae_cooccurrence.graph_generation import largest_component_size


@pytest.fixture
def sample_matrices():
    # Single component matrix
    single = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ]
    )

    # Two components matrix
    two = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ]
    )

    # Isolated nodes matrix
    isolated = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )

    # Matrix with weighted edges
    weighted = np.array(
        [
            [0, 0.5, 0.8],
            [0.5, 0, 0.3],
            [0.8, 0.3, 0],
        ]
    )

    return {
        "single": csr_matrix(single),
        "two": csr_matrix(two),
        "isolated": csr_matrix(isolated),
        "weighted": csr_matrix(weighted),
    }


def test_single_component(sample_matrices):
    result = largest_component_size(sample_matrices["single"], threshold=0.5)
    assert result == 3


def test_two_components(sample_matrices):
    result = largest_component_size(sample_matrices["two"], threshold=0.5)
    assert result == 2


def test_isolated_nodes(sample_matrices):
    result = largest_component_size(sample_matrices["isolated"], threshold=0.5)
    assert result == 1


def test_threshold_effects(sample_matrices):
    # All edges should be included with threshold 0.2
    result_low = largest_component_size(sample_matrices["weighted"], threshold=0.2)
    assert result_low == 3

    # Only strongest edges (0.8, 0.5) should be included with threshold 0.4
    result_mid = largest_component_size(sample_matrices["weighted"], threshold=0.4)
    assert result_mid == 3

    # Only the strongest edge (0.8) should be included with threshold 0.7
    result_high = largest_component_size(sample_matrices["weighted"], threshold=0.7)
    assert result_high == 2


def test_invalid_inputs():
    # Test with empty matrix
    empty_matrix = csr_matrix((0, 0))
    with pytest.raises(ValueError):
        largest_component_size(empty_matrix, threshold=0.5)

    # Test with negative threshold
    matrix = csr_matrix([[0, 1], [1, 0]])
    result = largest_component_size(matrix, threshold=-1.0)
    assert result == 2  # Should still work, treating negative threshold as 0
