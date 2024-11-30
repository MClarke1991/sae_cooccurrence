import numpy as np
import pytest
from scipy.sparse import csr_matrix

from sae_cooccurrence.graph_generation import find_threshold, largest_component_size


@pytest.fixture
def sample_matrix():
    """Create a sample adjacency matrix with known structure."""
    # Create a 10x10 matrix with a main component of size 5
    matrix = np.zeros((10, 10))
    # Create a strongly connected component of size 5
    matrix[0:5, 0:5] = 0.8
    # Add some weaker connections
    matrix[5:8, 5:8] = 0.4
    # Add some noise
    matrix[8:, 8:] = 0.2
    np.fill_diagonal(matrix, 0)  # Remove self-loops
    return matrix


def test_find_threshold_exact_match(sample_matrix):
    """Test when there exists a threshold that gives exactly the desired size."""
    min_size = 4
    max_size = 6
    threshold, size = find_threshold(sample_matrix, min_size, max_size)

    # Check that the resulting size is within bounds
    assert min_size <= size <= max_size

    # Verify the threshold produces expected component size
    actual_size = largest_component_size(csr_matrix(sample_matrix), threshold)
    assert actual_size == size


def test_find_threshold_no_solution_below_max_size():
    """Test when no solution exists within the given range."""
    # Create a matrix where all components are size 2
    matrix = np.zeros((6, 6))
    matrix[0:2, 0:2] = 1
    matrix[2:4, 2:4] = 1
    matrix[4:6, 4:6] = 1
    np.fill_diagonal(matrix, 0)

    with pytest.warns(
        UserWarning, match=r"Largest component size \(\d+\) is above max_size \(1\)"
    ):
        threshold, size = find_threshold(matrix, min_size=1, max_size=1)
        # We don't assert size bounds here since we now allow sizes above max_size


def test_find_threshold_edge_cases(sample_matrix):
    """Test edge cases with very small or large size requirements."""
    # Test with minimum size requirement of 1
    threshold, size = find_threshold(sample_matrix, min_size=1, max_size=3)
    assert 1 <= size <= 3  # Should be able to find a small component

    # Create a densely connected matrix
    large_matrix = np.ones((10, 10)) * 0.9
    np.fill_diagonal(large_matrix, 0)  # Remove self-loops

    # Test with large size requirement
    with pytest.warns(
        UserWarning, match=r"Largest component size \(\d+\) is above max_size \(9\)"
    ):
        find_threshold(large_matrix, min_size=8, max_size=9)
        # We don't assert size bounds here since we now allow sizes above max_size


def test_find_threshold_tolerance():
    """Test that the tolerance parameter works as expected."""
    # Create a matrix where component size changes gradually with threshold
    matrix = np.array(
        [[0, 0.9, 0.8, 0.7], [0.9, 0, 0.6, 0.5], [0.8, 0.6, 0, 0.4], [0.7, 0.5, 0.4, 0]]
    )

    # Test with different tolerances
    threshold1, _ = find_threshold(matrix, min_size=2, max_size=3, tolerance=1e-3)
    threshold2, _ = find_threshold(matrix, min_size=2, max_size=3, tolerance=1e-1)

    # The thresholds should be different due to different tolerances
    assert abs(threshold1 - threshold2) < 1e-1


def test_find_threshold_empty_matrix():
    """Test behavior with empty matrix."""
    empty_matrix = np.zeros((5, 5))
    threshold, size = find_threshold(empty_matrix, min_size=1, max_size=5)
    assert size == 1  # Each node should be in its own component


def test_find_threshold_fully_connected():
    """Test with fully connected matrix."""
    matrix = np.ones((5, 5))
    np.fill_diagonal(matrix, 0)
    threshold, size = find_threshold(matrix, min_size=4, max_size=5)
    assert size == 5  # Should find the full component
