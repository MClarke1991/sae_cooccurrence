import numpy as np

from sae_cooccurrence.graph_generation import remove_low_weight_edges


def test_remove_low_weight_edges_basic():
    # Test basic thresholding
    matrix = np.array([[0.1, 0.5, 0.3], [0.2, 0.4, 0.6], [0.8, 0.1, 0.2]])
    threshold = 0.3
    expected = np.array([[0.0, 0.5, 0.3], [0.0, 0.4, 0.6], [0.8, 0.0, 0.0]])

    result = remove_low_weight_edges(matrix, threshold)
    np.testing.assert_array_equal(result, expected)


def test_remove_low_weight_edges_all_below():
    # Test when all values are below threshold
    matrix = np.array([[0.1, 0.2], [0.2, 0.1]])
    threshold = 0.5
    expected = np.zeros((2, 2))

    result = remove_low_weight_edges(matrix, threshold)
    np.testing.assert_array_equal(result, expected)


def test_remove_low_weight_edges_all_above():
    # Test when all values are above threshold
    matrix = np.array([[0.6, 0.7], [0.8, 0.9]])
    threshold = 0.5

    result = remove_low_weight_edges(matrix, threshold)
    np.testing.assert_array_equal(result, matrix)


def test_remove_low_weight_edges_exact_threshold():
    # Test values exactly at threshold
    matrix = np.array([[0.5, 0.3], [0.7, 0.5]])
    threshold = 0.5
    expected = np.array([[0.5, 0.0], [0.7, 0.5]])

    result = remove_low_weight_edges(matrix, threshold)
    np.testing.assert_array_equal(result, expected)


def test_remove_low_weight_edges_preserves_input():
    # Test that original matrix is not modified
    matrix = np.array([[0.1, 0.5], [0.2, 0.4]])
    original = matrix.copy()
    threshold = 0.3

    _ = remove_low_weight_edges(matrix, threshold)
    np.testing.assert_array_equal(matrix, original)
