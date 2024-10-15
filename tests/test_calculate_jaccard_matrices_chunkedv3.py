import os
import pickle
import warnings

import numpy as np
import pytest
import torch

from sae_cooccurence.normalised_cooc_functions import calculate_jaccard_matrices_chunked


@pytest.fixture
def mock_feature_acts_totals():
    # Set a fixed seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create mock feature_acts_totals
    return {0.1: torch.rand(100, 100), 0.5: torch.rand(100, 100)}


@pytest.fixture
def expected_jaccard_results(mock_feature_acts_totals):
    # Define the path for our expected results file
    expected_results_path = os.path.join(
        os.path.dirname(__file__), "expected_jaccard_matrices.pkl"
    )

    if not os.path.exists(expected_results_path):
        # If the file doesn't exist, compute the results and save them
        results = calculate_jaccard_matrices_chunked(
            mock_feature_acts_totals, device="cpu", chunk_size=10
        )

        # Convert torch tensors to numpy arrays for easier serialization
        numpy_results = {k: v.numpy() for k, v in results.items()}

        with open(expected_results_path, "wb") as f:
            pickle.dump(numpy_results, f)

    # Load and return the expected results
    with open(expected_results_path, "rb") as f:
        numpy_results = pickle.load(f)

    # Convert numpy arrays back to torch tensors
    return {k: torch.from_numpy(v) for k, v in numpy_results.items()}


def test_calculate_jaccard_matrices_chunkedv3(
    mock_feature_acts_totals, expected_jaccard_results
):
    # Suppress warnings for this test
    warnings.simplefilter("ignore")

    # Compute actual results
    actual_results = calculate_jaccard_matrices_chunked(
        mock_feature_acts_totals, device="cpu", chunk_size=10
    )

    # Compare results
    assert set(actual_results.keys()) == set(expected_jaccard_results.keys())
    for threshold in actual_results.keys():
        assert torch.allclose(
            actual_results[threshold],
            expected_jaccard_results[threshold],
            rtol=1e-5,
            atol=1e-8,
        )

    # Restore warnings
    warnings.resetwarnings()
