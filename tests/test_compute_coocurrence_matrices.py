import torch
import pytest
import numpy as np
import pickle
import os
from PIBBSS.normalised_cooc_functions import compute_cooccurrence_matrices

@pytest.fixture
def mock_environment():
    # Set a fixed seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    class MockSAE:
        def __init__(self):
            self.cfg = type('obj', (object,), {'d_sae': 10})
        
        def encode(self, x):
            return torch.rand(x.shape[0], self.cfg.d_sae) # type: ignore

    class MockActivationStore:
        def next_batch(self):
            return torch.rand(100, 50)  # 100 tokens, 50 features

    sae = MockSAE()
    activation_store = MockActivationStore()
    sae_id = "mock_sae_id"
    n_batches = 5
    activation_thresholds = [0.1, 0.5]
    device = "cpu"

    return sae, activation_store, sae_id, n_batches, activation_thresholds, device

@pytest.fixture
def expected_results(mock_environment):
    # Define the path for our expected results file
    expected_results_path = os.path.join(os.path.dirname(__file__), "expected_cooccurrence_matrices.pkl")

    if not os.path.exists(expected_results_path):
        # If the file doesn't exist, compute the results and save them
        sae, activation_store, sae_id, n_batches, activation_thresholds, device = mock_environment
        results = compute_cooccurrence_matrices(sae, sae_id, activation_store, n_batches, activation_thresholds, device)
        
        with open(expected_results_path, "wb") as f:
            pickle.dump(results, f)
    
    # Load and return the expected results
    with open(expected_results_path, "rb") as f:
        return pickle.load(f)

def test_compute_cooccurrence_matrices(mock_environment, expected_results):
    sae, activation_store, sae_id, n_batches, activation_thresholds, device = mock_environment
    
    # Compute actual results
    actual_results = compute_cooccurrence_matrices(sae, sae_id, activation_store, n_batches, activation_thresholds, device)

    # Compare results
    assert set(actual_results.keys()) == set(expected_results.keys())
    for threshold in actual_results.keys():
        assert torch.allclose(actual_results[threshold], expected_results[threshold], rtol=1e-5, atol=1e-8)