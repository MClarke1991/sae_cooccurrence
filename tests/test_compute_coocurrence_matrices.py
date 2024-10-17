import json

import numpy as np
import pytest
import torch

from sae_cooccurrence.normalised_cooc_functions import compute_cooccurrence_matrices


def numpy_serializer(obj):
    if isinstance(obj, np.ndarray):
        return json.dumps(obj.tolist())
    raise TypeError(f"Type {type(obj)} not serializable")


@pytest.fixture
def mock_environment():
    # Set a fixed seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    class MockSAE:
        def __init__(self):
            self.cfg = type(
                "obj", (object,), {"d_sae": 10, "neuronpedia_id": "mock_sae_id"}
            )

        def encode(self, x):
            return torch.rand(x.shape[0], self.cfg.d_sae)  # type: ignore

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


def test_compute_cooccurrence_matrices(mock_environment, snapshot):
    sae, activation_store, sae_id, n_batches, activation_thresholds, device = (
        mock_environment
    )

    # Compute actual results
    actual_results = compute_cooccurrence_matrices(
        sae, activation_store, n_batches, activation_thresholds, device
    )

    # Serialize results
    serialized_results = {}
    for threshold, matrix in actual_results.items():
        serialized_results[str(threshold)] = numpy_serializer(matrix.cpu().numpy())

    # Compare results with snapshot
    for threshold, serialized_matrix in serialized_results.items():
        snapshot.assert_match(serialized_matrix, f"cooccurrence_matrix_{threshold}")

    print("Snapshot test completed successfully!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--snapshot-update"])
