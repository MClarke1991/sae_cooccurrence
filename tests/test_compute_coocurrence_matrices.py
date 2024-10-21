import json
from typing import NamedTuple

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

    class MockSAEConfig(NamedTuple):
        d_sae: int
        neuronpedia_id: str

    class MockSAE:
        def __init__(self):
            self.cfg = MockSAEConfig(d_sae=10, neuronpedia_id="mock_sae_id")
            self.threshold = torch.zeros(self.cfg.d_sae)

        def encode(self, x):
            return torch.rand(x.shape[0], self.cfg.d_sae)  # type: ignore

    class MockActivationStore:
        def __init__(self):
            self.train_batch_size_tokens = 100
            self.normalize_activations = None

        def next_batch(self):
            return torch.rand(100, 1, 50)  # 100 tokens, 50 features

        def get_batch_tokens(self):
            return torch.randint(0, 1000, (8, 13))  # 8 prompts, 13 tokens each

        def get_activations(self, batch_tokens):
            n_batches, n_context = batch_tokens.shape
            return torch.rand(
                n_batches, n_context, 1, 50
            )  # n_batches prompts, n_context tokens, 1, 50 features

    sae = MockSAE()
    activation_store = MockActivationStore()
    sae_id = "mock_sae_id"
    n_batches = 5
    activation_thresholds = [0.1, 0.5]
    device = "cpu"
    special_tokens = {0, 1, 2}  # Mock special tokens

    return (
        sae,
        activation_store,
        sae_id,
        n_batches,
        activation_thresholds,
        device,
        special_tokens,
    )


@pytest.mark.parametrize("remove_special_tokens", [False, True])
def test_compute_cooccurrence_matrices(
    mock_environment, snapshot, remove_special_tokens
):
    (
        sae,
        activation_store,
        sae_id,
        n_batches,
        activation_thresholds,
        device,
        special_tokens,
    ) = mock_environment

    # Compute actual results
    actual_results = compute_cooccurrence_matrices(
        sae,
        activation_store,
        n_batches,
        activation_thresholds,
        device,
        remove_special_tokens=remove_special_tokens,
        special_tokens=special_tokens,
    )

    # Serialize results
    serialized_results = {}
    for threshold, matrix in actual_results.items():
        serialized_results[str(threshold)] = numpy_serializer(matrix.cpu().numpy())

    # Compare results with snapshot
    for threshold, serialized_matrix in serialized_results.items():
        snapshot.assert_match(
            serialized_matrix,
            f"cooccurrence_matrix_{threshold}_remove_special_{remove_special_tokens}",
        )

    print(
        f"Snapshot test completed successfully for remove_special_tokens={remove_special_tokens}!"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--snapshot-update"])
