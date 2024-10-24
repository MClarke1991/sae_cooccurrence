from dataclasses import dataclass

import pandas as pd
import torch


# Mock classes to simulate the required inputs
class MockActivationStore:
    def __init__(self, batch_tokens):
        self.batch_tokens = batch_tokens

    def get_batch_tokens(self):
        return self.batch_tokens


class MockModel:
    def __init__(self, hook_output):
        self.hook_output = hook_output

    def to_str_tokens(self, tokens):
        return ["token" + str(i) for i in tokens.flatten()]

    def run_with_cache(self, tokens, stop_at_layer=None, names_filter=None):  # noqa: ARG002, C901
        return None, {"mock_hook": self.hook_output}


@dataclass
class MockConfig:
    hook_layer: int = 0
    hook_name: str = "mock_hook"


class MockSAE:
    def __init__(self, n_features, device="cpu"):
        self.cfg = MockConfig()
        self.W_dec = torch.randn(n_features, 10, device=device)

    def encode(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        return torch.randn(batch_size, seq_len, self.W_dec.shape[0])


def test_process_examples():
    # Setup test parameters
    batch_size = 2
    seq_len = 3
    hidden_dim = 10
    n_features = 5
    device = "cpu"

    # Create mock input tensors
    batch_tokens = torch.randint(0, 100, (batch_size, seq_len), device=device)
    hook_output = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # Create mock objects
    activation_store = MockActivationStore(batch_tokens)
    model = MockModel(hook_output)
    sae = MockSAE(n_features, device)
    feature_list = [0, 1, 2]
    n_batches_reconstruction = 1

    # Run the function
    from sae_cooccurrence.pca import process_examples

    results = process_examples(
        activation_store=activation_store,
        model=model,
        sae=sae,
        feature_list=feature_list,
        n_batches_reconstruction=n_batches_reconstruction,
    )

    # Test output types
    assert isinstance(results.all_token_dfs, pd.DataFrame)
    assert isinstance(results.all_fired_tokens, list)
    assert isinstance(results.all_reconstructions, torch.Tensor)
    assert isinstance(results.all_graph_feature_acts, torch.Tensor)
    assert isinstance(results.all_feature_acts, torch.Tensor)
    assert isinstance(results.all_max_feature_info, torch.Tensor)
    assert isinstance(results.all_examples_found, int)

    # Test shapes
    total_tokens = batch_size * seq_len
    assert len(results.all_fired_tokens) <= total_tokens
    assert results.all_feature_acts.shape[1] == n_features
    assert results.all_graph_feature_acts.shape[1] == len(feature_list)

    # Test DataFrame columns
    expected_columns = [
        "str_tokens",
        "unique_token",
        "context",
        "batch",
        "pos",
        "label",
    ]
    assert all(col in results.all_token_dfs.columns for col in expected_columns)

    # Test basic values
    assert results.all_examples_found > 0
    assert not torch.isnan(results.all_feature_acts).any()
    assert not torch.isnan(results.all_graph_feature_acts).any()
    assert not torch.isnan(results.all_reconstructions).any()
