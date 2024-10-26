from dataclasses import dataclass

import pandas as pd
import torch

# Note this test can break easily as the shape of the tensors from the mocks is
# hand tuned to be correct for the input batch size and sequence length. So if you
# change the sequnce length you can get errors because the mock SAE or model is
# producing the wrong shape, not because the function has broken.


# Mock classes to simulate the required inputs
class MockActivationStore:
    def __init__(self, batch_tokens):
        self.batch_tokens = batch_tokens

    def get_batch_tokens(self):
        return self.batch_tokens


class MockTokenizer:
    def __init__(self):
        # Define special token IDs
        self.bos_token_id = 0  # Beginning of sequence token ID
        self.eos_token_id = 1  # End of sequence token ID
        self.pad_token_id = 2  # Padding token ID

    def is_special_token(self, token):
        return token in {self.bos_token_id, self.eos_token_id, self.pad_token_id}


class MockModel:
    def __init__(self, hook_output):
        self.hook_output = hook_output
        self.tokenizer = MockTokenizer()  # Add tokenizer attribute

    def to_str_tokens(self, tokens):
        # Convert tokens to string representation
        return [str(token) for token in tokens]

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
    seq_len = 6  # Must be even
    hidden_dim = 10
    n_features = 5
    device = "cpu"

    # Include special tokens in the batch
    special_tokens = [0, 1, 2]  # bos_token_id, eos_token_id, pad_token_id
    non_special_tokens = [3, 4, 5]
    batch_tokens = torch.tensor(
        [
            [special_tokens[i % len(special_tokens)] for i in range(seq_len // 2)]
            + [
                non_special_tokens[i % len(non_special_tokens)]
                for i in range(seq_len // 2, seq_len)
            ]
            for _ in range(batch_size)
        ],
        device=device,
    )
    hook_output = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # Create mock objects
    activation_store = MockActivationStore(batch_tokens)
    model = MockModel(hook_output)
    sae = MockSAE(n_features, device)
    feature_list = [0, 1, 2]
    n_batches_reconstruction = 1

    # Run the function with remove_special_tokens=False
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

    # Run the function with remove_special_tokens=True
    results_with_special_tokens_removed = process_examples(
        activation_store=activation_store,
        model=model,
        sae=sae,
        feature_list=feature_list,
        n_batches_reconstruction=n_batches_reconstruction,
        remove_special_tokens=True,
    )

    # Test output types
    assert isinstance(results_with_special_tokens_removed.all_token_dfs, pd.DataFrame)
    assert isinstance(results_with_special_tokens_removed.all_fired_tokens, list)
    assert isinstance(
        results_with_special_tokens_removed.all_reconstructions, torch.Tensor
    )
    assert isinstance(
        results_with_special_tokens_removed.all_graph_feature_acts, torch.Tensor
    )
    assert isinstance(
        results_with_special_tokens_removed.all_feature_acts, torch.Tensor
    )
    assert isinstance(
        results_with_special_tokens_removed.all_max_feature_info, torch.Tensor
    )
    assert isinstance(results_with_special_tokens_removed.all_examples_found, int)

    # Test shapes
    assert len(results_with_special_tokens_removed.all_fired_tokens) <= total_tokens
    assert results_with_special_tokens_removed.all_feature_acts.shape[1] == n_features
    assert results_with_special_tokens_removed.all_graph_feature_acts.shape[1] == len(
        feature_list
    )

    # Test DataFrame columns
    assert all(
        col in results_with_special_tokens_removed.all_token_dfs.columns
        for col in expected_columns
    )

    # Test basic values
    assert results_with_special_tokens_removed.all_examples_found > 0
    assert not torch.isnan(results_with_special_tokens_removed.all_feature_acts).any()
    assert not torch.isnan(
        results_with_special_tokens_removed.all_graph_feature_acts
    ).any()
    assert not torch.isnan(
        results_with_special_tokens_removed.all_reconstructions
    ).any()
