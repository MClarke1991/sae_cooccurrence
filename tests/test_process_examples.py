from dataclasses import dataclass

import pandas as pd
import pytest
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
        return (
            torch.randn(batch_size, seq_len, self.W_dec.shape[0]) + 0.1
        )  # lower bound here is so we don't get cases where all tokens do not fire


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_process_examples(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    # Setup test parameters
    batch_size = 2
    seq_len = 6  # Must be even
    hidden_dim = 10
    n_features = 5

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
        device=device,
    )

    # Test output types
    assert isinstance(results.all_token_dfs, pd.DataFrame)
    assert isinstance(results.all_fired_tokens, list)
    assert isinstance(results.all_reconstructions, torch.Tensor)
    assert isinstance(results.all_graph_feature_acts, torch.Tensor)
    assert isinstance(results.all_feature_acts, torch.Tensor)
    assert isinstance(results.all_max_feature_info, torch.Tensor)
    assert isinstance(results.all_examples_found, int)
    # Add new assertions for the new fields
    assert isinstance(results.top_3_tokens, list)
    assert isinstance(results.example_context, str)

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
        device=device,
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
    # Add new assertions for the new fields
    assert isinstance(results_with_special_tokens_removed.top_3_tokens, list)
    assert isinstance(results_with_special_tokens_removed.example_context, str)

    # Test shapes
    assert len(results_with_special_tokens_removed.all_fired_tokens) <= total_tokens
    assert results_with_special_tokens_removed.all_feature_acts.shape[1] == n_features
    assert results_with_special_tokens_removed.all_graph_feature_acts.shape[1] == len(
        feature_list
    )

    # Test DataFrame columns
    expected_columns = [
        "str_tokens",
        "unique_token",
        "context",
        "batch",
        "pos",
        "label",
    ]
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

    # Test special tokens removed
    special_token_strs = model.to_str_tokens(special_tokens)
    assert all(
        token not in results_with_special_tokens_removed.all_fired_tokens
        for token in special_token_strs
    ), "Special tokens should not be present in all_fired_tokens when remove_special_tokens is True"

    assert all(
        token
        not in results_with_special_tokens_removed.all_token_dfs["str_tokens"].values
        for token in special_token_strs
    ), "Special tokens should not be present in all_token_dfs when remove_special_tokens is True"

    # Additional device-specific checks
    assert results.all_reconstructions.device.type == device
    assert results.all_graph_feature_acts.device.type == device
    assert results.all_feature_acts.device.type == device
    assert results.all_max_feature_info.device.type == device


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_process_custom_prompts(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    # Setup test parameters
    seq_len = 2
    hidden_dim = 10
    n_features = 5
    prompt_batch_size = 2

    # Create mock prompts
    test_prompts = ["Hello world", "Testing prompts"]

    # Create mock hook output with correct shape for the prompts
    hook_output = torch.randn(prompt_batch_size, seq_len, hidden_dim, device=device)

    # Create mock model with additional to_tokens method
    class MockModelWithTokens(MockModel):
        def to_tokens(self, prompts, prepend_bos=True):  # noqa: ARG002
            # Return tensor with shape [batch_size, seq_len]
            return torch.randint(3, 6, (len(prompts), seq_len), device=device)

    # Create mock objects
    model = MockModelWithTokens(hook_output)
    sae = MockSAE(n_features, device)
    feature_list = [0, 1, 2]

    # Run the function with remove_special_tokens=False
    from sae_cooccurrence.pca import process_custom_prompts

    results = process_custom_prompts(
        prompts=test_prompts,
        model=model,
        sae=sae,
        feature_list=feature_list,
        device=device,
        batch_size=prompt_batch_size,
    )

    if results is not None:
        # Test output types
        assert isinstance(results.all_token_dfs, pd.DataFrame)
        assert isinstance(results.all_fired_tokens, list)
        assert isinstance(results.all_reconstructions, torch.Tensor)
        assert isinstance(results.all_graph_feature_acts, torch.Tensor)
        assert isinstance(results.all_feature_acts, torch.Tensor)
        assert isinstance(results.all_max_feature_info, torch.Tensor)
        assert isinstance(results.all_examples_found, int)
        assert isinstance(results.top_3_tokens, list)
        assert isinstance(results.example_context, str)

        # Test shapes
        total_tokens = len(test_prompts) * seq_len
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
    else:
        pytest.fail("Results should not be None")

    # Run the function with remove_special_tokens=True
    results_with_special_tokens_removed = process_custom_prompts(
        prompts=test_prompts,
        model=model,
        sae=sae,
        feature_list=feature_list,
        remove_special_tokens=True,
        device=device,
        batch_size=2,
    )

    if results_with_special_tokens_removed is not None:
        # Test output types
        assert isinstance(
            results_with_special_tokens_removed.all_token_dfs, pd.DataFrame
        )
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
        assert isinstance(results_with_special_tokens_removed.top_3_tokens, list)
        assert isinstance(results_with_special_tokens_removed.example_context, str)

        # Test shapes
        total_tokens = len(test_prompts) * seq_len
        assert len(results_with_special_tokens_removed.all_fired_tokens) <= total_tokens
        assert (
            results_with_special_tokens_removed.all_feature_acts.shape[1] == n_features
        )
        assert results_with_special_tokens_removed.all_graph_feature_acts.shape[
            1
        ] == len(feature_list)

        # Test DataFrame columns
        expected_columns = [
            "str_tokens",
            "unique_token",
            "context",
            "batch",
            "pos",
            "label",
        ]
        assert all(
            col in results_with_special_tokens_removed.all_token_dfs.columns
            for col in expected_columns
        )

        # Test basic values
        assert results_with_special_tokens_removed.all_examples_found > 0
        assert not torch.isnan(
            results_with_special_tokens_removed.all_feature_acts
        ).any()
        assert not torch.isnan(
            results_with_special_tokens_removed.all_graph_feature_acts
        ).any()
        assert not torch.isnan(
            results_with_special_tokens_removed.all_reconstructions
        ).any()

        # Test special tokens removed
        special_token_strs = model.to_str_tokens([0, 1, 2])  # Special token IDs
        assert all(
            token not in results_with_special_tokens_removed.all_fired_tokens
            for token in special_token_strs
        ), "Special tokens should not be present in all_fired_tokens when remove_special_tokens is True"

        assert all(
            token
            not in results_with_special_tokens_removed.all_token_dfs[
                "str_tokens"
            ].values
            for token in special_token_strs
        ), "Special tokens should not be present in all_token_dfs when remove_special_tokens is True"

        # Additional device-specific checks
        assert (
            results_with_special_tokens_removed.all_reconstructions.device.type
            == device
        )
        assert (
            results_with_special_tokens_removed.all_graph_feature_acts.device.type
            == device
        )
        assert (
            results_with_special_tokens_removed.all_feature_acts.device.type == device
        )
        assert (
            results_with_special_tokens_removed.all_max_feature_info.device.type
            == device
        )
    else:
        pytest.fail("Results should not be None")
