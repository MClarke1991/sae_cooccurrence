from unittest.mock import Mock, patch

import pytest
import torch

# Assuming the function is in a module named 'your_module'
from sae_cooccurrence.normalised_cooc_functions import get_batch_without_first_token


@pytest.fixture
def mock_activations_store():
    store = Mock()
    store.train_batch_size_tokens = 100
    store.normalize_activations = "expected_average_only_in"
    return store


def test_get_batch_without_first_token_shape(mock_activations_store):
    # Mock the get_batch_tokens and get_activations methods
    mock_activations_store.get_batch_tokens.return_value = torch.randint(
        0, 1000, (10, 20)
    )
    mock_activations_store.get_activations.return_value = torch.randn(10, 20, 50)
    mock_activations_store.apply_norm_scaling_factor.return_value = torch.randn(
        190, 1, 50
    )

    result = get_batch_without_first_token(mock_activations_store)

    assert result.shape == (100, 1, 50)


def test_get_batch_without_first_token_removes_first(mock_activations_store):
    # Create a tensor with a distinct first token
    activations = torch.cat([torch.ones(10, 1, 50), torch.zeros(10, 19, 50)], dim=1)
    mock_activations_store.get_batch_tokens.return_value = torch.randint(
        0, 1000, (10, 20)
    )
    mock_activations_store.get_activations.return_value = activations
    mock_activations_store.apply_norm_scaling_factor.return_value = torch.zeros(
        190, 1, 50
    )

    result = get_batch_without_first_token(mock_activations_store)

    assert torch.all(result == 0)


def test_get_batch_without_first_token_normalization(mock_activations_store):
    mock_activations_store.get_batch_tokens.return_value = torch.randint(
        0, 1000, (10, 20)
    )
    mock_activations_store.get_activations.return_value = torch.ones(10, 20, 50)
    mock_activations_store.apply_norm_scaling_factor.return_value = (
        torch.ones(190, 1, 50) * 2
    )

    result = get_batch_without_first_token(mock_activations_store)

    assert torch.all(result == 2)
    mock_activations_store.apply_norm_scaling_factor.assert_called_once()


def test_get_batch_without_first_token_no_normalization(mock_activations_store):
    mock_activations_store.normalize_activations = None
    mock_activations_store.get_batch_tokens.return_value = torch.randint(
        0, 1000, (10, 20)
    )
    mock_activations_store.get_activations.return_value = torch.ones(10, 20, 50)

    result = get_batch_without_first_token(mock_activations_store)

    assert torch.all(result == 1)
    mock_activations_store.apply_norm_scaling_factor.assert_not_called()


def test_get_batch_without_first_token_correct_batch_size(mock_activations_store):
    mock_activations_store.train_batch_size_tokens = 50
    mock_activations_store.get_batch_tokens.return_value = torch.randint(
        0, 1000, (10, 20)
    )
    mock_activations_store.get_activations.return_value = torch.randn(10, 20, 50)
    mock_activations_store.apply_norm_scaling_factor.return_value = torch.randn(
        190, 1, 50
    )

    result = get_batch_without_first_token(mock_activations_store)

    assert result.shape[0] == 50


@patch("torch.randperm")
def test_get_batch_without_first_token_no_shuffle(
    mock_randperm, mock_activations_store
):
    mock_activations_store.get_batch_tokens.return_value = torch.randint(
        0, 1000, (10, 20)
    )
    mock_activations_store.get_activations.return_value = torch.randn(10, 20, 50)
    mock_activations_store.apply_norm_scaling_factor.return_value = torch.randn(
        190, 1, 50
    )

    get_batch_without_first_token(mock_activations_store)

    mock_randperm.assert_not_called()
