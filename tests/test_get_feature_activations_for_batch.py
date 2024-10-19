from unittest.mock import Mock, patch

import pytest
import torch

from sae_cooccurrence.normalised_cooc_functions import get_feature_activations_for_batch


@pytest.fixture
def mock_activation_store():
    store = Mock()
    store.next_batch.return_value = torch.randn(100, 1, 50)
    return store


@patch("sae_cooccurrence.normalised_cooc_functions.get_batch_without_first_token")
def test_get_feature_activations_for_batch_with_first_token(
    mock_get_batch_without_first, mock_activation_store
):
    result = get_feature_activations_for_batch(
        mock_activation_store, remove_first_token=False
    )

    mock_activation_store.next_batch.assert_called_once()
    mock_get_batch_without_first.assert_not_called()
    assert torch.is_tensor(result)
    assert result.shape == (100, 1, 50)


@patch("sae_cooccurrence.normalised_cooc_functions.get_batch_without_first_token")
def test_get_feature_activations_for_batch_without_first_token(
    mock_get_batch_without_first, mock_activation_store
):
    mock_get_batch_without_first.return_value = torch.randn(99, 1, 50)

    result = get_feature_activations_for_batch(
        mock_activation_store, remove_first_token=True
    )

    mock_activation_store.next_batch.assert_not_called()
    mock_get_batch_without_first.assert_called_once_with(mock_activation_store)
    assert torch.is_tensor(result)
    assert result.shape == (99, 1, 50)


def test_get_feature_activations_for_batch_returns_tensor(mock_activation_store):
    result = get_feature_activations_for_batch(mock_activation_store)
    assert torch.is_tensor(result)


def test_get_feature_activations_for_batch_preserves_shape(mock_activation_store):
    mock_activation_store.next_batch.return_value = torch.randn(100, 1, 50)
    result = get_feature_activations_for_batch(mock_activation_store)
    assert result.shape == (100, 1, 50)


@pytest.mark.parametrize("remove_first_token", [True, False])
def test_get_feature_activations_for_batch_device_consistency(
    mock_activation_store, remove_first_token
):
    # Determine available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    mock_activation_store.next_batch.return_value = torch.randn(
        100, 1, 50, device=device
    )

    with patch(
        "sae_cooccurrence.normalised_cooc_functions.get_batch_without_first_token"
    ) as mock_get_batch_without_first:
        mock_get_batch_without_first.return_value = torch.randn(
            99, 1, 50, device=device
        )

        result = get_feature_activations_for_batch(
            mock_activation_store, remove_first_token=remove_first_token
        )

        # Check if the device types match
        assert result.device.type == device.type

    # Additional assertion to ensure MPS is used when available
    if torch.backends.mps.is_available():
        assert result.device.type == "mps"
