from unittest.mock import Mock, patch

import pytest
import torch

from sae_cooccurrence.normalised_cooc_functions import get_feature_activations_for_batch

# Get all available devices
available_devices = []
if torch.cuda.is_available():
    available_devices.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])
if torch.backends.mps.is_available():
    available_devices.append("mps")
available_devices.append("cpu")


@pytest.fixture
def mock_activation_store():
    store = Mock()
    store.next_batch.return_value = torch.randn(100, 1, 50)
    return store


@pytest.mark.parametrize("device", available_devices)
@patch("sae_cooccurrence.normalised_cooc_functions.get_batch_without_first_token")
def test_get_feature_activations_for_batch_with_first_token(
    mock_get_batch_without_first, mock_activation_store, device
):
    mock_activation_store.next_batch.return_value = (
        mock_activation_store.next_batch.return_value.to(device)
    )
    result = get_feature_activations_for_batch(
        mock_activation_store, device=device, remove_first_token=False
    )

    mock_activation_store.next_batch.assert_called_once()
    mock_get_batch_without_first.assert_not_called()
    assert torch.is_tensor(result)
    assert result.shape == (100, 1, 50)
    assert result.device.type == device.split(":")[0]


@pytest.mark.parametrize("device", available_devices)
@patch("sae_cooccurrence.normalised_cooc_functions.get_batch_without_first_token")
def test_get_feature_activations_for_batch_without_first_token(
    mock_get_batch_without_first, mock_activation_store, device
):
    mock_get_batch_without_first.return_value = torch.randn(99, 1, 50).to(device)

    result = get_feature_activations_for_batch(
        mock_activation_store, device=device, remove_first_token=True
    )

    mock_activation_store.next_batch.assert_not_called()
    mock_get_batch_without_first.assert_called_once_with(mock_activation_store, device)
    assert torch.is_tensor(result)
    assert result.shape == (99, 1, 50)
    assert result.device.type == device.split(":")[0]


@pytest.mark.parametrize("device", available_devices)
def test_get_feature_activations_for_batch_returns_tensor(
    mock_activation_store, device
):
    mock_activation_store.next_batch.return_value = (
        mock_activation_store.next_batch.return_value.to(device)
    )
    result = get_feature_activations_for_batch(mock_activation_store, device=device)
    assert torch.is_tensor(result)
    assert result.device.type == device.split(":")[0]


@pytest.mark.parametrize("device", available_devices)
def test_get_feature_activations_for_batch_preserves_shape(
    mock_activation_store, device
):
    mock_activation_store.next_batch.return_value = torch.randn(100, 1, 50).to(device)
    result = get_feature_activations_for_batch(mock_activation_store, device=device)
    assert result.shape == (100, 1, 50)
    assert result.device.type == device.split(":")[0]


@pytest.mark.parametrize("remove_first_token", [True, False])
@pytest.mark.parametrize("device", available_devices)
def test_get_feature_activations_for_batch_device_consistency(
    mock_activation_store, remove_first_token, device
):
    mock_activation_store.next_batch.return_value = torch.randn(100, 1, 50).to(device)

    with patch(
        "sae_cooccurrence.normalised_cooc_functions.get_batch_without_first_token"
    ) as mock_get_batch_without_first:
        mock_get_batch_without_first.return_value = torch.randn(99, 1, 50).to(device)

        result = get_feature_activations_for_batch(
            mock_activation_store, device=device, remove_first_token=remove_first_token
        )

        # Check if the device types match
        assert result.device.type == device.split(":")[0]

    # Additional assertion to ensure MPS is used when available
    if device == "mps":
        assert result.device.type == "mps"
