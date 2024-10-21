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


@pytest.fixture
def mock_special_tokens():
    return {0, 1, 2}  # Example special tokens


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("remove_special_tokens", [True, False])
@patch(
    "sae_cooccurrence.normalised_cooc_functions.get_batch_without_special_token_activations"
)
def test_get_feature_activations_for_batch(
    mock_get_batch_without_special,
    mock_activation_store,
    mock_special_tokens,
    device,
    remove_special_tokens,
):
    mock_activation_store.next_batch.return_value = (
        mock_activation_store.next_batch.return_value.to(device)
    )
    mock_get_batch_without_special.return_value = torch.randn(99, 1, 50).to(device)

    result = get_feature_activations_for_batch(
        mock_activation_store,
        device=device,
        remove_special_tokens_acts=remove_special_tokens,
        special_tokens=mock_special_tokens,
    )

    if remove_special_tokens:
        mock_activation_store.next_batch.assert_not_called()
        mock_get_batch_without_special.assert_called_once_with(
            mock_activation_store, mock_special_tokens, device
        )
        assert result.shape == (99, 1, 50)
    else:
        mock_activation_store.next_batch.assert_called_once()
        mock_get_batch_without_special.assert_not_called()
        assert result.shape == (100, 1, 50)

    assert torch.is_tensor(result)
    assert result.device.type == device.split(":")[0]


@pytest.mark.parametrize("device", available_devices)
def test_get_feature_activations_for_batch_returns_tensor(
    mock_activation_store, mock_special_tokens, device
):
    mock_activation_store.next_batch.return_value = (
        mock_activation_store.next_batch.return_value.to(device)
    )
    result = get_feature_activations_for_batch(
        mock_activation_store,
        device=device,
        remove_special_tokens_acts=False,
        special_tokens=mock_special_tokens,
    )
    assert torch.is_tensor(result)
    assert result.device.type == device.split(":")[0]


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("remove_special_tokens", [True, False])
def test_get_feature_activations_for_batch_device_consistency(
    mock_activation_store, mock_special_tokens, remove_special_tokens, device
):
    mock_activation_store.next_batch.return_value = torch.randn(100, 1, 50).to(device)

    with patch(
        "sae_cooccurrence.normalised_cooc_functions.get_batch_without_special_token_activations"
    ) as mock_get_batch_without_special:
        mock_get_batch_without_special.return_value = torch.randn(99, 1, 50).to(device)

        result = get_feature_activations_for_batch(
            mock_activation_store,
            device=device,
            remove_special_tokens_acts=remove_special_tokens,
            special_tokens=mock_special_tokens,
        )

        # Check if the device types match
        assert result.device.type == device.split(":")[0]

    # Additional assertion to ensure MPS is used when available
    if device == "mps":
        assert result.device.type == "mps"
