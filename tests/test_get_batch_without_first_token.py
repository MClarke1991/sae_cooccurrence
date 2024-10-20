from unittest.mock import Mock, patch

import pytest
import torch

# Assuming the function is in a module named 'your_module'
from sae_cooccurrence.normalised_cooc_functions import get_batch_without_first_token

# Get all available devices
available_devices = []
if torch.cuda.is_available():
    available_devices.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])
if torch.backends.mps.is_available():
    available_devices.append("mps")
available_devices.append("cpu")


@pytest.fixture(params=available_devices)
def mock_activations_store(request):
    store = Mock()
    store.train_batch_size_tokens = 100
    store.normalize_activations = "expected_average_only_in"
    store.device = request.param
    return store


@pytest.mark.parametrize("device", available_devices)
def test_get_batch_without_first_token_shape(mock_activations_store, device):
    mock_activations_store.get_batch_tokens.return_value = torch.randint(
        0, 1000, (10, 20)
    ).to(device)
    mock_activations_store.get_activations.return_value = torch.randn(10, 20, 50).to(
        device
    )
    mock_activations_store.apply_norm_scaling_factor.return_value = torch.randn(
        190, 1, 50
    ).to(device)

    result = get_batch_without_first_token(mock_activations_store, device)

    assert result.shape == (100, 1, 50)
    assert result.device.type == device.split(":")[0]


@pytest.mark.parametrize("device", available_devices)
def test_get_batch_without_first_token_removes_first(mock_activations_store, device):
    activations = torch.cat([torch.ones(10, 1, 50), torch.zeros(10, 19, 50)], dim=1).to(
        device
    )
    mock_activations_store.get_batch_tokens.return_value = torch.randint(
        0, 1000, (10, 20)
    ).to(device)
    mock_activations_store.get_activations.return_value = activations
    mock_activations_store.apply_norm_scaling_factor.return_value = torch.zeros(
        190, 1, 50
    ).to(device)

    result = get_batch_without_first_token(mock_activations_store, device)

    assert torch.all(result == 0)
    assert result.device.type == device.split(":")[0]


@pytest.mark.parametrize("device", available_devices)
def test_get_batch_without_first_token_normalization(mock_activations_store, device):
    mock_activations_store.get_batch_tokens.return_value = torch.randint(
        0, 1000, (10, 20)
    ).to(device)
    mock_activations_store.get_activations.return_value = torch.ones(10, 20, 50).to(
        device
    )
    mock_activations_store.apply_norm_scaling_factor.return_value = (
        torch.ones(190, 1, 50).to(device) * 2
    )

    result = get_batch_without_first_token(mock_activations_store, device)

    assert torch.all(result == 2)
    assert result.device.type == device.split(":")[0]
    mock_activations_store.apply_norm_scaling_factor.assert_called_once()


@pytest.mark.parametrize("device", available_devices)
def test_get_batch_without_first_token_no_normalization(mock_activations_store, device):
    mock_activations_store.normalize_activations = None
    mock_activations_store.get_batch_tokens.return_value = torch.randint(
        0, 1000, (10, 20)
    ).to(device)
    mock_activations_store.get_activations.return_value = torch.ones(10, 20, 50).to(
        device
    )

    result = get_batch_without_first_token(mock_activations_store, device)

    assert torch.all(result == 1)
    assert result.device.type == device.split(":")[0]
    mock_activations_store.apply_norm_scaling_factor.assert_not_called()


@pytest.mark.parametrize("device", available_devices)
def test_get_batch_without_first_token_correct_batch_size(
    mock_activations_store, device
):
    mock_activations_store.train_batch_size_tokens = 50
    mock_activations_store.get_batch_tokens.return_value = torch.randint(
        0, 1000, (10, 20)
    ).to(device)
    mock_activations_store.get_activations.return_value = torch.randn(10, 20, 50).to(
        device
    )
    mock_activations_store.apply_norm_scaling_factor.return_value = torch.randn(
        190, 1, 50
    ).to(device)

    result = get_batch_without_first_token(mock_activations_store, device)

    assert result.shape[0] == 50
    assert result.device.type == device.split(":")[0]


@pytest.mark.parametrize("device", available_devices)
def test_get_batch_without_first_token_no_shuffle(mock_activations_store, device):
    with patch("torch.randperm") as mock_randperm:
        mock_activations_store.get_batch_tokens.return_value = torch.randint(
            0, 1000, (10, 20)
        ).to(device)
        mock_activations_store.get_activations.return_value = torch.randn(
            10, 20, 50
        ).to(device)
        mock_activations_store.apply_norm_scaling_factor.return_value = torch.randn(
            190, 1, 50
        ).to(device)

        get_batch_without_first_token(mock_activations_store, device)

        mock_randperm.assert_not_called()
