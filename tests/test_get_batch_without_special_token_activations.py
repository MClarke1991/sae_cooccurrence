from unittest.mock import Mock, patch

import pytest
import torch

# Assuming the function is in a module named 'your_module'
from sae_cooccurrence.normalised_cooc_functions import (
    get_batch_without_special_token_activations,
)

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


@pytest.fixture
def special_tokens():
    return {0, 1, 2}  # Example special tokens


@pytest.mark.parametrize("device", available_devices)
def test_get_batch_without_special_tokens_shape(
    mock_activations_store, special_tokens, device
):
    mock_activations_store.get_batch_tokens.return_value = torch.randint(
        0, 1000, (10, 20)
    ).to(device)
    mock_activations_store.get_activations.return_value = torch.randn(10, 20, 50).to(
        device
    )
    mock_activations_store.apply_norm_scaling_factor.return_value = torch.randn(
        190, 1, 50
    ).to(device)

    result = get_batch_without_special_token_activations(
        mock_activations_store, special_tokens, device
    )

    assert result.shape == (100, 1, 50)
    assert result.device.type == device.split(":")[0]


@pytest.mark.parametrize("device", available_devices)
def test_get_batch_without_special_tokens_removes_special(
    mock_activations_store, special_tokens, device
):
    # Create a batch of tokens where the first token of each sequence is a special token
    tokens = torch.tensor([[0, 3, 4, 5], [1, 6, 7, 8], [2, 9, 10, 11]]).to(device)

    # Create corresponding activations
    activations = torch.stack(
        [
            torch.cat([torch.ones(1, 50), torch.zeros(3, 50)]),
            torch.cat([torch.ones(1, 50), torch.zeros(3, 50)]),
            torch.cat([torch.ones(1, 50), torch.zeros(3, 50)]),
        ]
    ).to(device)

    mock_activations_store.get_batch_tokens.return_value = tokens
    mock_activations_store.get_activations.return_value = activations
    mock_activations_store.apply_norm_scaling_factor.return_value = torch.zeros(
        9, 1, 50
    ).to(device)

    result = get_batch_without_special_token_activations(
        mock_activations_store, special_tokens, device
    )

    # Check that special tokens (first token of each sequence) are removed
    assert torch.all(result == 0)
    assert result.shape == (9, 1, 50)  # 3 sequences * 3 non-special tokens each
    assert result.device.type == device.split(":")[0]


@pytest.mark.parametrize("device", available_devices)
def test_get_batch_without_special_tokens_normalization(
    mock_activations_store, special_tokens, device
):
    mock_activations_store.get_batch_tokens.return_value = torch.randint(
        3, 1000, (10, 20)
    ).to(device)
    mock_activations_store.get_activations.return_value = torch.ones(10, 20, 50).to(
        device
    )
    mock_activations_store.apply_norm_scaling_factor.return_value = (
        torch.ones(200, 1, 50).to(device) * 2
    )

    result = get_batch_without_special_token_activations(
        mock_activations_store, special_tokens, device
    )

    assert torch.all(result == 2)
    assert result.device.type == device.split(":")[0]
    mock_activations_store.apply_norm_scaling_factor.assert_called_once()


@pytest.mark.parametrize("device", available_devices)
def test_get_batch_without_special_tokens_no_normalization(
    mock_activations_store, special_tokens, device
):
    mock_activations_store.normalize_activations = None
    mock_activations_store.get_batch_tokens.return_value = torch.randint(
        3, 1000, (10, 20)
    ).to(device)
    mock_activations_store.get_activations.return_value = torch.ones(10, 20, 50).to(
        device
    )

    result = get_batch_without_special_token_activations(
        mock_activations_store, special_tokens, device
    )

    assert torch.all(result == 1)
    assert result.device.type == device.split(":")[0]
    mock_activations_store.apply_norm_scaling_factor.assert_not_called()


@pytest.mark.parametrize("device", available_devices)
def test_get_batch_without_special_tokens_correct_batch_size(
    mock_activations_store, special_tokens, device
):
    mock_activations_store.train_batch_size_tokens = 50
    mock_activations_store.get_batch_tokens.return_value = torch.randint(
        3, 1000, (10, 20)
    ).to(device)
    mock_activations_store.get_activations.return_value = torch.randn(10, 20, 50).to(
        device
    )
    mock_activations_store.apply_norm_scaling_factor.return_value = torch.randn(
        200, 1, 50
    ).to(device)

    result = get_batch_without_special_token_activations(
        mock_activations_store, special_tokens, device
    )

    assert result.shape[0] == 50
    assert result.device.type == device.split(":")[0]


@pytest.mark.parametrize("device", available_devices)
def test_get_batch_without_special_token_activations_shape(
    mock_activations_store, special_tokens, device
):
    mock_activations_store.get_batch_tokens.return_value = torch.randint(
        0, 1000, (10, 20)
    ).to(device)
    mock_activations_store.get_activations.return_value = torch.randn(10, 20, 50).to(
        device
    )
    mock_activations_store.apply_norm_scaling_factor.return_value = torch.randn(
        190, 1, 50
    ).to(device)

    result = get_batch_without_special_token_activations(
        mock_activations_store, special_tokens, device
    )

    assert result.shape == (100, 1, 50)
    assert result.device.type == device.split(":")[0]


# @pytest.mark.parametrize("device", available_devices)
# def test_get_batch_without_special_tokens_removes_special(
#     mock_activations_store, special_tokens, device
# ):
#     # Create a batch of tokens where the first token of each sequence is a special token
#     tokens = torch.tensor([[0, 3, 4, 5], [1, 6, 7, 8], [2, 9, 10, 11]]).to(device)

#     # Create corresponding activations
#     activations = torch.stack(
#         [
#             torch.cat([torch.ones(1, 50), torch.zeros(3, 50)]),
#             torch.cat([torch.ones(1, 50), torch.zeros(3, 50)]),
#             torch.cat([torch.ones(1, 50), torch.zeros(3, 50)]),
#         ]
#     ).to(device)

#     mock_activations_store.get_batch_tokens.return_value = tokens
#     mock_activations_store.get_activations.return_value = activations
#     mock_activations_store.apply_norm_scaling_factor.return_value = torch.zeros(
#         9, 1, 50
#     ).to(device)

#     result = get_batch_without_special_token_activations(
#         mock_activations_store, special_tokens, device
#     )

#     # Check that special tokens (first token of each sequence) are removed
#     assert torch.all(result == 0)
#     assert result.shape == (9, 1, 50)  # 3 sequences * 3 non-special tokens each
#     assert result.device.type == device.split(":")[0]


@pytest.mark.parametrize("device", available_devices)
def test_get_batch_without_special_token_activations_normalization(
    mock_activations_store, special_tokens, device
):
    mock_activations_store.get_batch_tokens.return_value = torch.randint(
        0, 1000, (10, 20)
    ).to(device)
    mock_activations_store.get_activations.return_value = torch.ones(10, 20, 50).to(
        device
    )
    mock_activations_store.apply_norm_scaling_factor.return_value = (
        torch.ones(190, 1, 50).to(device) * 2
    )

    result = get_batch_without_special_token_activations(
        mock_activations_store, special_tokens, device
    )

    assert torch.all(result == 2)
    assert result.device.type == device.split(":")[0]
    mock_activations_store.apply_norm_scaling_factor.assert_called_once()


@pytest.mark.parametrize("device", available_devices)
def test_get_batch_without_special_token_activations_no_normalization(
    mock_activations_store, special_tokens, device
):
    mock_activations_store.normalize_activations = None
    mock_activations_store.get_batch_tokens.return_value = torch.randint(
        0, 1000, (10, 20)
    ).to(device)
    mock_activations_store.get_activations.return_value = torch.ones(10, 20, 50).to(
        device
    )

    result = get_batch_without_special_token_activations(
        mock_activations_store, special_tokens, device
    )

    assert torch.all(result == 1)
    assert result.device.type == device.split(":")[0]
    mock_activations_store.apply_norm_scaling_factor.assert_not_called()


@pytest.mark.parametrize("device", available_devices)
def test_get_batch_without_special_token_activations_correct_batch_size(
    mock_activations_store, special_tokens, device
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

    result = get_batch_without_special_token_activations(
        mock_activations_store, special_tokens, device
    )

    assert result.shape[0] == 50
    assert result.device.type == device.split(":")[0]


@pytest.mark.parametrize("device", available_devices)
def test_get_batch_without_special_token_activations_no_shuffle(
    mock_activations_store, special_tokens, device
):
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

        get_batch_without_special_token_activations(
            mock_activations_store, special_tokens, device
        )

        mock_randperm.assert_not_called()
