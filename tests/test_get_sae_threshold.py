import logging
from unittest.mock import Mock

import pytest
import torch

from sae_cooccurence.normalised_cooc_functions import (
    check_if_sae_has_threshold,
    get_sae_threshold,
)


# Mock SAE class
class MockSAE:
    def __init__(self, d_sae, has_threshold=False, threshold_value=None):
        self.cfg = Mock()
        self.cfg.d_sae = d_sae
        if has_threshold:
            self.threshold = threshold_value


# Test fixtures
@pytest.fixture
def mock_sae_with_threshold():
    return MockSAE(d_sae=10, has_threshold=True, threshold_value=torch.ones(10))


@pytest.fixture
def mock_sae_without_threshold():
    return MockSAE(d_sae=10, has_threshold=False)


# Tests
def test_get_sae_threshold_with_threshold(mock_sae_with_threshold):
    device = "cpu"
    result = get_sae_threshold(mock_sae_with_threshold, device)
    assert torch.all(result == torch.ones(10))
    assert result.device.type == device


def test_get_sae_threshold_without_threshold(mock_sae_without_threshold):
    device = "cpu"
    result = get_sae_threshold(mock_sae_without_threshold, device)
    assert torch.all(result == torch.zeros(10))
    assert result.device.type == device


def test_get_sae_threshold_cuda(mock_sae_with_threshold):
    device = "cuda"
    if torch.cuda.is_available():
        result = get_sae_threshold(mock_sae_with_threshold, device)
        assert result.device.type == device
    else:
        pytest.skip("CUDA not available")


def test_check_if_sae_has_threshold(
    mock_sae_with_threshold, mock_sae_without_threshold
):
    assert check_if_sae_has_threshold(mock_sae_with_threshold)
    assert not check_if_sae_has_threshold(mock_sae_without_threshold)


def test_get_sae_threshold_none_sae():
    with pytest.raises(AttributeError):
        get_sae_threshold(None, "cpu")  # type: ignore


def test_get_sae_threshold_logging(mock_sae_with_threshold, caplog):
    caplog.set_level(logging.INFO)
    get_sae_threshold(mock_sae_with_threshold, "cpu")
    assert "Correcting for SAE threshold:" in caplog.text


def test_get_sae_threshold_incorrect_shape():
    incorrect_sae = MockSAE(d_sae=10, has_threshold=True, threshold_value=torch.ones(5))
    result = get_sae_threshold(incorrect_sae, "cpu")  # type: ignore
    assert torch.all(result == torch.zeros(10))
