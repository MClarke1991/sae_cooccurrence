from typing import NamedTuple

import pytest
import torch
from PIBBSS.normalised_cooc_functions import check_if_sae_has_threshold


# Mock SAE class
class MockSAEConfig(NamedTuple):
    d_sae: int


class MockSAE:
    def __init__(self, d_sae: int, has_threshold: bool = True):
        self.cfg = MockSAEConfig(d_sae=d_sae)
        if has_threshold:
            self.threshold = torch.rand(d_sae)  # Random tensor of length d_sae


def test_check_if_sae_has_threshold():
    # Test case 1: SAE with correct threshold
    sae_with_correct_threshold = MockSAE(d_sae=10)
    assert check_if_sae_has_threshold(sae_with_correct_threshold)  # type: ignore

    # Test case 2: SAE without threshold
    sae_without_threshold = MockSAE(d_sae=10, has_threshold=False)
    assert not check_if_sae_has_threshold(sae_without_threshold)  # type: ignore

    # Test case 3: SAE with incorrect threshold shape
    sae_with_incorrect_threshold = MockSAE(d_sae=10)
    sae_with_incorrect_threshold.threshold = torch.rand(5)  # Incorrect length
    assert not check_if_sae_has_threshold(sae_with_incorrect_threshold)  # type: ignore

    # Test case 4: SAE with non-tensor threshold
    sae_with_non_tensor_threshold = MockSAE(d_sae=10)
    sae_with_non_tensor_threshold.threshold = [
        0.5
    ] * 10  # List instead of tensor # type:ignore
    assert not check_if_sae_has_threshold(sae_with_non_tensor_threshold)  # type: ignore

    # Test case 5: Passing None
    with pytest.raises(AttributeError):
        check_if_sae_has_threshold(None)  # type: ignore


# If the function is in a different file, you might need to import it like this:
# from your_module import check_if_sae_has_threshold
