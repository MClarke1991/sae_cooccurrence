import numpy as np
import pytest
from transformer_lens import HookedTransformer

from sae_cooccurrence.graph_generation import create_decode_tokens_function


@pytest.fixture
def model():
    # Load a small model for testing
    return HookedTransformer.from_pretrained("tiny-stories-1M")


def test_create_decode_tokens_function(model):
    # Create the vectorized function
    decode_tokens = create_decode_tokens_function(model)

    # Test cases
    test_cases = [
        # Single token
        ([42], ["K"]),  # Assuming token 42 maps to "the" in the model's vocabulary
        # None value
        ([None], ["None"]),
        # Multiple tokens
        ([42, 90, None], ["K", "{", "None"]),  # Adjust expected tokens as needed
    ]

    for input_tokens, expected_output in test_cases:
        # Convert input to numpy array
        input_array = np.array(input_tokens)
        result = decode_tokens(input_array)
        assert list(result) == expected_output

    # Test vectorization with 2D array
    input_2d = np.array([[42, None], [100, 42]])
    result_2d = decode_tokens(input_2d)
    assert result_2d.shape == (2, 2)
