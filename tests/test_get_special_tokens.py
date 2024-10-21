import pytest
from transformer_lens import HookedTransformer

from sae_cooccurrence.normalised_cooc_functions import get_special_tokens


class MockTokenizer:
    def __init__(self, bos_token_id, eos_token_id, pad_token_id):
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id


class MockModel(HookedTransformer):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


def test_get_special_tokens_normal():
    mock_tokenizer = MockTokenizer(1, 2, 3)
    mock_model = MockModel(mock_tokenizer)
    result = get_special_tokens(mock_model)
    assert result == {1, 2, 3}


def test_get_special_tokens_some_none():
    mock_tokenizer = MockTokenizer(1, None, 3)
    mock_model = MockModel(mock_tokenizer)
    result = get_special_tokens(mock_model)
    assert result == {1, None, 3}


def test_get_special_tokens_all_none():
    mock_tokenizer = MockTokenizer(None, None, None)
    mock_model = MockModel(mock_tokenizer)
    result = get_special_tokens(mock_model)
    assert result == {None}


def test_get_special_tokens_duplicate_ids():
    mock_tokenizer = MockTokenizer(1, 1, 1)
    mock_model = MockModel(mock_tokenizer)
    result = get_special_tokens(mock_model)
    assert result == {1}


def test_get_special_tokens_no_tokenizer():
    mock_model = MockModel(None)
    with pytest.raises(ValueError, match="Model tokenizer is None"):
        get_special_tokens(mock_model)


@pytest.mark.parametrize(
    "bos,eos,pad,expected",
    [
        (0, 1, 2, {0, 1, 2}),
        (-1, 0, 1, {-1, 0, 1}),
        (100, 101, 102, {100, 101, 102}),
    ],
)
def test_get_special_tokens_various_ids(bos, eos, pad, expected):
    mock_tokenizer = MockTokenizer(bos, eos, pad)
    mock_model = MockModel(mock_tokenizer)
    result = get_special_tokens(mock_model)
    assert result == expected


def test_get_special_tokens_real_model():
    model = HookedTransformer.from_pretrained("tiny-stories-1M")
    result = get_special_tokens(model)
    assert isinstance(result, set)
    assert all(isinstance(token, (int, type(None))) for token in result)
