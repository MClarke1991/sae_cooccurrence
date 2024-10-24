from unittest.mock import Mock

import pandas as pd
import pytest
import torch

from sae_cooccurrence.pca import make_token_df


@pytest.fixture
def mock_model():
    model = Mock()
    model.to_str_tokens.side_effect = lambda t: [f"token_{i}" for i in t.tolist()]
    return model


def test_make_token_df_basic(mock_model):
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    df = make_token_df(tokens, mock_model)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 6  # 2 batches * 3 tokens
    assert list(df.columns) == [
        "str_tokens",
        "unique_token",
        "context",
        "batch",
        "pos",
        "label",
    ]


def test_make_token_df_str_tokens(mock_model):
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    df = make_token_df(tokens, mock_model)

    expected_str_tokens = [
        "token_1",
        "token_2",
        "token_3",
        "token_4",
        "token_5",
        "token_6",
    ]
    assert df["str_tokens"].tolist() == expected_str_tokens


def test_make_token_df_unique_token(mock_model):
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    df = make_token_df(tokens, mock_model)

    expected_unique_tokens = [
        "token_1/0",
        "token_2/1",
        "token_3/2",
        "token_4/0",
        "token_5/1",
        "token_6/2",
    ]
    assert df["unique_token"].tolist() == expected_unique_tokens


def test_make_token_df_context(mock_model):
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    df = make_token_df(tokens, mock_model, len_prefix=1, len_suffix=1)

    expected_contexts = [
        "|token_1|token_2",
        "token_1|token_2|token_3",
        "token_2|token_3|",
        "|token_4|token_5",
        "token_4|token_5|token_6",
        "token_5|token_6|",
    ]
    assert df["context"].tolist() == expected_contexts


def test_make_token_df_batch_and_pos(mock_model):
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    df = make_token_df(tokens, mock_model)

    expected_batch = [0, 0, 0, 1, 1, 1]
    expected_pos = [0, 1, 2, 0, 1, 2]
    assert df["batch"].tolist() == expected_batch
    assert df["pos"].tolist() == expected_pos


def test_make_token_df_label(mock_model):
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    df = make_token_df(tokens, mock_model)

    expected_labels = ["0/0", "0/1", "0/2", "1/0", "1/1", "1/2"]
    assert df["label"].tolist() == expected_labels


def test_make_token_df_custom_context_length(mock_model):
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    df = make_token_df(tokens, mock_model, len_prefix=2, len_suffix=2)

    expected_contexts = [
        "|token_1|token_2token_3",
        "token_1|token_2|token_3token_4",
        "token_1token_2|token_3|token_4token_5",
        "token_2token_3|token_4|token_5",
        "token_3token_4|token_5|",
    ]
    assert df["context"].tolist() == expected_contexts


def test_make_token_df_single_token(mock_model):
    tokens = torch.tensor([[1]])
    df = make_token_df(tokens, mock_model)

    assert len(df) == 1
    assert df["str_tokens"].tolist() == ["token_1"]
    assert df["unique_token"].tolist() == ["token_1/0"]
    assert df["context"].tolist() == ["|token_1|"]
    assert df["batch"].tolist() == [0]
    assert df["pos"].tolist() == [0]
    assert df["label"].tolist() == ["0/0"]


def test_make_token_df_empty_input(mock_model):
    tokens = torch.tensor([[]])
    df = make_token_df(tokens, mock_model)

    assert len(df) == 0
    assert list(df.columns) == [
        "str_tokens",
        "unique_token",
        "context",
        "batch",
        "pos",
        "label",
    ]
