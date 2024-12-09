from unittest.mock import patch

import pandas as pd
import pytest

from sae_cooccurrence.pca import get_top_tokens_and_context

# Mock plotly.io.kaleido before importing the module
with patch("plotly.io.kaleido.scope", create=True):
    from sae_cooccurrence.pca import get_top_tokens_and_context


def test_get_top_tokens_and_context():
    # Test case 1: Normal case with multiple tokens
    all_fired_tokens = ["token1", "token2", "token1", "token3", "token1", "token2"]
    all_token_dfs = pd.DataFrame(
        {
            "str_tokens": ["token1", "token2", "token3"],
            "context": ["context1", "context2", "context3"],
        }
    )

    top_3_tokens, example_context = get_top_tokens_and_context(
        all_fired_tokens, all_token_dfs
    )

    assert len(top_3_tokens) == 3
    assert top_3_tokens[0] == ("token1", 3)  # Most common token
    assert top_3_tokens[1] == ("token2", 2)  # Second most common
    assert top_3_tokens[2] == ("token3", 1)  # Third most common
    assert example_context == "context1"  # Context of most common token

    # Test case 2: Empty input
    empty_tokens = []
    empty_df = pd.DataFrame({"str_tokens": [], "context": []})

    top_3_tokens, example_context = get_top_tokens_and_context(empty_tokens, empty_df)

    assert len(top_3_tokens) == 0
    assert example_context == ""

    # Test case 3: Single token
    single_token = ["token1"]
    single_df = pd.DataFrame({"str_tokens": ["token1"], "context": ["single_context"]})

    top_3_tokens, example_context = get_top_tokens_and_context(single_token, single_df)

    assert len(top_3_tokens) == 1
    assert top_3_tokens[0] == ("token1", 1)
    assert example_context == "single_context"

    # Test case 4: Multiple tokens with same frequency
    tied_tokens = ["token1", "token2", "token1", "token2"]
    tied_df = pd.DataFrame(
        {"str_tokens": ["token1", "token2"], "context": ["context1", "context2"]}
    )

    top_3_tokens, example_context = get_top_tokens_and_context(tied_tokens, tied_df)

    assert len(top_3_tokens) == 2
    assert (
        top_3_tokens[0][1] == top_3_tokens[1][1] == 2
    )  # Both tokens have same frequency
    assert example_context in [
        "context1",
        "context2",
    ]  # Context should be from the first token alphabetically


def test_get_top_tokens_and_context_error_handling():
    # Test case: Token in fired_tokens not present in DataFrame
    all_fired_tokens = ["token1", "token2", "missing_token"]
    all_token_dfs = pd.DataFrame(
        {"str_tokens": ["token1", "token2"], "context": ["context1", "context2"]}
    )

    with pytest.raises(IndexError):
        get_top_tokens_and_context(all_fired_tokens, all_token_dfs)
