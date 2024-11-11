import ast
import io
import logging
import os
import pickle
import re
from collections import Counter
from dataclasses import dataclass, field
from os.path import join as pj
from typing import Any, Literal, Mapping

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.colors
import plotly.express as px
import plotly.graph_objects as go
import torch
from PIL import Image
from plotly.subplots import make_subplots
from pyvis.network import Network
from scipy import sparse
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

from sae_cooccurrence.graph_generation import load_subgraph, plot_subgraph_static
from sae_cooccurrence.normalised_cooc_functions import (
    get_special_tokens,
)

# Disable mathjax for plotly
# This is required because the context for Gemma contains a lot of
# LaTeX symbols and no matter how you try to escape the dollar signs they
# WILL cause the formatting to mess up in plotly.
plotly.io.kaleido.scope.mathjax = None


def assign_category(row, fs_splitting_cluster, order_other_subgraphs=False):
    """
    Determines if a node is in or out of the subgraph of interest.

    Args:
        row (dict): A dictionary representing a node with 'subgraph_id' and 'subgraph_size' keys.
        fs_splitting_cluster (str): The ID of the subgraph of interest.
        order_other_subgraphs (bool, optional): If True, distinguishes between other subgraphs and isolated nodes. Defaults to False.

    Returns:
        int: 0 if the node is in the subgraph of interest,
             1 if the node is in another subgraph (when order_other_subgraphs is True),
             2 if the node is not in any subgraph or in a subgraph of size 1.

    This function categorizes nodes based on their subgraph membership:
    - Category 0: Nodes in the subgraph of interest (fs_splitting_cluster)
    - Category 1: Nodes in other subgraphs (if order_other_subgraphs is True)
    - Category 2: Nodes not in any subgraph or in subgraphs of size 1
    """
    if row["subgraph_id"] == fs_splitting_cluster:
        return 0
    elif (
        row["subgraph_id"] != "Not in subgraph"
        and row["subgraph_size"] > 1
        and order_other_subgraphs
    ):
        return 1
    else:
        return 2


def save_data_to_pickle(data: dict[str, Any], file_path: str):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    logging.info(f"Data saved to {file_path}")


def load_data_from_pickle(file_path: str) -> dict[str, Any]:
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    logging.info(f"Data loaded from {file_path}")
    return data


def list_flatten(nested_list):
    """
    Flattens a nested list into a single-level list.

    Args:
        nested_list (list): A list that may contain other lists as elements.

    Returns:
        list: A flattened list containing all elements from the nested structure.

    Example:
        >>> nested = [[1, 2], [3, 4, [5, 6]], 7]
        >>> list_flatten(nested)
        [1, 2, 3, 4, 5, 6, 7]
    """
    return [x for y in nested_list for x in y]


def make_token_df(tokens, model, len_prefix=10, len_suffix=10):
    """
    Create a DataFrame containing token information and context for each token in the input.

    Args:
        tokens (torch.Tensor): Input tensor of token ids.
        model (HookedTransformer): The transformer model used for tokenization.
        len_prefix (int, optional): Length of the prefix context. Defaults to 5.
        len_suffix (int, optional): Length of the suffix context. Defaults to 5.

    Returns:
        pd.DataFrame: A DataFrame with columns:
            - str_tokens: String representation of each token.
            - unique_token: Unique identifier for each token (format: "token/position").
            - context: Context string for each token (format: "prefix|current|suffix").
            - batch: Batch index for each token.
            - pos: Position of each token within its sequence.
            - label: Unique label for each token (format: "batch/position").

    The function processes the input tokens to create a detailed DataFrame, including:
    - Converting tokens to strings
    - Creating unique identifiers for each token
    - Generating context strings for each token
    - Tracking batch and position information
    """
    str_tokens = [model.to_str_tokens(t) for t in tokens]
    unique_token = [
        [f"{s}/{i}" for i, s in enumerate(str_tok)] for str_tok in str_tokens
    ]

    context, batch, pos, label = [], [], [], []
    for b in range(tokens.shape[0]):
        for p in range(tokens.shape[1]):
            prefix = "".join(str_tokens[b][max(0, p - len_prefix) : p])
            suffix = "".join(
                str_tokens[b][p + 1 : min(tokens.shape[1], p + 1 + len_suffix)]
            )
            current = str_tokens[b][p]
            context.append(f"{prefix}|{current}|{suffix}")
            batch.append(b)
            pos.append(p)
            label.append(f"{b}/{p}")

    return pd.DataFrame(
        {
            "str_tokens": list_flatten(str_tokens),
            "unique_token": list_flatten(unique_token),
            "context": context,
            "batch": batch,
            "pos": pos,
            "label": label,
        }
    )


@dataclass
class ProcessedExamples:
    """
    A data class to store processed examples from the model.

    Attributes:
        - all_token_dfs (pd.DataFrame): DataFrame containing token information for all examples.
        - all_fired_tokens (list): List of tokens that fired during processing.
        - all_reconstructions (torch.Tensor): Tensor containing reconstructions of the fired tokens.
        - all_graph_feature_acts (torch.Tensor): Tensor containing graph feature activations.
        - all_feature_acts (torch.Tensor): Tensor containing feature activations.
        - all_max_feature_info (torch.Tensor): Tensor containing maximum feature information
            (i.e. whether the graph features were the maximally active).
        - all_examples_found (int): Total number of examples found.
        - top_3_tokens (list[tuple[str, int]]): List of the top 3 tokens and their indices.
        - example_context (str): The context of the example.
    """

    all_token_dfs: pd.DataFrame
    all_fired_tokens: list
    all_reconstructions: torch.Tensor
    all_graph_feature_acts: torch.Tensor
    all_feature_acts: torch.Tensor
    all_max_feature_info: torch.Tensor
    all_examples_found: int
    top_3_tokens: list[tuple[str, int]] = field(default_factory=list)
    example_context: str = ""


@dataclass
class ReprocessedResults:
    all_graph_feature_acts: torch.Tensor
    all_max_feature_info: torch.Tensor


def reprocess_results_to_processed_examples(existing_results):
    """
    Reprocess existing results to match the ProcessedExamples class format.

    Args:
        existing_results: The existing results to be reprocessed.

    Returns:
        ProcessedExamples: A new instance of ProcessedExamples with reprocessed data.
    """
    all_graph_feature_acts = existing_results["all_graph_feature_acts"]
    all_max_feature_info = existing_results["all_max_feature_info"]

    # Create a new ProcessedExamples object
    return ReprocessedResults(
        all_graph_feature_acts=all_graph_feature_acts,
        all_max_feature_info=all_max_feature_info,
    )


def run_model_with_cache(model, tokens, sae):
    """
    Run the model with caching and return the cached activations for the SAE hook.

    Args:
        model: The transformer model to run.
        tokens: The input tokens to process.
        sae: The Sparse Autoencoder (SAE) object containing configuration.

    Returns:
        torch.Tensor: The cached activations at the SAE hook layer.

    This function runs the model with caching enabled, stopping at the layer
    after the SAE hook layer and filtering for only the SAE hook name. It then
    returns the cached activations for the SAE hook.
    """
    _, cache = model.run_with_cache(
        tokens,
        stop_at_layer=sae.cfg.hook_layer + 1,
        names_filter=[sae.cfg.hook_name],
    )
    return cache[sae.cfg.hook_name]


def get_max_feature_info(feature_acts, fired_mask, feature_list_tensor, device="cpu"):
    """
    Calculate maximum feature information for fired tokens.

    Args:
        feature_acts (torch.Tensor): Feature activations for all tokens.
        fired_mask (torch.Tensor): Boolean mask indicating which tokens fired.
        feature_list_tensor (torch.Tensor): Tensor of feature indices in the graph.

    Returns:
        torch.Tensor: A tensor containing maximum feature information for each fired token.
            The tensor has shape (num_fired_tokens, 3) where each row contains:
            - max_feature_value: The maximum activation value
            - max_feature_index: The index of the maximally activated feature
            - max_feature_in_graph: Binary indicator (1.0 if max feature is in the graph, 0.0 otherwise)

    This function computes the maximum feature activation for each fired token,
    determines if the maximally activated feature is in the graph, and stacks
    this information into a single tensor.
    """
    max_feature_values, max_feature_indices = feature_acts[fired_mask].max(dim=1)
    max_feature_in_graph = torch.tensor(
        [float(idx in feature_list_tensor) for idx in max_feature_indices],
        dtype=torch.float32,
        device=device,
    )
    max_feature_info = torch.stack(
        [max_feature_values, max_feature_indices.float(), max_feature_in_graph],
        dim=1,
    )
    return max_feature_info


# Add new analysis for top tokens
def get_top_tokens_and_context(all_fired_tokens, all_token_dfs):
    """Get the top 3 most common tokens and an example context for the most common token.

    Args:
        all_fired_tokens (list): List of all fired tokens
        all_token_dfs (pd.DataFrame): DataFrame containing token information and contexts

    Returns:
        tuple: (list of top 3 token tuples, example context string)

    Raises:
        IndexError: If a token in all_fired_tokens is not found in all_token_dfs
    """
    # Validate all tokens exist in DataFrame
    token_set = set(all_token_dfs["str_tokens"])
    for token in all_fired_tokens:
        if token not in token_set:
            raise IndexError(f"Token '{token}' not found in DataFrame")

    token_counts = Counter(all_fired_tokens)
    if token_counts:
        top_3_tokens = token_counts.most_common(3)
        most_common_token = top_3_tokens[0][0]
        example_context = all_token_dfs.loc[
            all_token_dfs["str_tokens"] == most_common_token
        ]["context"].iloc[0]
    else:
        top_3_tokens = []
        example_context = ""
    return top_3_tokens, example_context


def process_examples(
    activation_store,
    model,
    sae,
    feature_list,
    n_batches_reconstruction,
    remove_special_tokens=False,
    device="cpu",
    max_examples=5_000_000,  # Add max_examples parameter
    trim_excess=False,
):
    """
    Process examples from the activation store using the given model and SAE, extract the tokens that the
    features of the cluster/subgraph fire on, with context, as well as the reconstruction of the activations
    by the SAE in these instances. Also extract the maximum feature activation for each token.

    Args:
    - activation_store: The store containing activation data.
    - model: The model used for processing.
    - sae: The SAE model instance.
    - feature_list: List of features to be considered i.e. those within a cluster/subgraph.
    - n_batches_reconstruction: Number of batches to process for reconstruction from the activation store.
    - remove_special_tokens: Whether to remove special tokens from the processed examples e.g. BOS, EOS, PAD.
    - max_examples (int, optional): Maximum number of examples to process. If None, process all examples.

    Returns:
    - ProcessedExamples: A data class containing processed examples and related information.
    """
    examples_found = 0
    all_fired_tokens = []
    all_graph_feature_acts = []
    all_max_feature_info = []
    all_feature_acts = []
    all_reconstructions = []
    all_token_dfs = []

    feature_list_tensor = torch.tensor(feature_list, device=sae.W_dec.device)

    pbar = tqdm(range(n_batches_reconstruction), leave=False)
    for _ in pbar:
        # Get a batch of tokens from the activation store
        tokens = activation_store.get_batch_tokens()

        # Create a DataFrame containing token information
        tokens_df = make_token_df(tokens, model)

        # Flatten the tokens tensor for easier processing
        flat_tokens = tokens.flatten()

        # Run the model and get activations
        sae_in = run_model_with_cache(model, tokens, sae)
        # Encode the activations using the SAE
        feature_acts = sae.encode(sae_in).squeeze()
        # Flatten the feature activations to 2D
        feature_acts = feature_acts.flatten(0, 1).to(device)

        # Create a mask for tokens where any feature in the feature_list is activated
        fired_mask = (feature_acts[:, feature_list]).sum(dim=-1) > 0
        fired_mask = fired_mask.to(device)

        if remove_special_tokens:
            special_tokens = get_special_tokens(model)
            # Create a mask for non-special tokens
            # non_special_mask = ~torch.tensor(
            #     [token in special_tokens for token in flat_tokens],
            #     device=fired_mask.device,
            # )
            non_special_mask = ~torch.isin(
                flat_tokens,
                torch.tensor(list(special_tokens), device=device),
            )
            # Combine the fired_mask with the non_special_mask
            fired_mask = fired_mask & non_special_mask

        # Convert the fired tokens to string representations
        fired_tokens = model.to_str_tokens(flat_tokens[fired_mask])

        # Reconstruct the activations for the fired tokens using only the features in the feature_list
        reconstruction = (
            feature_acts[fired_mask][:, feature_list] @ sae.W_dec[feature_list]
        )

        # Get max feature info
        max_feature_info = get_max_feature_info(
            feature_acts, fired_mask, feature_list_tensor, device=device
        )

        # Append the rows of tokens_df where fired_mask is True
        # Convert fired_mask to CPU, get non-zero indices, flatten, and convert to numpy
        # Use these indices to select rows from tokens_df
        all_token_dfs.append(
            tokens_df.iloc[fired_mask.cpu().nonzero().flatten().numpy()]
        )
        # Append feature activations for fired tokens, filtered by feature_list
        all_graph_feature_acts.append(feature_acts[fired_mask][:, feature_list])

        # Append all feature activations for fired tokens
        all_feature_acts.append(feature_acts[fired_mask])

        # Append maximum feature information for fired tokens
        all_max_feature_info.append(max_feature_info)

        # Append the string representations of fired tokens
        all_fired_tokens.append(fired_tokens)

        # Append reconstructions for fired tokens using selected features
        all_reconstructions.append(reconstruction)

        examples_found += len(fired_tokens)
        pbar.set_description(f"Examples found: {examples_found}")

        # Add early termination check
        if max_examples is not None and examples_found >= max_examples:
            if trim_excess:
                # Trim excess examples if we went over the limit
                excess = examples_found - max_examples
                if excess > 0:
                    all_fired_tokens[-1] = all_fired_tokens[-1][:-excess]
                    all_token_dfs[-1] = all_token_dfs[-1][:-excess]
                    all_graph_feature_acts[-1] = all_graph_feature_acts[-1][:-excess]
                    all_feature_acts[-1] = all_feature_acts[-1][:-excess]
                    all_max_feature_info[-1] = all_max_feature_info[-1][:-excess]
                    all_reconstructions[-1] = all_reconstructions[-1][:-excess]
                    examples_found = max_examples
                break
            else:
                break

    print(f"Total examples found: {examples_found}")
    # Flatten the list of lists
    all_token_dfs = pd.concat(all_token_dfs)
    all_fired_tokens = list_flatten(all_fired_tokens)
    all_reconstructions = torch.cat(all_reconstructions)
    all_graph_feature_acts = torch.cat(all_graph_feature_acts)
    all_feature_acts = torch.cat(all_feature_acts)
    all_max_feature_info = torch.cat(all_max_feature_info)

    top_3_tokens, example_context = get_top_tokens_and_context(
        all_fired_tokens, all_token_dfs
    )

    return ProcessedExamples(
        all_token_dfs=all_token_dfs,
        all_fired_tokens=all_fired_tokens,
        all_reconstructions=all_reconstructions,
        all_graph_feature_acts=all_graph_feature_acts,
        all_examples_found=examples_found,
        all_max_feature_info=all_max_feature_info,
        all_feature_acts=all_feature_acts,
        top_3_tokens=top_3_tokens,
        example_context=example_context,
    )


def process_custom_prompts(
    prompts: list[str],
    model,
    sae,
    feature_list,
    remove_special_tokens=False,
    device="cpu",
    batch_size=10,
):
    """
    Process custom prompts using the given model and SAE, extract the tokens that the
    features of the cluster/subgraph fire on, with context.

    Args:
    - prompts: List of strings to process
    - model: The model used for processing
    - sae: The SAE model instance
    - feature_list: List of features to be considered i.e. those within a cluster/subgraph
    - remove_special_tokens: Whether to remove special tokens from the processed examples e.g. BOS, EOS, PAD
    - device: Device to run computations on
    - batch_size: Number of prompts to process at once (default: 10)

    Returns:
    - ProcessedExamples: A data class containing processed examples and related information
    """
    feature_list_tensor = torch.tensor(feature_list, device=sae.W_dec.device)

    # Initialize lists to store results
    all_filtered_tokens_df = []
    all_fired_tokens = []
    all_reconstructions = []
    all_graph_feature_acts = []
    all_feature_acts = []
    all_max_feature_info = []

    # Process prompts in batches
    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing prompts"):
        batch_prompts = prompts[i : i + batch_size]

        # Convert prompts to tokens
        tokens = model.to_tokens(batch_prompts, prepend_bos=True)

        # Create a DataFrame containing token information
        tokens_df = make_token_df(tokens, model)

        # Flatten the tokens tensor for easier processing
        flat_tokens = tokens.flatten()

        # Run the model and get activations
        sae_in = run_model_with_cache(model, tokens, sae)
        # Encode the activations using the SAE
        feature_acts = sae.encode(sae_in).squeeze()
        # Flatten the feature activations to 2D
        feature_acts = feature_acts.flatten(0, 1).to(device)

        # Create a mask for tokens where any feature in the feature_list is activated
        fired_mask = (feature_acts[:, feature_list]).sum(dim=-1) > 0
        fired_mask = fired_mask.to(device)

        if remove_special_tokens:
            special_tokens = get_special_tokens(model)
            non_special_mask = ~torch.isin(
                flat_tokens,
                torch.tensor(list(special_tokens), device=device),
            )
            # Combine the fired_mask with the non_special_mask
            fired_mask = fired_mask & non_special_mask

        # Skip batch if no tokens fired
        if not fired_mask.any():
            continue

        # Convert the fired tokens to string representations
        fired_tokens = model.to_str_tokens(flat_tokens[fired_mask])

        # Reconstruct the activations for the fired tokens
        reconstruction = (
            feature_acts[fired_mask][:, feature_list] @ sae.W_dec[feature_list]
        )

        # Get max feature info
        max_feature_info = get_max_feature_info(
            feature_acts, fired_mask, feature_list_tensor, device=device
        )

        # Get the rows of tokens_df where fired_mask is True
        filtered_tokens_df = tokens_df.iloc[
            fired_mask.cpu().nonzero().flatten().numpy()
        ]

        # Append batch results
        all_filtered_tokens_df.append(filtered_tokens_df)
        all_fired_tokens.extend(fired_tokens)
        all_reconstructions.append(reconstruction)
        all_graph_feature_acts.append(feature_acts[fired_mask][:, feature_list])
        all_feature_acts.append(feature_acts[fired_mask])
        all_max_feature_info.append(max_feature_info)

    # Combine results from all batches
    if not all_filtered_tokens_df:
        print("No tokens fired for any prompts")
        return None

    combined_tokens_df = pd.concat(all_filtered_tokens_df, ignore_index=True)
    combined_reconstructions = torch.cat(all_reconstructions, dim=0)
    combined_graph_feature_acts = torch.cat(all_graph_feature_acts, dim=0)
    combined_feature_acts = torch.cat(all_feature_acts, dim=0)
    combined_max_feature_info = torch.cat(all_max_feature_info, dim=0)

    examples_found = len(all_fired_tokens)
    print(f"Total examples found: {examples_found}")

    top_3_tokens, example_context = get_top_tokens_and_context(
        all_fired_tokens, combined_tokens_df
    )

    return ProcessedExamples(
        all_token_dfs=combined_tokens_df,
        all_fired_tokens=all_fired_tokens,
        all_reconstructions=combined_reconstructions,
        all_graph_feature_acts=combined_graph_feature_acts,
        all_examples_found=examples_found,
        all_max_feature_info=combined_max_feature_info,
        all_feature_acts=combined_feature_acts,
        top_3_tokens=top_3_tokens,
        example_context=example_context,
    )


def process_batch(batch_data):
    """Helper function to process a single batch of data"""
    tokens, model, sae, feature_list, remove_special_tokens, device = batch_data

    # Create a DataFrame containing token information
    tokens_df = make_token_df(tokens, model)

    # Flatten the tokens tensor for easier processing
    flat_tokens = tokens.flatten()

    # Run the model and get activations
    sae_in = run_model_with_cache(model, tokens, sae)
    # Encode the activations using the SAE
    feature_acts = sae.encode(sae_in).squeeze()
    # Flatten the feature activations to 2D
    feature_acts = feature_acts.flatten(0, 1).to(device)

    # Create a mask for tokens where any feature in the feature_list is activated
    fired_mask = (feature_acts[:, feature_list]).sum(dim=-1) > 0
    fired_mask = fired_mask.to(device)

    if remove_special_tokens:
        special_tokens = get_special_tokens(model)
        non_special_mask = ~torch.isin(
            flat_tokens,
            torch.tensor(list(special_tokens), device=device),
        )
        fired_mask = fired_mask & non_special_mask

    # Convert the fired tokens to string representations
    fired_tokens = model.to_str_tokens(flat_tokens[fired_mask])

    # Reconstruct the activations
    reconstruction = feature_acts[fired_mask][:, feature_list] @ sae.W_dec[feature_list]

    # Get max feature info
    feature_list_tensor = torch.tensor(feature_list, device=sae.W_dec.device)
    max_feature_info = get_max_feature_info(
        feature_acts, fired_mask, feature_list_tensor, device=device
    )

    return {
        "tokens_df": tokens_df.iloc[fired_mask.cpu().nonzero().flatten().numpy()],
        "fired_tokens": fired_tokens,
        "reconstruction": reconstruction,
        "graph_feature_acts": feature_acts[fired_mask][:, feature_list],
        "feature_acts": feature_acts[fired_mask],
        "max_feature_info": max_feature_info,
        "n_examples": len(fired_tokens),
    }


def process_examples_parallel(
    activation_store,
    model,
    sae,
    feature_list,
    n_batches_reconstruction,
    remove_special_tokens=False,
    device="cpu",
    max_examples=5_000_000,
    trim_excess=False,
    num_workers=4,  # Number of parallel workers
):
    """Parallelized version of process_examples"""
    from torch.multiprocessing import Pool, set_start_method

    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    examples_found = 0
    all_results = []

    # Create a pool of workers
    with Pool(num_workers) as pool:
        # Prepare batch data
        batch_data = [
            (
                activation_store.get_batch_tokens(),
                model,
                sae,
                feature_list,
                remove_special_tokens,
                device,
            )
            for _ in range(n_batches_reconstruction)
        ]

        # Process batches in parallel
        for result in tqdm(
            pool.imap(process_batch, batch_data),
            total=n_batches_reconstruction,
            desc="Processing batches",
        ):
            all_results.append(result)
            examples_found += result["n_examples"]

            if max_examples is not None and examples_found >= max_examples:
                if trim_excess:
                    # Trim excess examples from the last batch
                    excess = examples_found - max_examples
                    if excess > 0:
                        last_result = all_results[-1]
                        for key in [
                            "tokens_df",
                            "fired_tokens",
                            "reconstruction",
                            "graph_feature_acts",
                            "feature_acts",
                            "max_feature_info",
                        ]:
                            if isinstance(last_result[key], (list, torch.Tensor)):
                                last_result[key] = last_result[key][:-excess]
                        last_result["n_examples"] -= excess
                        examples_found = max_examples
                break

    # Combine results
    all_token_dfs = pd.concat([r["tokens_df"] for r in all_results])
    all_fired_tokens = list_flatten([r["fired_tokens"] for r in all_results])
    all_reconstructions = torch.cat([r["reconstruction"] for r in all_results])
    all_graph_feature_acts = torch.cat([r["graph_feature_acts"] for r in all_results])
    all_feature_acts = torch.cat([r["feature_acts"] for r in all_results])
    all_max_feature_info = torch.cat([r["max_feature_info"] for r in all_results])

    top_3_tokens, example_context = get_top_tokens_and_context(
        all_fired_tokens, all_token_dfs
    )

    return ProcessedExamples(
        all_token_dfs=all_token_dfs,
        all_fired_tokens=all_fired_tokens,
        all_reconstructions=all_reconstructions,
        all_graph_feature_acts=all_graph_feature_acts,
        all_examples_found=examples_found,
        all_max_feature_info=all_max_feature_info,
        all_feature_acts=all_feature_acts,
        top_3_tokens=top_3_tokens,
        example_context=example_context,
    )


def perform_pca_on_results(
    results: ProcessedExamples,
    n_components: int = 3,
    method: Literal["auto", "full", "arpack", "randomized"] = "full",
):
    """
    Perform PCA on the reconstructions from ProcessedExamples and return a DataFrame with the results.

    Args:
    results (ProcessedExamples): The results from process_examples function
    n_components (int): Number of PCA components to compute (default: 3)
    method (Literal["auto", "full", "arpack", "randomized"]): The method to use for PCA (default: "full")
    Returns:
    Tuple[Optional[pd.DataFrame], Optional[PCA]]: DataFrame containing PCA results and PCA object,
                                                 or (None, None) if PCA cannot be performed
    """
    # Get dimensions of the data
    n_samples, n_features = results.all_reconstructions.cpu().numpy().shape
    max_components = min(n_samples, n_features)

    if n_components > max_components:
        import warnings

        warnings.warn(
            f"Cannot perform PCA: requested n_components ({n_components}) is greater than "
            f"max possible components ({max_components}). Returning None.",
            UserWarning,
        )
        return None, None

    # Perform PCA
    pca = PCA(n_components=n_components, svd_solver=method)
    pca_embedding = pca.fit_transform(results.all_reconstructions.cpu().numpy())

    # Create DataFrame with PCA results
    pca_df = pd.DataFrame(
        pca_embedding, columns=pd.Index([f"PC{i+1}" for i in range(n_components)])
    )

    # Add tokens and context information
    pca_df["tokens"] = results.all_fired_tokens
    pca_df["context"] = results.all_token_dfs.context.values
    pca_df["point_id"] = range(len(pca_df))

    return pca_df, pca


def generate_data(
    model,
    sae,
    activation_store,
    fs_splitting_nodes,
    n_batches_reconstruction,
    decoder=False,
    remove_special_tokens=False,
    device="cpu",
    max_examples=5_000_000,
    trim_excess=False,
    custom_prompts=None,
):
    if custom_prompts is None:
        results = process_examples(
            activation_store,
            model,
            sae,
            fs_splitting_nodes,
            n_batches_reconstruction,
            remove_special_tokens,
            device=device,
            max_examples=max_examples,
            trim_excess=trim_excess,
        )
    else:
        results = process_custom_prompts(
            custom_prompts,
            model,
            sae,
            fs_splitting_nodes,
            remove_special_tokens,
            device=device,
        )
    if results is None:
        raise ValueError("No results found")
    pca_df, pca = perform_pca_on_results(results, n_components=3)
    if decoder:
        pca_decoder, pca_decoder_df = calculate_pca_decoder(sae, fs_splitting_nodes)
    else:
        pca_decoder = None
        pca_decoder_df = None

    return {
        "results": results,
        "pca_df": pca_df,
        "pca": pca,
        "pca_decoder": pca_decoder,
        "pca_decoder_df": pca_decoder_df,
    }


def plot_pca_with_top_feature(
    pca_df, results, fs_splitting_nodes, fs_splitting_cluster, pca_path="", save=False
):
    # Create a dictionary to map feature indices to colors
    color_map = {
        str(fs_splitting_nodes[i]): px.colors.qualitative.Dark24[
            i % len(px.colors.qualitative.Dark24)
        ]
        for i in range(len(fs_splitting_nodes))
    }
    color_map["NA"] = "#808080"  # Add gray color for "NA" category

    # Function to get the active feature and its activation for each data point
    def get_active_feature_and_activation(idx):
        activations = results.all_graph_feature_acts[idx]
        activations = activations.cpu().numpy()
        if np.all(activations == 0):
            return "NA", 0
        active_idx = np.argmax(activations)
        return str(fs_splitting_nodes[active_idx]), activations[active_idx]

    # Add columns for the active feature and its activation
    pca_df["active_feature"], pca_df["top_activation"] = zip(
        *[get_active_feature_and_activation(i) for i in range(len(pca_df))]
    )

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("PC1 vs PC2", "PC1 vs PC3", "PC2 vs PC3"),
        shared_yaxes=True,
    )

    # Add traces for each subplot
    pc_combinations = [("PC1", "PC2"), ("PC1", "PC3"), ("PC2", "PC3")]
    for i, (x, y) in enumerate(pc_combinations):
        for feature in pca_df["active_feature"].unique():
            df_feature = pca_df[pca_df["active_feature"] == feature]
            fig.add_trace(
                go.Scatter(
                    x=df_feature[x],
                    y=df_feature[y],
                    mode="markers",
                    marker=dict(color=color_map[feature]),
                    name=feature,
                    text=[
                        f"Token: {t}<br>Context: {c}<br>Active Feature: {f}<br>Top Activation: {a:.4f}"
                        for t, c, f, a in zip(
                            df_feature["tokens"],
                            df_feature["context"],
                            df_feature["active_feature"],
                            df_feature["top_activation"],
                        )
                    ],
                    hoverinfo="text",
                    showlegend=(i == 0),  # Only show legend for the first subplot
                ),
                row=1,
                col=i + 1,
            )

    # Update layout
    fig.update_layout(
        height=600,
        width=1800,
        title_text=f"PCA Analysis - Cluster {fs_splitting_cluster}",
        legend_title_text="Active SAE Feature",
    )

    # Update axes labels
    fig.update_xaxes(title_text="PC1", row=1, col=1)
    fig.update_xaxes(title_text="PC1", row=1, col=2)
    fig.update_xaxes(title_text="PC2", row=1, col=3)
    fig.update_yaxes(title_text="PC2", row=1, col=1)
    fig.update_yaxes(title_text="PC3", row=1, col=2)
    fig.update_yaxes(title_text="PC3", row=1, col=3)

    if save:
        # Save as PNG
        png_path = os.path.join(
            pca_path, f"pca_plot_graph_{fs_splitting_cluster}_top_feature.png"
        )
        fig.write_image(png_path, scale=4.0)

        svg_path = os.path.join(
            pca_path, f"pca_plot_graph_{fs_splitting_cluster}_top_feature.svg"
        )
        fig.write_image(svg_path)

        # Save as HTML
        html_path = os.path.join(
            pca_path, f"pca_plot_graph_{fs_splitting_cluster}_top_feature.html"
        )
        fig.write_html(html_path)
    else:
        fig.show()


def plot_pca_feature_strength(
    pca_df,
    results,
    fs_splitting_nodes,
    fs_splitting_cluster,
    pca_path,
    pc_x="PC1",
    pc_y="PC2",
    activation_threshold=0.1,
    save=False,
):
    """Plot PCA colored by activation strength for all features."""
    # Create subplots
    n_features = len(fs_splitting_nodes)
    n_cols = 3  # You can adjust this for layout
    n_rows = (n_features + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"Feature {fs_splitting_nodes[i]}" for i in range(n_features)],
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
    )

    # Get activations for all features
    all_activations = results.all_graph_feature_acts.cpu().numpy()

    # Prepare the color map
    cmap = plt.cm.get_cmap("viridis")
    n_colors = 256

    # Get the colormap in RGB
    colormap_RGB = cmap(np.arange(cmap.N))

    # Set the color for zero values to be white
    colormap_RGB[0] = (
        1,
        1,
        1,
        1,
    )  # This line sets the first color in the colormap to white

    # Prepare custom color scale (in Plotly format)
    colorscale = [
        [i / (n_colors - 1), mcolors.rgb2hex(colormap_RGB[i])] for i in range(n_colors)
    ]

    for i, feature in enumerate(fs_splitting_nodes):
        row = i // n_cols + 1
        col = i % n_cols + 1

        # Get activations for this feature
        feature_activations = all_activations[:, i]

        # Create scatter trace
        scatter = go.Scatter(
            x=pca_df[pc_x],
            y=pca_df[pc_y],
            mode="markers",
            marker=dict(
                size=5,
                color=feature_activations,
                colorscale=colorscale,
                line=dict(width=1, color="DarkSlateGrey"),
                cmin=activation_threshold,
                cmax=np.max(feature_activations),
            ),
            text=[
                f"Token: {token}<br>Context: {context}<br>Activation: {act:.3f}"
                for token, context, act in zip(
                    pca_df["tokens"], pca_df["context"], feature_activations
                )
            ],
            hoverinfo="text",
            showlegend=False,
        )

        fig.add_trace(scatter, row=row, col=col)

        # Update axes
        # fig.update_xaxes(title_text=pc_x, row=row, col=col)
        # fig.update_yaxes(title_text=pc_y, row=row, col=col)
        fig.update_xaxes(title_text="", row=row, col=col)
        fig.update_yaxes(title_text="", row=row, col=col)

    # Update layout
    fig.update_layout(
        height=300 * n_rows,
        width=300 * n_cols,
        title_text=f"{pc_x} vs {pc_y}, colored by feature activation",
    )

    if save:
        # Save as PNG
        png_path = os.path.join(
            pca_path, f"pca_plot_graph_{fs_splitting_cluster}_fs_{pc_x}_vs_{pc_y}.png"
        )
        fig.write_image(png_path, scale=4.0)

        svg_path = os.path.join(
            pca_path, f"pca_plot_graph_{fs_splitting_cluster}_fs_{pc_x}_vs_{pc_y}.svg"
        )
        fig.write_image(svg_path)

        # Save as HTML
        html_path = os.path.join(
            pca_path, f"pca_plot_graph_{fs_splitting_cluster}_fs_{pc_x}_vs_{pc_y}.html"
        )
        fig.write_html(html_path)

        # print(f"Plots saved as PNG: {png_path}")
        # print(f"Plots saved as HTML: {html_path}")
    else:
        fig.show()
    return None


def plot_pca_feature_strength_streamlit(
    pca_df: pd.DataFrame,
    feature_activations: np.ndarray,
    feature_idx: int,
    pc_x: str = "PC1",
    pc_y: str = "PC2",
    activation_threshold: float = 0.1,
) -> go.Figure:
    """Plot PCA colored by activation strength for a single selected feature.

    Args:
        pca_df: DataFrame with PCA coordinates and metadata
        feature_activations: Array of activation values for the selected feature
        feature_idx: Index of the selected feature
        pc_x: X-axis principal component (default "PC1")
        pc_y: Y-axis principal component (default "PC2")
    """
    # Create custom colormap that uses white for low values
    cmap = plt.cm.get_cmap("viridis")
    colormap_RGB = cmap(np.arange(cmap.N))
    colormap_RGB[0] = (1, 1, 1, 1)  # Set first color to white

    # Convert colormap to Plotly format
    n_colors = 256
    colorscale = [
        [i / (n_colors - 1), mcolors.rgb2hex(colormap_RGB[i])] for i in range(n_colors)
    ]

    fig = go.Figure()

    # Add scatter trace
    fig.add_trace(
        go.Scatter(
            x=pca_df[pc_x],
            y=pca_df[pc_y],
            mode="markers",
            marker=dict(
                size=5,
                color=feature_activations,
                colorscale=colorscale,
                colorbar=dict(title="Activation"),
                line=dict(width=1, color="DarkSlateGrey"),
                cmin=activation_threshold,
                cmax=np.max(feature_activations),
            ),
            hovertemplate=(
                "Token: %{customdata[0]}<br>"
                "Context: %{customdata[1]}<br>"
                "Activation: %{customdata[2]:.3f}<br>"
                "<extra></extra>"
            ),
            customdata=np.column_stack(
                (pca_df["tokens"], pca_df["context"], feature_activations)
            ),
        )
    )

    # Update layout
    fig.update_layout(
        title=f"SAE Latent {feature_idx} Activation Strength",
        xaxis_title=pc_x,
        yaxis_title=pc_y,
        hovermode="closest",
        height=600,
    )

    return fig


def plot_pca_single_feature_strength(
    pca_df,
    results,
    feature_index,
    fs_splitting_cluster,
    pca_path,
    pc_x="PC1",
    pc_y="PC2",
    save=False,
):
    # Create a single plot
    fig = go.Figure()

    # Get activations for all features
    all_activations = results.all_feature_acts.cpu().numpy()

    # Get activations for this feature
    feature_activations = all_activations[:, feature_index]

    # Prepare the color map
    cmap = plt.cm.get_cmap("viridis")
    n_colors = 256

    # Get the colormap in RGB
    colormap_RGB = cmap(np.arange(cmap.N))

    # Set the color for zero values to be white
    colormap_RGB[0] = (
        1,
        1,
        1,
        1,
    )  # This line sets the first color in the colormap to white

    # Prepare custom color scale (in Plotly format)
    colorscale = [
        [i / (n_colors - 1), mcolors.rgb2hex(colormap_RGB[i])] for i in range(n_colors)
    ]

    # Create scatter trace
    scatter = go.Scatter(
        x=pca_df[pc_x],
        y=pca_df[pc_y],
        mode="markers",
        marker=dict(
            size=10,
            color=feature_activations,
            colorscale=colorscale,
            colorbar=dict(title=f"Feature {feature_index} Activation"),
            line=dict(width=1, color="DarkSlateGrey"),
            cmin=0,  # Set to 0 to include 0 activation
            cmax=np.max(feature_activations),
        ),
        text=[
            f"Token: {token}<br>Context: {context}<br>Activation: {act:.3f}"
            for token, context, act in zip(
                pca_df["tokens"], pca_df["context"], feature_activations
            )
        ],
        hoverinfo="text",
    )

    fig.add_trace(scatter)

    # Update layout
    fig.update_layout(
        height=800,
        width=800,
        title_text=f"{pc_x} vs {pc_y}, colored by Feature {feature_index} activation",
        xaxis_title=pc_x,
        yaxis_title=pc_y,
    )

    fig.update_xaxes(constrain="domain")
    fig.update_yaxes(scaleanchor="x")

    if save:
        # Save as PNG
        png_path = os.path.join(
            pca_path,
            f"pca_plot_graph_{fs_splitting_cluster}_feature_{feature_index}_{pc_x}_vs_{pc_y}.png",
        )
        fig.write_image(png_path, scale=4.0)

        # Save as SVG
        svg_path = os.path.join(
            pca_path,
            f"pca_plot_graph_{fs_splitting_cluster}_feature_{feature_index}_{pc_x}_vs_{pc_y}.svg",
        )
        fig.write_image(svg_path)

        # Save as HTML
        html_path = os.path.join(
            pca_path,
            f"pca_plot_graph_{fs_splitting_cluster}_feature_{feature_index}_{pc_x}_vs_{pc_y}.html",
        )
        fig.write_html(html_path)

        print(f"Plots saved as PNG: {png_path}")
        print(f"Plots saved as SVG: {svg_path}")
        print(f"Plots saved as HTML: {html_path}")
    else:
        fig.show()

    return fig


def plot_pca_with_active_features(
    pca_df,
    results,
    fs_splitting_nodes,
    fs_splitting_cluster,
    pca_path,
    n_top_features=2,
    activation_threshold=0.1,
    save=False,
):
    # Create a dictionary to map feature indices to colors
    # color_map = {
    #     i: px.colors.qualitative.Dark24[i % len(px.colors.qualitative.Dark24)]
    #     for i in range(len(fs_splitting_nodes))
    # }

    def get_top_n_features(idx):
        activations = (
            results.all_graph_feature_acts[idx].cpu().numpy()
        )  # Convert to NumPy array
        # Apply threshold
        activations[activations < activation_threshold] = 0
        # If all activations are below threshold, return 'None'
        if np.all(activations == 0):
            return "None"
        top_n_indices = np.argsort(activations)[-n_top_features:]
        top_n_indices = top_n_indices[::-1]  # Reverse to get descending order
        return ", ".join(
            f"{fs_splitting_nodes[i]}" for i in top_n_indices if activations[i] > 0
        )

    # Add a column for the top n active features
    pca_df["top_features"] = [get_top_n_features(i) for i in range(len(pca_df))]

    # Create a color mapping for feature combinations
    unique_combinations = pca_df["top_features"].unique()
    combination_color_map = {
        comb: px.colors.qualitative.Dark24[i % len(px.colors.qualitative.Dark24)]
        for i, comb in enumerate(unique_combinations)
    }

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("PC1 vs PC2", "PC2 vs PC3", "PC1 vs PC3"),
        shared_yaxes=True,
    )

    # Add traces for each subplot
    for i, (x, y) in enumerate([("PC1", "PC2"), ("PC1", "PC3"), ("PC2", "PC3")]):
        for feature, color in combination_color_map.items():
            df_feature = pca_df[pca_df["top_features"] == feature]
            fig.add_trace(
                go.Scatter(
                    x=df_feature[x],
                    y=df_feature[y],
                    mode="markers",
                    marker=dict(color=color),
                    name=feature,
                    text=[
                        f"Tokens: {t}<br>Context: {c}<br>Top Features: {f}"
                        for t, c, f in zip(
                            df_feature["tokens"],
                            df_feature["context"],
                            df_feature["top_features"],
                        )
                    ],
                    hoverinfo="text",
                ),
                row=1,
                col=i + 1,
            )

    # Update layout
    fig.update_layout(
        height=600,
        width=1800,
        showlegend=False,
    )

    # Update axes labels
    fig.update_xaxes(title_text="PC1", row=1, col=1)
    fig.update_xaxes(title_text="PC1", row=1, col=2)
    fig.update_xaxes(title_text="PC2", row=1, col=3)
    fig.update_yaxes(title_text="PC2", row=1, col=1)
    fig.update_yaxes(title_text="PC3", row=1, col=2)
    fig.update_yaxes(title_text="PC3", row=1, col=3)

    if save:
        # Save as PNG
        png_path = os.path.join(
            pca_path,
            f"pca_plot_graph_{fs_splitting_cluster}_top_{n_top_features}_at_{activation_threshold}.png",
        )
        fig.write_image(png_path, scale=4.0)

        svg_path = os.path.join(
            pca_path,
            f"pca_plot_graph_{fs_splitting_cluster}_top_{n_top_features}_at_{activation_threshold}.svg",
        )
        fig.write_image(svg_path)

        # Save as HTML
        html_path = os.path.join(
            pca_path,
            f"pca_plot_graph_{fs_splitting_cluster}_top_{n_top_features}_at_{activation_threshold}.html",
        )
        fig.write_html(html_path)

        # print(f"Plots saved as PNG: {png_path}")
        # print(f"Plots saved as HTML: {html_path}")
    else:
        fig.show()


def get_neighboring_tokens(tokens: pd.DataFrame, center_token: str, window_size=4):
    # Filter the dataframe to only include rows with the center token
    center_rows = tokens[tokens["tokens"] == center_token]

    results = []

    for _, row in center_rows.iterrows():
        context = str(row["context"])
        parts = context.split("|")

        if len(parts) != 3:
            continue  # Skip if the context doesn't have the expected format

        left_context = parts[0].strip().split()
        right_context = parts[2].strip().split()

        # Get neighboring tokens from left context
        left_neighbors = (
            left_context[-window_size:]
            if len(left_context) > window_size
            else left_context
        )

        # Get neighboring tokens from right context
        right_neighbors = (
            right_context[:window_size]
            if len(right_context) > window_size
            else right_context
        )

        # Combine left and right neighbors
        neighbors = left_neighbors + right_neighbors

        results.append(neighbors)

    return results


def plot_token_pca_and_save(
    pca_df, pca_path, fs_splitting_cluster, color_by="token", save=False
):
    """
    Create a single figure with three subplots for PCA results and save as PNG and HTML.

    Args:
    pca_df (pd.DataFrame): DataFrame containing PCA results
    pca_path (str): Path to save the output files
    color_by (str): 'token', 'before', or 'after' to specify coloring scheme

    Returns:
    None
    """
    # Get the Alphabet color scale
    alphabet_colors = plotly.colors.qualitative.Dark24

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2, subplot_titles=("PC1 vs PC2", "PC1 vs PC3", "PC2 vs PC3")
    )

    # Determine color categories based on color_by parameter
    if color_by == "token":
        color_categories = pca_df["tokens"]
    elif color_by == "before" or color_by == "after":
        # Get neighboring tokens
        neighboring_tokens = [
            get_neighboring_tokens(pca_df, token) for token in pca_df["tokens"]
        ]
        if color_by == "before":
            color_categories = [
                neighbors[0][-1] if neighbors and neighbors[0] else "Unknown"
                for neighbors in neighboring_tokens
            ]
        else:  # color_by == 'after'
            color_categories = [
                neighbors[0][0] if neighbors and len(neighbors[0]) > 1 else "Unknown"
                for neighbors in neighboring_tokens
            ]
    else:
        raise ValueError("color_by must be 'token', 'before', or 'after'")

    # Get unique categories and assign colors
    unique_categories = sorted(set(color_categories))
    color_map = {
        cat: alphabet_colors[i % len(alphabet_colors)]
        for i, cat in enumerate(unique_categories)
    }

    # Add traces for each subplot
    for row, col, x, y in [
        (1, 1, "PC1", "PC2"),
        (1, 2, "PC1", "PC3"),
        (2, 1, "PC2", "PC3"),
    ]:
        for category in unique_categories:
            mask = [cat == category for cat in color_categories]
            fig.add_trace(
                go.Scatter(
                    x=pca_df[x][mask],
                    y=pca_df[y][mask],
                    mode="markers",
                    marker=dict(color=color_map[category]),
                    name=category,
                    text=pca_df["tokens"][mask],
                    hoverinfo="text",
                    hovertext=[
                        f"Point ID: {id}<br>Token: {t}<br>Context: {c}<br>Color Category: {cat}"
                        for id, t, c, cat in zip(
                            pca_df["point_id"][mask],
                            pca_df["tokens"][mask],
                            pca_df["context"][mask],
                            [category] * sum(mask),
                        )
                    ],
                    showlegend=(
                        row == 1 and col == 1
                    ),  # Only show legend for the first subplot
                ),
                row=row,
                col=col,
            )

        fig.update_xaxes(title_text=x, row=row, col=col)
        fig.update_yaxes(title_text=y, row=row, col=col)

    # Update layout
    fig.update_layout(
        height=1600,
        width=1600,
        title_text=f"PCA Subspace Reconstructions (Colored by {color_by})",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.05),
    )

    # Create the directory if it doesn't exist
    os.makedirs(pca_path, exist_ok=True)

    if save:
        # Save as PNG
        png_path = os.path.join(
            pca_path, f"pca_plot_graph_{fs_splitting_cluster}_{color_by}.png"
        )
        fig.write_image(png_path, scale=4.0)

        svg_path = os.path.join(
            pca_path, f"pca_plot_graph_{fs_splitting_cluster}_{color_by}.svg"
        )
        fig.write_image(svg_path)

        # Save as HTML
        html_path = os.path.join(
            pca_path, f"pca_plot_graph_{fs_splitting_cluster}_{color_by}.html"
        )
        fig.write_html(html_path)

        # print(f"Plots saved as PNG: {png_path}")
        # print(f"Plots saved as HTML: {html_path}")
    else:
        fig.show()


def plot_pca_explanation_and_save(pca, pca_path, fs_splitting_cluster, save=False):
    fig = px.bar(pca.explained_variance_ratio_, width=400, height=200)
    png_path = os.path.join(
        pca_path, f"pca_explanation_graph_{fs_splitting_cluster}.png"
    )
    svg_path = os.path.join(
        pca_path, f"pca_explanation_graph_{fs_splitting_cluster}.svg"
    )
    if save:
        fig.write_image(png_path, scale=4.0)
        fig.write_image(svg_path)
    else:
        fig.show()
    return None


def plot_simple_scatter(
    results, pca_path, fs_splitting_cluster, fs_splitting_nodes, save=False
):
    fig = px.scatter_matrix(
        results.all_graph_feature_acts.cpu().numpy(),
        dimensions=list(range(len(fs_splitting_nodes))),
        height=800,
        width=1200,
        opacity=0.2,
    )
    png_path = os.path.join(pca_path, f"scatter_plot_graph_{fs_splitting_cluster}.png")
    svg_path = os.path.join(pca_path, f"scatter_plot_graph_{fs_splitting_cluster}.svg")
    if save:
        fig.write_image(png_path, scale=4.0)
        fig.write_image(svg_path)
    else:
        fig.show()
    return None


def calculate_pca_decoder(sae, fs_splitting_nodes):
    # Perform PCA
    pca = PCA(n_components=3)
    pca_embedding = pca.fit_transform(sae.W_dec[fs_splitting_nodes].cpu().numpy())
    pca_df = pd.DataFrame(
        pca_embedding, columns=pd.Index([f"PC{i+1}" for i in range(3)])
    )
    return pca, pca_df


def create_pca_plots_decoder(pca_df, fs_splitting_cluster, pca_path, save=False):
    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("PC1 vs PC2", "PC2 vs PC3", "PC1 vs PC3"),
        horizontal_spacing=0.1,
    )

    # PC1 vs PC2
    fig.add_trace(
        go.Scatter(x=pca_df["PC1"], y=pca_df["PC2"], mode="markers"), row=1, col=1
    )
    fig.update_xaxes(title_text="PC1", row=1, col=1)
    fig.update_yaxes(title_text="PC2", row=1, col=1)

    # PC2 vs PC3
    fig.add_trace(
        go.Scatter(x=pca_df["PC2"], y=pca_df["PC3"], mode="markers"), row=1, col=2
    )
    fig.update_xaxes(title_text="PC2", row=1, col=2)
    fig.update_yaxes(title_text="PC3", row=1, col=2)

    # PC1 vs PC3
    fig.add_trace(
        go.Scatter(x=pca_df["PC1"], y=pca_df["PC3"], mode="markers"), row=1, col=3
    )
    fig.update_xaxes(title_text="PC1", row=1, col=3)
    fig.update_yaxes(title_text="PC3", row=1, col=3)

    # Update layout
    fig.update_layout(
        height=600,
        width=1800,
        title_text="PCA Subspace Reconstructions",
        showlegend=False,
    )

    if save:
        # Save as PNG
        png_path = os.path.join(
            pca_path, f"pca_decoder_graph_{fs_splitting_cluster}.png"
        )
        fig.write_image(png_path, scale=4.0)

        svg_path = os.path.join(
            pca_path, f"pca_decoder_graph_{fs_splitting_cluster}.svg"
        )
        fig.write_image(svg_path)

        # Save as HTML
        html_path = os.path.join(
            pca_path, f"pca_decoder_graph_{fs_splitting_cluster}.html"
        )
        fig.write_html(html_path)
    else:
        fig.show()

    return fig


def plot_doubly_clustered_activation_heatmap(
    results,
    fs_splitting_nodes,
    pca_df,
    pca_path,
    fs_splitting_cluster,
    max_examples=100,
    save=False,
):
    # Extract feature activations
    feature_activations = results.all_graph_feature_acts.cpu().numpy()

    # Limit the number of examples if there are too many
    n_examples = min(feature_activations.shape[0], max_examples)
    feature_activations = feature_activations[:n_examples]

    # Perform hierarchical clustering on the features (rows)
    feature_distances = pdist(feature_activations.T)
    feature_linkage = linkage(feature_distances, method="ward")
    feature_dendrogram = dendrogram(feature_linkage, no_plot=True)
    feature_order = feature_dendrogram["leaves"]

    # Perform hierarchical clustering on the examples (columns)
    example_distances = pdist(feature_activations)
    example_linkage = linkage(example_distances, method="ward")
    example_dendrogram = dendrogram(example_linkage, no_plot=True)
    example_order = example_dendrogram["leaves"]

    # Reorder the feature activations, feature names, and example names
    feature_activations_reordered = feature_activations.T[feature_order][
        :, example_order
    ]
    feature_names_reordered = [fs_splitting_nodes[i] for i in feature_order]
    example_names_reordered = [f"Example {i+1}" for i in example_order]

    # Prepare hover text
    hover_text = []
    for feature_idx in feature_order:
        feature_hover = []
        for example_idx in example_order:
            token = pca_df["tokens"].iloc[example_idx]
            context = pca_df["context"].iloc[example_idx]
            activation = feature_activations[example_idx, feature_idx]
            hover_info = f"Feature: {fs_splitting_nodes[feature_idx]}<br>Example: {example_idx+1}<br>Token: {token}<br>Context: {context}<br>Activation: {activation:.4f}"
            feature_hover.append(hover_info)
        hover_text.append(feature_hover)

    # Create the heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=feature_activations_reordered,
            x=example_names_reordered,
            y=[f"Feature {node}" for node in feature_names_reordered],
            colorscale="Viridis",
            colorbar=dict(title="Activation Strength"),
            hoverinfo="text",
            text=hover_text,
        )
    )

    # Update layout
    fig.update_layout(
        xaxis_title="Clustered Examples",
        yaxis_title="Clustered Features",
        width=1200,
        height=800,
        yaxis=dict(
            autorange="reversed"
        ),  # To have the first feature cluster at the top
    )

    # Show the plot
    if save:
        # Save as PNG
        png_path = os.path.join(pca_path, f"pca_heatmap_{fs_splitting_cluster}.png")
        fig.write_image(png_path, scale=4.0)

        svg_path = os.path.join(pca_path, f"pca_heatmap_{fs_splitting_cluster}.svg")
        fig.write_image(svg_path)

        # Save as HTML
        html_path = os.path.join(pca_path, f"pca_heatmap_{fs_splitting_cluster}.html")
        fig.write_html(html_path)
    else:
        fig.show()


# Feature activation bar plots ------------------


def prepare_data(results, fs_splitting_nodes, node_df):
    """
    Prepare data for plotting, including subgraph information.

    Args:
    results (ProcessedExamples): The results from process_examples function
    fs_splitting_nodes (list): list of feature splitting nodes
    node_df (pd.DataFrame): DataFrame containing node information including subgraph_id and subgraph_size

    Returns:
    pd.DataFrame: Prepared data for plotting
    str: First context from the results
    """
    first_result = results.all_feature_acts[0].cpu().numpy()
    first_context = results.all_token_dfs["context"].iloc[0]

    df = pd.DataFrame(
        {
            "Feature Index": np.arange(len(first_result)),
            "Activation": first_result,
            "Is Feature Splitting": np.isin(
                np.arange(len(first_result)), fs_splitting_nodes
            ),
        }
    )

    # Convert Feature Index to string for easier merging
    df["Feature Index"] = df["Feature Index"].astype(str)

    # Merge with node_df to get subgraph information
    node_df_copy = node_df.copy()
    node_df_copy["node_id"] = node_df_copy["node_id"].astype(str)
    df = df.merge(
        node_df_copy[["node_id", "subgraph_id", "subgraph_size"]],
        left_on="Feature Index",
        right_on="node_id",
        how="left",
    )

    # Fill NaN values for features not in node_df
    df["subgraph_id"] = df["subgraph_id"].fillna("Not in subgraph")
    df["subgraph_size"] = df["subgraph_size"].fillna(0)

    # Remove rows with zero activation
    df = df[df["Activation"] != 0]

    # Sort by activation in descending order
    df = df.sort_values("Activation", ascending=False)  # type: ignore

    return df, first_context


def create_bar_plot(
    df,
    context,
    fs_splitting_cluster,
    color_other_subgraphs=False,
    order_other_subgraphs=False,
):
    """
    Create bar plot of feature activations with specific ordering and highlighting.

    Args:
    df (pd.DataFrame): DataFrame containing the data
    context (str): Context for the plot title
    fs_splitting_nodes (list): list of feature splitting nodes
    fs_splitting_cluster (int): The subgraph ID of the feature splitting cluster

    Returns:
    go.Figure: Plotly figure object
    """
    import plotly.express as px
    import plotly.graph_objects as go

    # Create a copy of the dataframe to avoid modifying the original
    plot_df = df.copy()

    # Create a category column for sorting

    plot_df["Category"] = plot_df.apply(
        lambda row: assign_category(row, fs_splitting_cluster, order_other_subgraphs),
        axis=1,
    )

    # For non-fs_splitting_cluster subgraphs, find the max activation
    subgraph_max_activations = (
        plot_df[plot_df["Category"] == 1].groupby("subgraph_id")["Activation"].max()
    )
    subgraph_order = subgraph_max_activations.sort_values(ascending=False).index

    # Create a mapping for subgraph order
    subgraph_order_map = {
        subgraph: order for order, subgraph in enumerate(subgraph_order)
    }

    # Assign order within category
    plot_df["SubgraphOrder"] = plot_df["subgraph_id"].map(subgraph_order_map)
    plot_df["SubgraphOrder"] = plot_df["SubgraphOrder"].fillna(len(subgraph_order_map))

    # Sort the dataframe
    plot_df = plot_df.sort_values(
        ["Category", "SubgraphOrder", "Activation"], ascending=[True, True, False]
    )

    # Reset index to get the new order
    plot_df = plot_df.reset_index(drop=True)

    # Create color column
    def assign_color(row, color_other_subgraphs):
        if row["subgraph_id"] == fs_splitting_cluster:
            return "fs_splitting_cluster"
        elif row["Category"] == 1 and color_other_subgraphs:
            return f'subgraph_{row["subgraph_id"]}'
        else:
            return "Other"

    plot_df["Color"] = plot_df.apply(
        lambda row: assign_color(row, color_other_subgraphs), axis=1
    )

    # Create a color map
    color_map = {"fs_splitting_cluster": "red", "Other": "grey"}
    other_colors = px.colors.qualitative.Set2  # You can choose any color palette
    for i, subgraph in enumerate(subgraph_order):
        color_map[f"subgraph_{subgraph}"] = other_colors[i % len(other_colors)]

    # Set opacity based on fs_splitting_cluster
    plot_df["Opacity"] = plot_df["subgraph_id"].apply(
        lambda x: 1.0 if x == fs_splitting_cluster else 0.5
    )

    # Create the figure
    fig = go.Figure()

    # Add bars for each color group
    for color_group in plot_df["Color"].unique():
        group_df = plot_df[plot_df["Color"] == color_group]
        fig.add_trace(
            go.Bar(
                x=group_df.index,
                y=group_df["Activation"],
                name=color_group,
                marker_color=color_map[color_group],
                # opacity=group_df['Opacity'],
                hovertemplate="Feature Index: %{customdata[0]}<br>Activation: %{y:.4f}<br>Subgraph ID: %{customdata[1]}<br>Subgraph Size: %{customdata[2]}<extra></extra>",
                customdata=group_df[
                    ["Feature Index", "subgraph_id", "subgraph_size"]
                ].values,
            )
        )

    # Update layout
    fig.update_layout(
        title=context,
        xaxis_title="Feature Index (Sorted)",
        yaxis_title="Feature Activation",
        legend_title_text="Subgraph ID",
        showlegend=True,
        height=600,
        width=1000,
        bargap=0.2,
    )

    # Update x-axis to show original Feature Index
    fig.update_xaxes(
        tickmode="array",
        tickvals=plot_df.index,
        ticktext=plot_df["Feature Index"],
        tickangle=45,
    )

    # Add mean activation line
    mean_activation = plot_df["Activation"].mean()
    fig.add_hline(
        y=mean_activation,
        line_dash="dash",
        line_color="green",
        annotation_text="Mean activation",
        annotation_position="bottom right",
    )

    return fig


def create_pie_charts(df, activation_threshold, context, color=None):
    """Create pie charts for feature activation comparison."""
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "domain"}, {"type": "domain"}]],
        subplot_titles=(
            "Active Features Above Threshold",
            "Sum of Activation Strengths",
        ),
    )

    colors = [color, "lightgrey"] if color else ["red", "blue"]

    active_splitting = sum(
        (df["Is Feature Splitting"]) & (df["Activation"] > activation_threshold)
    )
    active_non_splitting = sum(
        (~df["Is Feature Splitting"]) & (df["Activation"] > activation_threshold)
    )

    fig.add_trace(
        go.Pie(
            labels=["Splitting", "Non-splitting"],
            values=[active_splitting, active_non_splitting],
        ),
        1,
        1,
    )

    sum_splitting = df[df["Is Feature Splitting"]]["Activation"].sum()
    sum_non_splitting = df[~df["Is Feature Splitting"]]["Activation"].sum()

    fig.add_trace(
        go.Pie(
            labels=["Splitting", "Non-splitting"],
            values=[sum_splitting, sum_non_splitting],
        ),
        1,
        2,
    )

    fig.update_traces(
        marker=dict(colors=colors), hole=0.4, hoverinfo="label+percent+value"
    )
    fig.update_layout(height=400, width=800, title_text=context)

    return fig


def print_statistics(df, fs_splitting_nodes, activation_threshold):
    """Print statistics about feature activations."""
    print(f"Number of non-zero features: {len(df)}")
    print(
        f"Number of non-zero feature splitting nodes: {sum(df['Is Feature Splitting'])}"
    )
    print(f"Total number of feature splitting nodes: {len(fs_splitting_nodes)}")
    print(
        f"Mean activation of non-zero feature splitting nodes: {df[df['Is Feature Splitting']]['Activation'].mean():.4f}"
    )
    print(
        f"Mean activation of non-zero non-feature splitting nodes: {df[~df['Is Feature Splitting']]['Activation'].mean():.4f}"
    )
    print(
        f"Median activation of non-zero feature splitting nodes: {df[df['Is Feature Splitting']]['Activation'].median():.4f}"
    )
    print(
        f"Median activation of non-zero non-feature splitting nodes: {df[~df['Is Feature Splitting']]['Activation'].median():.4f}"
    )
    print(
        f"Number of splitting features active above threshold: {sum((df['Is Feature Splitting']) & (df['Activation'] > activation_threshold))}"
    )
    print(
        f"Number of non-splitting features active above threshold: {sum((~df['Is Feature Splitting']) & (df['Activation'] > activation_threshold))}"
    )
    print(
        f"Sum of activation strengths for splitting features: {df[df['Is Feature Splitting']]['Activation'].sum():.4f}"
    )
    print(
        f"Sum of activation strengths for non-splitting features: {df[~df['Is Feature Splitting']]['Activation'].sum():.4f}"
    )


def get_point_result(
    results: ProcessedExamples | ReprocessedResults, idx: int
) -> ProcessedExamples | ReprocessedResults:
    if isinstance(results, ReprocessedResults):
        point_result = ReprocessedResults(
            all_graph_feature_acts=results.all_graph_feature_acts[[idx]],
            all_max_feature_info=results.all_max_feature_info[[idx]],
        )
    else:
        point_result = ProcessedExamples(
            all_token_dfs=results.all_token_dfs.iloc[[idx]],
            all_fired_tokens=[results.all_fired_tokens[idx]],
            all_reconstructions=results.all_reconstructions[[idx]],
            all_graph_feature_acts=results.all_graph_feature_acts[[idx]],
            all_feature_acts=results.all_feature_acts[[idx]],
            all_max_feature_info=results.all_max_feature_info[[idx]],
            all_examples_found=1,
        )
    return point_result


def select_representative_points(pca_df, n_extremes=3, n_middle=2):
    """
    Select representative points from the PCA plot.

    Args:
    pca_df (pd.DataFrame): DataFrame containing PCA results
    n_extremes (int): Number of extreme points to select
    n_middle (int): Number of middle points to select

    Returns:
    list: Indices of selected points
    """
    # Normalize PCA components to [0, 1] range
    scaler = MinMaxScaler()
    normalized_pca = scaler.fit_transform(pca_df[["PC1", "PC2", "PC3"]])

    # Calculate distance from center (0.5, 0.5, 0.5)
    distances = np.linalg.norm(normalized_pca - 0.5, axis=1)

    # Select extreme points
    extreme_indices = distances.argsort()[-n_extremes:][::-1]

    # Select middle points
    middle_indices = np.argsort(np.abs(distances - np.median(distances)))[:n_middle]

    return list(extreme_indices) + list(middle_indices)


def plot_pca_with_highlights(pca_df, highlighted_indices):
    """
    Create a 3D scatter plot of PCA results with highlighted points.

    Args:
    pca_df (pd.DataFrame): DataFrame containing PCA results
    highlighted_indices (list): Indices of points to highlight

    Returns:
    go.Figure: Plotly figure object
    """
    fig = go.Figure()

    # Add all points
    fig.add_trace(
        go.Scatter3d(
            x=pca_df["PC1"],
            y=pca_df["PC2"],
            z=pca_df["PC3"],
            mode="markers",
            marker=dict(size=4, color="grey", opacity=0.8),
            name="All Points",
        )
    )

    # Add highlighted points
    highlighted_df = pca_df.iloc[highlighted_indices]
    fig.add_trace(
        go.Scatter3d(
            x=highlighted_df["PC1"],
            y=highlighted_df["PC2"],
            z=highlighted_df["PC3"],
            mode="markers",
            marker=dict(size=8, color="red", symbol="star"),
            name="Highlighted Points",
        )
    )

    fig.update_layout(
        title="PCA Plot with Highlighted Representative Points",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        width=800,
        height=800,
        legend=dict(x=0.7, y=0.9),
    )

    return fig


def plot_pca_with_highlights_2d(pca_df, highlighted_indices, contexts):
    """
    Create three 2D scatter plots of PCA results with highlighted points.

    Args:
    pca_df (pd.DataFrame): DataFrame containing PCA results
    highlighted_indices (list): Indices of points to highlight
    contexts (list): list of context strings for highlighted points

    Returns:
    go.Figure: Plotly figure object
    """
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("PC1 vs PC2", "PC1 vs PC3", "PC2 vs PC3"),
        shared_yaxes=True,
    )

    # Color palette for highlighted points
    colors = px.colors.qualitative.Set1[: len(highlighted_indices)]

    pc_combinations = [("PC1", "PC2"), ("PC1", "PC3"), ("PC2", "PC3")]
    for i, (pc_x, pc_y) in enumerate(pc_combinations):
        # Add all points
        fig.add_trace(
            go.Scatter(
                x=pca_df[pc_x],
                y=pca_df[pc_y],
                mode="markers",
                marker=dict(size=4, color="lightgrey", opacity=0.6),
                name="All Points",
                showlegend=(i == 0),
            ),
            row=1,
            col=i + 1,
        )

        # Add highlighted points
        for j, idx in enumerate(highlighted_indices):
            fig.add_trace(
                go.Scatter(
                    x=[pca_df[pc_x].iloc[idx]],
                    y=[pca_df[pc_y].iloc[idx]],
                    mode="markers",
                    marker=dict(size=10, color=colors[j], symbol="star"),
                    name=f"Point {j+1}",
                    showlegend=(i == 0),
                    text=contexts[j],
                    hoverinfo="text",
                ),
                row=1,
                col=i + 1,
            )

        fig.update_xaxes(title_text=pc_x, row=1, col=i + 1)
        fig.update_yaxes(title_text=pc_y, row=1, col=i + 1)

    fig.update_layout(
        title="PCA Plots with Highlighted Representative Points",
        height=500,
        width=1500,
        legend=dict(x=1.05, y=0.5),
    )

    return fig


def analyze_representative_points(
    results,
    fs_splitting_nodes,
    fs_splitting_cluster,
    activation_threshold,
    node_df,
    results_path,
    pca_df,
    save_figs=False,
    pca_path=None,
):
    """
    Analyze and visualize representative points from the PCA plot.

    Args:
    results (ProcessedExamples): Results from process_examples function
    fs_splitting_nodes (list): list of feature splitting node indices
    activation_threshold (float): Threshold for considering a feature as active
    pca_df (pd.DataFrame): DataFrame containing PCA results
    save_figs (bool): Whether to save the figures
    pca_path (str): Path to save the figures
    """
    # Select representative points
    rep_indices = select_representative_points(pca_df)

    # Get contexts for representative points
    contexts = [results.all_token_dfs["context"].iloc[idx] for idx in rep_indices]

    # Plot PCA with highlighted points
    pca_fig = plot_pca_with_highlights_2d(pca_df, rep_indices, contexts)
    pca_fig.show()
    if save_figs and pca_path:
        pca_fig.write_image(pj(pca_path, "pca_with_highlights.png"), scale=4.0)
        pca_fig.write_image(pj(pca_path, "pca_with_highlights.svg"))
        pca_fig.write_image(pj(pca_path, "pca_with_highlights.pdf"))
        pca_fig.write_html(pj(pca_path, "pca_with_highlights.html"))

    # Color palette for plots
    colors = px.colors.qualitative.Set1[: len(rep_indices)]

    # Analyze each representative point
    for i, idx in enumerate(rep_indices):
        print(f"\nAnalyzing representative point {i+1}:")

        # Extract data for this point
        point_result = get_point_result(results, idx)

        # Create plots for this point with color coding
        plot_feature_activations(
            point_result,
            fs_splitting_nodes,
            fs_splitting_cluster,
            activation_threshold,
            node_df,
            results_path,
            save_figs=save_figs,
            pca_path=pj(pca_path, f"rep_point_{i+1}") if pca_path else None,
            color=colors[i],
        )


def plot_feature_activations(
    results,
    fs_splitting_nodes,
    fs_splitting_cluster,
    activation_threshold,
    node_df,
    results_path,
    save_figs=False,
    pca_path=None,
    color=None,
):
    """Main function to create and display all plots."""
    df, context = prepare_data(results, fs_splitting_nodes, node_df)

    if pca_path is not None:
        if not os.path.exists(pca_path):
            os.makedirs(pca_path)

    activation_array = results.all_feature_acts.flatten().cpu().numpy()

    # Get all active subgraphs of size > 1
    active_subgraphs = get_active_subgraphs(df, activation_threshold, results_path)

    bar_fig = create_bar_plot(df, context, fs_splitting_nodes, fs_splitting_cluster)
    pie_fig = create_pie_charts(df, activation_threshold, context, color)

    # Plot all active subgraphs
    subgraph_figs = []
    for subgraph_id, subgraph in active_subgraphs.items():
        subgraph_path = (
            pj(pca_path, f"subgraph_{subgraph_id}") if save_figs and pca_path else None
        )
        subgraph_fig = plot_subgraph_static(
            subgraph,
            node_df,
            subgraph_path,
            activation_array,
            save_figs=save_figs,
        )
        subgraph_figs.append(subgraph_fig)

    bar_fig.show()
    pie_fig.show()
    # for fig in subgraph_figs:
    #     fig.show()

    if save_figs and pca_path:
        bar_fig.write_image(
            pj(pca_path, "non_zero_feature_activations_comparison.png"), scale=4.0
        )
        bar_fig.write_image(pj(pca_path, "non_zero_feature_activations_comparison.svg"))
        bar_fig.write_image(pj(pca_path, "non_zero_feature_activations_comparison.pdf"))
        bar_fig.write_html(pj(pca_path, "non_zero_feature_activations_comparison.html"))
        pie_fig.write_image(
            pj(pca_path, "feature_activation_pie_charts.png"), scale=4.0
        )
        pie_fig.write_image(pj(pca_path, "feature_activation_pie_charts.svg"))
        pie_fig.write_image(pj(pca_path, "feature_activation_pie_charts.pdf"))
        pie_fig.write_html(pj(pca_path, "feature_activation_pie_charts.html"))

    print_statistics(df, fs_splitting_nodes, activation_threshold)


def get_active_subgraphs(df, activation_threshold, results_path):
    """Get all active subgraphs of size > 1."""
    active_subgraphs = {}
    active_subgraph_ids = df[
        (df["Activation"] > activation_threshold) & (df["subgraph_size"] > 4)
    ]["subgraph_id"].unique()

    for subgraph_id in active_subgraph_ids:
        subgraph = load_subgraph(results_path, activation_threshold, subgraph_id)
        active_subgraphs[subgraph_id] = subgraph

    return active_subgraphs


def get_active_subgraph_ids(df, activation_threshold):
    """Get all active subgraph IDs of size > 1."""
    active_subgraph_ids = df[
        (df["Activation"] > activation_threshold) & (df["subgraph_size"] > 4)
    ]["subgraph_id"].unique()
    return active_subgraph_ids


def create_bar_plot_specific(
    df,
    context,
    fs_splitting_nodes,
    fs_splitting_cluster,
    highlight_color,
    color_other_subgraphs=False,
    order_other_subgraphs=False,
    height=600,
    width=1000,
    plot_only_fs_nodes=False,
):
    """
    Create bar plot of feature activations with specific ordering and highlighting.

    Args:
    df (pd.DataFrame): DataFrame containing the data
    context (str): Context for the plot title
    fs_splitting_nodes (list): list of feature splitting nodes
    fs_splitting_cluster (int): The subgraph ID of the feature splitting cluster
    highlight_color (str): Color to use for the fs_splitting_cluster
    color_other_subgraphs (bool): Whether to color other subgraphs
    order_other_subgraphs (bool): Whether to order other subgraphs
    height (int): Height of the plot
    width (int): Width of the plot
    plot_only_fs_nodes (bool): Whether to plot only features in fs_splitting_nodes

    Returns:
    go.Figure: Plotly figure object
    """
    # Create a copy of the dataframe to avoid modifying the original
    plot_df = df.copy()

    # Ensure 'Feature Index' is integer type
    plot_df["Feature Index"] = plot_df["Feature Index"].astype(int)

    # Add missing fs_splitting_nodes with activity 0
    missing_nodes = set(fs_splitting_nodes) - set(plot_df["Feature Index"])
    if missing_nodes:
        missing_df = pd.DataFrame(
            {
                "Feature Index": list(missing_nodes),
                "Activation": [0] * len(missing_nodes),
                "subgraph_id": [None] * len(missing_nodes),
                "subgraph_size": [None] * len(missing_nodes),
            }
        )
        plot_df = pd.concat([plot_df, missing_df], ignore_index=True)

    # Filter the dataframe if plot_only_fs_nodes is True
    if plot_only_fs_nodes:
        plot_df = plot_df[plot_df["Feature Index"].isin(fs_splitting_nodes)]

    # Create a category column for sorting
    def assign_category(row, fs_splitting_cluster, order_other_subgraphs):
        if row["subgraph_id"] == fs_splitting_cluster:
            return 0
        elif order_other_subgraphs and row["subgraph_id"] is not None:
            return row["subgraph_id"]
        else:
            return 1

    plot_df["Category"] = plot_df.apply(
        lambda row: assign_category(row, fs_splitting_cluster, order_other_subgraphs),
        axis=1,
    )

    # Sort the dataframe
    if plot_only_fs_nodes:
        plot_df = plot_df.sort_values("Feature Index")  # type: ignore
    else:
        plot_df = plot_df.sort_values(
            ["Category", "Activation"], ascending=[True, False]
        )  # type: ignore

    # Reset index to get the new order
    plot_df = plot_df.reset_index(drop=True)

    # Create color column
    def assign_color(row, color_other_subgraphs):
        if row["subgraph_id"] == fs_splitting_cluster:
            return highlight_color
        elif row["Category"] != 1 and color_other_subgraphs:
            return "darkgrey"
        else:
            return "lightgrey"

    plot_df["Color"] = plot_df.apply(
        lambda row: assign_color(row, color_other_subgraphs), axis=1
    )

    # Convert 'Feature Index' to string for plotting
    plot_df["Feature Index"] = plot_df["Feature Index"].astype(str)

    # Create the figure
    fig = go.Figure()

    # Add bars
    fig.add_trace(
        go.Bar(
            x=plot_df["Feature Index"],
            y=plot_df["Activation"],
            marker_color=plot_df["Color"],
            hovertemplate="Feature Index: %{x}<br>Activation: %{y:.4f}<br>Subgraph ID: %{customdata[0]}<br>Subgraph Size: %{customdata[1]}<extra></extra>",
            customdata=plot_df[["subgraph_id", "subgraph_size"]].values,
        )
    )

    # Update layout
    fig.update_layout(
        title=context,
        xaxis_title="Feature Index",
        yaxis_title="Feature Activation",
        showlegend=False,
        height=height,
        width=width,
        bargap=0.2,
    )

    # Update x-axis to show Feature Index
    fig.update_xaxes(
        type="category",
        tickmode="array",
        tickvals=plot_df["Feature Index"],
        ticktext=plot_df["Feature Index"],
        tickangle=45,
    )

    # Add mean activation line
    mean_activation = plot_df["Activation"].mean()
    fig.add_hline(
        y=mean_activation,
        line_dash="dash",
        line_color="green",
        annotation_text="Mean activation",
        annotation_position="bottom right",
    )

    return fig


def analyze_specific_points(
    results,
    fs_splitting_nodes,
    fs_splitting_cluster,
    activation_threshold,
    node_df,
    results_path,
    pca_df,
    point_ids,
    plot_only_fs_nodes=False,
    subdir=None,
    save_figs=False,
    pca_path=None,
):
    """
    Analyze and visualize specific points from the PCA plot based on their IDs.
    """
    # Color palette for plots
    colors = px.colors.qualitative.Safe[: len(point_ids)]

    for i, point_id in enumerate(point_ids):
        print(f"\nAnalyzing point with ID {point_id}:")

        # Extract data for this point
        point_result = get_point_result(results, point_id)

        if isinstance(point_result, ReprocessedResults):
            raise ValueError(
                "Passed ReprocessedResults, but cannot plot other subgraphs from Streamlit data. "
                "Please set plot_without_other_subgraphs=True when loading data from generation or pass ProcessedExamples."
            )

        point_pca_path = pj(pca_path, f"point_{point_id}") if pca_path else None

        if point_pca_path is not None:
            if not os.path.exists(point_pca_path):
                os.makedirs(point_pca_path)

        # Prepare data for plotting
        df, context = prepare_data(point_result, fs_splitting_nodes, node_df)

        # Create bar plot
        bar_fig = create_bar_plot_specific(
            df,
            context,
            fs_splitting_nodes,
            fs_splitting_cluster,
            colors[i],
            height=800,
            width=800,
            plot_only_fs_nodes=plot_only_fs_nodes,
        )

        # Create pie charts
        pie_fig = create_pie_charts(df, activation_threshold, context, color=colors[i])

        # Get active subgraphs
        activation_array = point_result.all_feature_acts.flatten().cpu().numpy()
        active_subgraphs = get_active_subgraphs(df, activation_threshold, results_path)

        if save_figs and pca_path:
            if subdir is not None:
                point_path = pj(pca_path, subdir, f"point_{point_id}")
            else:
                point_path = pj(pca_path, f"point_{point_id}")
            if not os.path.exists(point_path):
                os.makedirs(point_path)
        else:
            point_path = None

        # Plot active subgraphs
        subgraph_figs = []
        for subgraph_id, subgraph in active_subgraphs.items():
            # subgraph_path = (
            #     pj(pca_path, f"point_{point_id}", f"subgraph_{subgraph_id}")
            #     if save_figs and pca_path
            #     else None
            # )
            if save_figs and pca_path:
                if point_path is None:
                    raise ValueError("point_path is None")
                if subdir is not None:
                    subgraph_path = pj(point_path, f"subgraph_{subgraph_id}")
                else:
                    subgraph_path = pj(point_path, f"subgraph_{subgraph_id}")
                subgraph_fig = plot_subgraph_static(
                    subgraph,
                    node_df,
                    subgraph_path,
                    activation_array,
                    save_figs=save_figs,
                )
                subgraph_figs.append(subgraph_fig)

        # Save or show figures
        if save_figs and pca_path:
            # if subdir is not None:
            #     point_path = pj(pca_path, subdir, f"point_{point_id}")
            # else:
            #     point_path = pj(pca_path, f"point_{point_id}")
            if point_path is None:
                raise ValueError("point_path is None")
            os.makedirs(point_path, exist_ok=True)
            bar_fig.write_image(
                pj(point_path, f"bar_plot_point_{point_id}.png"), scale=4.0
            )
            bar_fig.write_html(pj(point_path, f"bar_plot_point_{point_id}.html"))
            pie_fig.write_image(
                pj(point_path, f"pie_charts_point_{point_id}.png"), scale=4.0
            )
            pie_fig.write_html(pj(point_path, f"pie_charts_point_{point_id}.html"))

        bar_fig.show()
        pie_fig.show()
        for fig in subgraph_figs:
            if fig is not None:
                fig.show()
            else:
                print(
                    "Subgraph figure is None. Likely no latents are within a subgraph."
                )

        # Print statistics for this point
        # print_statistics(df, fs_splitting_nodes, activation_threshold)

        # Create PCA plot (PC2 vs PC3) for this point
        fig = go.Figure()

        # Add all points in grey
        fig.add_trace(
            go.Scatter(
                x=pca_df["PC2"],
                y=pca_df["PC3"],
                mode="markers",
                marker=dict(
                    color="lightgrey",
                    size=10,
                    line=dict(width=2, color="DarkSlateGrey"),
                ),
                name="Other points",
                hoverinfo="none",
            )
        )

        # Add the specific point in color
        fig.add_trace(
            go.Scatter(
                x=[pca_df.loc[point_id, "PC2"]],
                y=[pca_df.loc[point_id, "PC3"]],
                mode="markers",
                marker=dict(
                    color=colors[i],
                    size=15,
                    symbol="star",
                    line=dict(width=2, color="DarkSlateGrey"),
                ),
                name=f"Point {point_id}",
                text=pca_df.loc[point_id, "context"],
                hoverinfo="text",
            )
        )

        fig.update_layout(
            title=f"'{context}' - {point_id}",
            xaxis_title="PC2",
            yaxis_title="PC3",
            width=800,
            height=800,
            showlegend=False,
            # template = 'plotly_white'
        )

        fig.update_xaxes(constrain="domain")
        fig.update_yaxes(scaleanchor="x")

        if save_figs and pca_path:
            if point_path is None:
                raise ValueError("point_path is None")
            fig.write_image(
                pj(point_path, f"pca_pc2_vs_pc3_point_{point_id}.png"), scale=4.0
            )
            fig.write_html(pj(point_path, f"pca_pc2_vs_pc3_point_{point_id}.html"))
        else:
            fig.show()


def analyze_specific_points_from_thresholded(
    results: ProcessedExamples | ReprocessedResults,
    thresholded_matrix: np.ndarray,
    fs_splitting_nodes: list[int],
    fs_splitting_cluster: int,
    activation_threshold: float,
    node_df: pd.DataFrame,
    pca_df: pd.DataFrame,
    point_ids: list[int],
    plot_only_fs_nodes: bool = False,
    subdir: str | None = None,
    save_figs: bool = False,
    pca_path: str | None = None,
    plot_without_other_subgraphs: bool = False,
) -> None:
    """
    Analyze and visualize specific points from the PCA plot based on their IDs.
    """
    # Color palette for plots
    colors = px.colors.qualitative.Safe[: len(point_ids)]

    for i, point_id in enumerate(point_ids):
        print(f"\nAnalyzing point with ID {point_id}:")

        # Extract data for this point
        point_result = get_point_result(results, point_id)
        point_pca_path = pj(pca_path, f"point_{point_id}") if pca_path else None

        if point_pca_path is not None:
            if not os.path.exists(point_pca_path):
                os.makedirs(point_pca_path)

        # Prepare data for plotting
        df, context = prepare_data(point_result, fs_splitting_nodes, node_df)

        # Create bar plot
        bar_fig = create_bar_plot_specific(
            df,
            context,
            fs_splitting_nodes,
            fs_splitting_cluster,
            colors[i],
            height=800,
            width=800,
            plot_only_fs_nodes=plot_only_fs_nodes,
        )

        # Create pie charts
        pie_fig = create_pie_charts(df, activation_threshold, context, color=colors[i])

        # Get active subgraphs
        if plot_without_other_subgraphs:
            # If loading data from the generation for streamlit then all graph feature apps
            # will not have been saved and therefore we cannot plot the feature
            # activations of the other subgraphs
            activation_array = (
                point_result.all_graph_feature_acts.flatten().cpu().numpy()
            )
            active_subgraph_ids = [fs_splitting_cluster]
        elif isinstance(point_result, ProcessedExamples):
            activation_array = point_result.all_feature_acts.flatten().cpu().numpy()
            # If all feature acts is not part of point result raise a error that we cannot plot other subgraphs from Streamlit data
            active_subgraph_ids = get_active_subgraph_ids(df, activation_threshold)
        else:
            raise ValueError(
                "Passed ReprocessedResults, but cannot plot other subgraphs from Streamlit data. "
                "Please set plot_without_other_subgraphs=True when loading data from generation or pass ProcessedExamples."
            )

        if save_figs and pca_path:
            if subdir is not None:
                point_path = pj(pca_path, subdir, f"point_{point_id}")
            else:
                point_path = pj(pca_path, f"point_{point_id}")
            if not os.path.exists(point_path):
                os.makedirs(point_path)
        else:
            point_path = None

        # Plot active subgraphs
        subgraph_figs = []
        for subgraph_id in active_subgraph_ids:
            # subgraph_path = (
            #     pj(pca_path, f"point_{point_id}", f"subgraph_{subgraph_id}")
            #     if save_figs and pca_path
            #     else None
            # )
            if save_figs and pca_path:
                if point_path is None:
                    raise ValueError("point_path is None")
                if subdir is not None:
                    subgraph_path = pj(point_path, f"subgraph_{subgraph_id}")
                else:
                    subgraph_path = pj(point_path, f"subgraph_{subgraph_id}")
                subgraph, subgraph_df = generate_subgraph_plot_data(
                    thresholded_matrix=thresholded_matrix,
                    node_df=node_df,
                    subgraph_id=subgraph_id,
                )
                # mask activation array to only be the nodes in the subgraph
                activation_array_subgraph = activation_array[
                    subgraph_df["node_id"].astype(int).values
                ]  # type: ignore
                subgraph_fig = plot_subgraph_static_from_nx(
                    subgraph=subgraph,
                    subgraph_df=subgraph_df,
                    node_info_df=node_df,
                    output_path=subgraph_path,
                    activation_array=activation_array_subgraph,
                    save_figs=save_figs,
                    show_plot=False,
                )
                subgraph_figs.append(subgraph_fig)

        # Save or show figures
        if save_figs and pca_path:
            # if subdir is not None:
            #     point_path = pj(pca_path, subdir, f"point_{point_id}")
            # else:
            #     point_path = pj(pca_path, f"point_{point_id}")
            if point_path is None:
                raise ValueError("point_path is None")
            os.makedirs(point_path, exist_ok=True)
            bar_fig.write_image(
                pj(point_path, f"bar_plot_point_{point_id}.png"), scale=4.0
            )
            bar_fig.write_html(pj(point_path, f"bar_plot_point_{point_id}.html"))
            pie_fig.write_image(
                pj(point_path, f"pie_charts_point_{point_id}.png"), scale=4.0
            )
            pie_fig.write_html(pj(point_path, f"pie_charts_point_{point_id}.html"))

        bar_fig.show()
        pie_fig.show()
        for fig in subgraph_figs:
            if fig is not None:
                fig.show()
            else:
                print(
                    "Subgraph figure is None. Likely no latents are within a subgraph."
                )

        # Print statistics for this point
        # print_statistics(df, fs_splitting_nodes, activation_threshold)

        # Create PCA plot (PC2 vs PC3) for this point
        fig = go.Figure()

        # Add all points in grey
        fig.add_trace(
            go.Scatter(
                x=pca_df["PC2"],
                y=pca_df["PC3"],
                mode="markers",
                marker=dict(
                    color="lightgrey",
                    size=10,
                    line=dict(width=2, color="DarkSlateGrey"),
                ),
                name="Other points",
                hoverinfo="none",
            )
        )

        # Add the specific point in color
        fig.add_trace(
            go.Scatter(
                x=[pca_df.loc[point_id, "PC2"]],
                y=[pca_df.loc[point_id, "PC3"]],
                mode="markers",
                marker=dict(
                    color=colors[i],
                    size=15,
                    symbol="star",
                    line=dict(width=2, color="DarkSlateGrey"),
                ),
                name=f"Point {point_id}",
                text=pca_df.loc[point_id, "context"],
                hoverinfo="text",
            )
        )

        fig.update_layout(
            title=f"'{context}' - {point_id}",
            xaxis_title="PC2",
            yaxis_title="PC3",
            width=800,
            height=800,
            showlegend=False,
            # template = 'plotly_white'
        )

        fig.update_xaxes(constrain="domain")
        fig.update_yaxes(scaleanchor="x")

        if save_figs and pca_path:
            if point_path is None:
                raise ValueError("point_path is None")
            fig.write_image(
                pj(point_path, f"pca_pc2_vs_pc3_point_{point_id}.png"), scale=4.0
            )
            fig.write_html(pj(point_path, f"pca_pc2_vs_pc3_point_{point_id}.html"))
        else:
            fig.show()


def analyze_specific_points_animated(
    results,
    fs_splitting_nodes,
    fs_splitting_cluster,
    activation_threshold,
    node_df,
    results_path,
    pca_df,
    point_ids,
    plot_only_fs_nodes=False,
    save_gif=False,
    gif_path="animation.gif",
):
    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("PCA Plot", "Feature Activation", "Subgraph Visualization"),
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
        horizontal_spacing=0.1,
    )

    # Calculate fixed node positions for the subgraph
    subgraph = load_subgraph(results_path, activation_threshold, fs_splitting_cluster)
    fixed_pos = nx.spring_layout(subgraph, seed=42)  # Use a fixed seed for consistency

    # Calculate global maximum activation
    global_max_activation = 0
    for point_id in point_ids:
        point_result = get_point_result(results, point_id)
        df, _ = prepare_data(point_result, fs_splitting_nodes, node_df)
        global_max_activation = max(global_max_activation, df["Activation"].max())

    # Create frames for animation
    frames = []
    for point_id in point_ids:
        frame_data, context = create_frame_data(
            results=results,
            fs_splitting_nodes=fs_splitting_nodes,
            fs_splitting_cluster=fs_splitting_cluster,
            activation_threshold=activation_threshold,
            node_df=node_df,
            results_path=results_path,
            pca_df=pca_df,
            point_id=point_id,
            plot_only_fs_nodes=plot_only_fs_nodes,
            fixed_pos=fixed_pos,
        )
        frame = go.Frame(
            data=frame_data,
            name=str(point_id),
            layout=go.Layout(title=f"Point ID: {point_id}<br>{context}"),
        )
        frames.append(frame)

    # Add traces for initial state (first point)
    initial_frame_data = frames[0].data
    for trace in initial_frame_data:
        fig.add_trace(trace)

    # Update layout
    fig.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 500, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ],
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Point ID: ",
                    "visible": True,
                    "xanchor": "right",
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [str(point_id)],
                            {
                                "frame": {"duration": 300, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 300},
                            },
                        ],
                        "label": str(point_id),
                        "method": "animate",
                    }
                    for point_id in point_ids
                ],
            }
        ],
        height=600,
        width=1500,
        title=f"Point ID: {point_ids[0]}<br>{frames[0].layout.title.text.split('<br>')[1]}",
        yaxis2=dict(
            range=[0, global_max_activation]
        ),  # Set fixed y-axis range for bar plot
    )

    # Add frames to the figure
    fig.frames = frames

    if save_gif:
        # Create a folder to save individual frames
        gif_name = os.path.splitext(os.path.basename(gif_path))[0]
        # if gif_name does not end in .gif append this
        if not gif_name.endswith(".gif"):
            gif_name += ".gif"
        gif_dir = os.path.dirname(gif_path)
        frames_folder = os.path.join(gif_dir, f"{gif_name}_gif_frames")
        print(frames_folder)
        os.makedirs(frames_folder, exist_ok=True)

        # Generate images for each frame in the animation
        gif_frames = []
        for i, frame in enumerate(fig.frames):
            # Set main traces to appropriate traces within plotly frame
            fig.update(data=frame.data)
            # Update the title with the context for this frame
            fig.update_layout(title=repr(frame.layout.title.text))
            # Generate image of current state with higher resolution
            img_bytes = fig.to_image(format="png", scale=4.0)

            # Save the frame as PNG
            frame_path = os.path.join(frames_folder, f"frame_{i:04d}.png")
            with open(frame_path, "wb") as f:
                f.write(img_bytes)

            gif_frames.append(Image.open(io.BytesIO(img_bytes)))

        # Create animated GIF
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            optimize=True,
            duration=1000,
            loop=0,
        )
        print(f"Animation saved as GIF: {gif_path}")
        print(f"Individual frames saved in folder: {frames_folder}")

    return fig


def analyze_specific_points_animated_from_thresholded(
    results: ProcessedExamples | ReprocessedResults,
    thresholded_matrix: np.ndarray,
    fs_splitting_nodes: list[int],
    fs_splitting_cluster: int,
    # activation_threshold: float,
    node_df: pd.DataFrame,
    # results_path: str,
    pca_df: pd.DataFrame,
    point_ids: list[int],
    plot_only_fs_nodes: bool = False,
    save_gif: bool = False,
    gif_path: str = "animation.gif",
    gif_filename: str = "animation.gif",
    frame_folder_name: str = "gif_frames",
):
    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("PCA Plot", "Feature Activation", "Subgraph Visualization"),
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
        horizontal_spacing=0.1,
    )

    # Calculate fixed node positions for the subgraph
    subgraph, subgraph_df = generate_subgraph_plot_data(
        thresholded_matrix, node_df, fs_splitting_cluster
    )
    fixed_pos = nx.spring_layout(
        subgraph, seed=42, k=0.5
    )  # Use a fixed seed for consistency

    # Calculate global maximum activation
    global_max_activation = 0
    for point_id in point_ids:
        point_result = get_point_result(results, point_id)
        df, _ = prepare_data(point_result, fs_splitting_nodes, node_df)
        global_max_activation = max(global_max_activation, df["Activation"].max())

    # Create frames for animation
    frames = []
    for point_id in point_ids:
        frame_data, context = create_frame_data_from_thresholded(
            results=results,
            thresholded_matrix=thresholded_matrix,
            fs_splitting_nodes=fs_splitting_nodes,
            fs_splitting_cluster=fs_splitting_cluster,
            # activation_threshold=activation_threshold,
            node_df=node_df,
            # results_path=results_path,
            pca_df=pca_df,
            point_id=point_id,
            plot_only_fs_nodes=plot_only_fs_nodes,
            fixed_pos=fixed_pos,
        )
        # context = "TESTTESTTEST"
        frame = go.Frame(
            data=frame_data,
            name=str(point_id),
            layout=go.Layout(
                title=f"Point ID: {point_id}<br>{sanitise_context(context)}"
            ),
        )
        frames.append(frame)

    # Add traces for initial state (first point)
    initial_frame_data = frames[0].data
    for trace in initial_frame_data:
        fig.add_trace(trace)
    print(frames[0].layout.title.text)
    # Update layout
    fig.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 500, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ],
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Point ID: ",
                    "visible": True,
                    "xanchor": "right",
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [str(point_id)],
                            {
                                "frame": {"duration": 300, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 300},
                            },
                        ],
                        "label": str(point_id),
                        "method": "animate",
                    }
                    for point_id in point_ids
                ],
            }
        ],
        height=600,
        width=1500,
        # title=f"Point ID: {point_ids[0]}<br>{frames[0].layout.title.text.split('<br>')[1]}",
        yaxis2=dict(
            range=[0, global_max_activation]
        ),  # Set fixed y-axis range for bar plot
    )

    # Add frames to the figure
    fig.frames = frames

    if save_gif:
        if not gif_filename.endswith(".gif"):
            gif_filename += ".gif"
        print(gif_filename)
        # Create a folder to save individual frames
        frames_folder = os.path.join(gif_path, frame_folder_name)
        print(frames_folder)
        os.makedirs(frames_folder, exist_ok=True)

        # Generate images for each frame in the animation
        gif_frames = []
        for i, frame in enumerate(fig.frames):
            # Set main traces to appropriate traces within plotly frame
            fig.update(data=frame.data)
            # Update the title with the context for this frame
            fig.update_layout(title=repr(frame.layout.title.text))
            # Generate image of current state with higher resolution
            img_bytes = fig.to_image(format="png", scale=4.0)

            # Save the frame as PNG
            frame_path = os.path.join(frames_folder, f"frame_{i:04d}.png")
            with open(frame_path, "wb") as f:
                f.write(img_bytes)

            gif_frames.append(Image.open(io.BytesIO(img_bytes)))

        # Create animated GIF
        gif_frames[0].save(
            os.path.join(gif_path, gif_filename),
            save_all=True,
            append_images=gif_frames[1:],
            optimize=True,
            duration=1000,
            loop=0,
        )
        print(f"Animation saved as GIF: {gif_path}")
        print(f"Individual frames saved in folder: {frames_folder}")

    return fig


def create_frame_data(
    results: ProcessedExamples,
    fs_splitting_nodes: list[int],
    fs_splitting_cluster: int,
    activation_threshold: float,
    node_df: pd.DataFrame,
    results_path: str,
    pca_df: pd.DataFrame,
    point_id: int,
    plot_only_fs_nodes: bool = False,
    fixed_pos: Mapping[int, tuple[float, float]] | None = None,
):
    frame_data = []

    # PCA Plot
    pca_trace = go.Scatter(
        x=pca_df["PC2"],
        y=pca_df["PC3"],
        mode="markers",
        marker=dict(
            color=["red" if idx == point_id else "lightgrey" for idx in pca_df.index],
            size=[15 if idx == point_id else 5 for idx in pca_df.index],
        ),
        # text=[
        #     context if idx == point_id else None
        #     for idx, context in zip(pca_df.index, pca_df["context"])
        # ],
        # hoverinfo="text",
        showlegend=False,
        xaxis="x",
        yaxis="y",
    )
    frame_data.append(pca_trace)

    # Bar Plot
    point_result = get_point_result(results, point_id)

    if isinstance(point_result, ReprocessedResults):
        raise ValueError(
            "Passed ReprocessedResults, but cannot plot other subgraphs from Streamlit data. "
            "Please set plot_without_other_subgraphs=True when loading data from generation or pass ProcessedExamples."
        )

    df, context = prepare_data(point_result, fs_splitting_nodes, node_df)

    # Add missing fs_splitting_nodes with activity 0
    missing_nodes = set(fs_splitting_nodes) - set(df["Feature Index"])
    if missing_nodes:
        missing_df = pd.DataFrame(
            {
                "Feature Index": list(missing_nodes),
                "Activation": [0] * len(missing_nodes),
                "subgraph_id": [None] * len(missing_nodes),
                "subgraph_size": [None] * len(missing_nodes),
            }
        )
        df = pd.concat([df, missing_df], ignore_index=True)

    if plot_only_fs_nodes:
        df["Feature Index"] = df["Feature Index"].astype(int)
        df = df[df["Feature Index"].isin(fs_splitting_nodes)]
        df = df.sort_values("Feature Index")  # type: ignore
        df["Feature Index"] = df["Feature Index"].astype(str)
    else:  # Sort by activation
        df = df.sort_values("Activation", ascending=False)

    bar_trace = go.Bar(
        x=df["Feature Index"].astype(str),
        y=df["Activation"],
        marker_color=[
            "red" if idx == fs_splitting_cluster else "blue"
            for idx in df["subgraph_id"]
        ],
        showlegend=False,
        xaxis="x2",
        yaxis="y2",
    )
    frame_data.append(bar_trace)

    # Subgraph Visualization
    subgraph = load_subgraph(results_path, activation_threshold, fs_splitting_cluster)
    activation_array = point_result.all_feature_acts.flatten().cpu().numpy()

    edge_trace, node_trace = create_subgraph_traces(
        subgraph, activation_array, fixed_pos
    )
    frame_data.extend([edge_trace, node_trace])

    return frame_data, context


def escape_latex(string: str) -> str:
    string = string.replace("$", "\\$")
    return string


def remove_markdown(string: str) -> str:
    # Remove markdown formatting
    # Bold and italic
    string = re.sub(r"\*\*.*?\*\*", lambda m: m.group(0).replace("**", ""), string)
    string = re.sub(r"\*.*?\*", lambda m: m.group(0).replace("*", ""), string)
    string = re.sub(r"__.*?__", lambda m: m.group(0).replace("__", ""), string)
    string = re.sub(r"_.*?_", lambda m: m.group(0).replace("_", ""), string)

    # Headers
    string = re.sub(r"#{1,6}\s", "", string)

    # Links
    string = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", string)

    # Backticks for code
    string = re.sub(r"`([^`]+)`", r"\1", string)

    return string


def sanitise_context(context: str) -> str:
    """Sanitize context string by escaping special characters, unless already quoted."""
    # context = escape(context, quote=True)
    # context = remove_markdown(context)
    # context = escape_latex(context)
    if context.startswith('"') and context.endswith('"'):
        return context
    return repr(str(context))[1:-1]


def create_frame_data_from_thresholded(
    results: ProcessedExamples | ReprocessedResults,
    thresholded_matrix: np.ndarray,
    fs_splitting_nodes: list[int],
    fs_splitting_cluster: int,
    # activation_threshold: float,
    node_df: pd.DataFrame,
    # results_path: str,
    pca_df: pd.DataFrame,
    point_id: int,
    plot_only_fs_nodes: bool = False,
    fixed_pos: Mapping[int, tuple[float, float]] | None = None,
):
    frame_data = []

    # Convert the context to a string
    # Use repr() to escape special characters as plaintext (e.g., \n becomes \\n)
    # Remove the quotes that repr() adds by slicing [1:-1]
    # title = [
    #         sanitise_context(context) if idx == point_id else None
    #         for idx, context in zip(pca_df.index, pca_df["context"])
    #     ]

    # PCA Plot
    pca_trace = go.Scatter(
        x=pca_df["PC2"],
        y=pca_df["PC3"],
        mode="markers",
        marker=dict(
            color=["red" if idx == point_id else "lightgrey" for idx in pca_df.index],
            size=[15 if idx == point_id else 5 for idx in pca_df.index],
        ),
        # text=title,
        # hoverinfo="text",
        showlegend=False,
        xaxis="x",
        yaxis="y",
    )
    frame_data.append(pca_trace)

    # Bar Plot
    point_result = get_point_result(results, point_id)

    df, context = prepare_data(point_result, fs_splitting_nodes, node_df)
    # context = sanitise_context(context)

    # Add missing fs_splitting_nodes with activity 0
    missing_nodes = set(fs_splitting_nodes) - set(df["Feature Index"])
    if missing_nodes:
        missing_df = pd.DataFrame(
            {
                "Feature Index": list(missing_nodes),
                "Activation": [0] * len(missing_nodes),
                "subgraph_id": [None] * len(missing_nodes),
                "subgraph_size": [None] * len(missing_nodes),
            }
        )
        df = pd.concat([df, missing_df], ignore_index=True)

    if plot_only_fs_nodes:
        df["Feature Index"] = df["Feature Index"].astype(int)
        df = df[df["Feature Index"].isin(fs_splitting_nodes)]
        df = df.sort_values("Feature Index")  # type: ignore
        df["Feature Index"] = df["Feature Index"].astype(str)
    else:  # Sort by activation
        df = df.sort_values("Activation", ascending=False)

    bar_trace = go.Bar(
        x=df["Feature Index"].astype(str),
        y=df["Activation"],
        marker_color=[
            "red" if idx == fs_splitting_cluster else "blue"
            for idx in df["subgraph_id"]
        ],
        showlegend=False,
        xaxis="x2",
        yaxis="y2",
    )
    frame_data.append(bar_trace)

    # Subgraph Visualization
    subgraph, subgraph_df = generate_subgraph_plot_data(
        thresholded_matrix, node_df, fs_splitting_cluster
    )

    # Relabel nodes with their actual node_ids from subgraph_df
    node_mapping = {i: node_id for i, node_id in enumerate(subgraph_df["node_id"])}
    subgraph = nx.relabel_nodes(subgraph, node_mapping)
    activation_array = point_result.all_graph_feature_acts.flatten().cpu().numpy()
    fixed_pos_seq = fixed_pos
    fixed_pos = (
        {node_mapping[idx]: pos for idx, pos in fixed_pos_seq.items()}
        if fixed_pos_seq
        else None
    )

    if fixed_pos is None:
        raise ValueError("fixed_pos is None")

    edge_trace, node_trace = create_subgraph_traces(
        subgraph, activation_array, fixed_pos, short_array=True
    )
    frame_data.extend([edge_trace, node_trace])

    return frame_data, context


def create_subgraph_traces(
    subgraph: nx.Graph,
    activation_array: np.ndarray,
    pos: Mapping[int, tuple[float, float]] | None,
    short_array: bool = False,
):
    if pos is None:
        raise ValueError("pos is None")

    edge_x, edge_y = [], []
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
        showlegend=False,
        xaxis="x3",
        yaxis="y3",
    )

    node_x, node_y = [], []
    for node in subgraph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    # Normalize activation values for color scaling
    if not short_array:
        node_activations = [activation_array[node] for node in subgraph.nodes()]
    else:
        node_activations = activation_array
    normalized_activations = (
        np.array(node_activations) - np.min(node_activations)  # type: ignore
    ) / (np.max(node_activations) - np.min(node_activations))  # type: ignore

    # Prepare the color map
    cmap = plt.cm.get_cmap("viridis")
    n_colors = 256

    # Get the colormap in RGB
    colormap_RGB = cmap(np.arange(cmap.N))

    # Set the color for zero values to be white
    colormap_RGB[0] = (
        1,
        1,
        1,
        1,
    )  # This line sets the first color in the colormap to white

    # Prepare custom color scale (in Plotly format)
    colorscale = [
        [i / (n_colors - 1), mcolors.rgb2hex(colormap_RGB[i])] for i in range(n_colors)
    ]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale=colorscale,
            reversescale=False,
            color=normalized_activations,
            size=15,
            colorbar=dict(
                thickness=15,
                title="Normalized Activation",
                xanchor="left",
                titleside="right",
            ),
            line_width=2,
        ),
        text=[str(node) for node in subgraph.nodes()],  # Add feature number as text
        textposition="top center",  # Position the text above the node
        textfont=dict(size=10, color="black"),  # Customize text appearance
        showlegend=False,
        xaxis="x3",
        yaxis="y3",
    )

    # node_hover_text = []
    # for node in subgraph.nodes():
    #     node_info = node_df[node_df["node_id"] == node].iloc[0]
    #     top_tokens = ast.literal_eval(node_info["top_10_tokens"])
    #     node_hover_text.append(
    #         f"Feature: {node}<br>Activation: {node_activations[node]:.4f}<br>Top token: {top_tokens[0]}"
    #     )

    # node_trace.hovertext = node_hover_text

    return edge_trace, node_trace


#### Plotting from thresholded matrices


def generate_subgraph_plot_data(
    thresholded_matrix: np.ndarray,
    node_df: pd.DataFrame,
    subgraph_id: int,
) -> tuple[nx.Graph, pd.DataFrame]:
    """
    Generate subgraph plot data from a thresholded matrix.
    thresholded_matrix: numpy array, thresholded adjacency matrix
    node_df: pandas DataFrame, node information
    subgraph_id: int, subgraph ID
    """

    if not isinstance(node_df, pd.DataFrame):
        raise TypeError("node_df must be a pandas DataFrame")

    if thresholded_matrix is None or node_df is None:
        raise TypeError("thresholded_matrix and node_df cannot be None")

    # raise error if thresholded_matrix or node_df are empty
    if thresholded_matrix.size == 0 or node_df.size == 0:
        raise ValueError("thresholded_matrix and node_df cannot be empty")

    subgraph_df = node_df[node_df["subgraph_id"] == subgraph_id]
    subgraph_df = subgraph_df[["node_id", "feature_activations"]]

    # ensure subgraph_df is dataframe
    if not isinstance(subgraph_df, pd.DataFrame):
        raise TypeError("subgraph_df must be a pandas DataFrame")

    subgraph_matrix = thresholded_matrix[subgraph_df["node_id"].tolist(), :][
        :, subgraph_df["node_id"].tolist()
    ]
    subgraph = nx.from_numpy_array(subgraph_matrix)
    return subgraph, subgraph_df


def generate_subgraph_plot_data_sparse(
    sparse_thresholded_matrix: sparse.csr_matrix,
    node_df: pd.DataFrame,
    subgraph_id: int,
):
    if not isinstance(node_df, pd.DataFrame):
        raise TypeError("node_df must be a pandas DataFrame")

    if sparse_thresholded_matrix is None or node_df is None:
        raise TypeError("sparse_thresholded_matrix and node_df cannot be None")

    # raise error if sparse_thresholded_matrix or node_df are empty
    if sparse_thresholded_matrix.size == 0 or node_df.size == 0:
        raise ValueError("sparse_thresholded_matrix and node_df cannot be empty")

    # Get nodes for this subgraph
    subgraph_df = node_df[node_df["subgraph_id"] == subgraph_id]
    subgraph_df = subgraph_df[["node_id", "feature_activations"]]
    node_indices = subgraph_df["node_id"].tolist()

    # ensure subgraph_df is dataframe
    if not isinstance(subgraph_df, pd.DataFrame):
        raise TypeError("subgraph_df must be a pandas DataFrame")

    # Extract submatrix using sparse indexing
    subgraph_matrix = sparse_thresholded_matrix[node_indices, :][:, node_indices]  # type: ignore

    # Convert to networkx graph while keeping sparsity
    subgraph = nx.from_scipy_sparse_array(subgraph_matrix)  # type: ignore[attr-defined]
    # pyright reports access error but this function is in nx for the version we are using

    return subgraph, subgraph_df


def plot_subgraph_static_from_nx(
    subgraph: nx.Graph,
    subgraph_df: pd.DataFrame,
    node_info_df: pd.DataFrame | None = None,
    output_path: str | None = None,
    activation_array: np.ndarray | None = None,
    save_figs: bool = False,
    normalize_globally: bool = True,
    base_node_size: int = 500,
    colour_when_inactive: bool = True,
    plot_token_factors: bool = False,
    show_plot: bool = True,
) -> None:
    """
    Plot a static subgraph from a networkx graph, as provided by generate_subgraph_plot_data.
    subgraph: networkx graph
    subgraph_df: pandas DataFrame generated by `generate_subgraph_plot_data`
    node_info_df: pandas DataFrame | None, node information dataframe
    output_path: str | None, output path
    activation_array: numpy array | None, activation array
    save_figs: bool, save figures
    normalize_globally: bool, normalize globally
    """

    if not isinstance(subgraph, nx.Graph):
        raise TypeError("subgraph must be a networkx graph")

    if not isinstance(subgraph_df, pd.DataFrame):
        raise TypeError("subgraph_df must be a pandas DataFrame")

    if not {"node_id", "feature_activations"}.issubset(subgraph_df.columns):
        raise ValueError(
            "subgraph_df must contain the columns 'node_id' and 'feature_activations'"
        )

    # Create a new figure
    plt.figure(figsize=(7, 7))

    # Create a layout for our nodes - matched to interactive version
    pos = nx.spring_layout(subgraph, seed=42, k=0.5)  # Fixed seed and k value to match

    # Scale positions to match interactive version's scale
    pos = {node: (coord[0] * 500, coord[1] * 500) for node, coord in pos.items()}

    # Use provided activation_array if available, otherwise use subgraph_df
    if activation_array is None:
        if colour_when_inactive:
            activation_array = np.array(subgraph_df["feature_activations"].values)
        else:
            activation_array = np.zeros(len(list(subgraph.nodes())))

    # Extract activations for nodes in this subgraph
    subgraph_activations = [
        activation_array[i] for i in range(len(list(subgraph.nodes())))
    ]

    # Calculate node sizes based on feature activations
    feature_acts = np.array(subgraph_df["feature_activations"].values)
    min_act = feature_acts.min()
    max_act = feature_acts.max()
    act_range = max_act - min_act
    if act_range != 0:
        normalized_sizes = (feature_acts - min_act) / act_range
        node_sizes = base_node_size + (normalized_sizes * base_node_size)
    else:
        node_sizes = [base_node_size] * len(feature_acts)

    # Determine normalization range for colors
    if normalize_globally:
        min_activation = min(activation_array)
        max_activation = max(activation_array)
    else:
        min_activation = min(subgraph_activations)
        max_activation = max(subgraph_activations)

    activation_range = max_activation - min_activation

    # Prepare node labels and colors
    labels = {}
    node_colors = []

    for i, node in enumerate(subgraph.nodes()):
        node_id = subgraph_df["node_id"].iloc[i]

        # If node_info_df is provided, use top tokens, otherwise just use node_id
        if node_info_df is not None:
            node_info = node_info_df[node_info_df["node_id"] == node_id].iloc[0]
            if plot_token_factors:
                top_tokens = ast.literal_eval(node_info["top_10_tokens"])
                top_token = top_tokens[0]
                labels[node] = f"ID: {node_id}\n{top_token}"
            else:
                labels[node] = f"ID: {node_id}"
        else:
            labels[node] = f"ID: {node_id}"

        # Set fill to white if activation is 0, otherwise use the color map
        if subgraph_activations[i] == 0:
            node_colors.append("white")
        else:
            if activation_range != 0:
                normalized_activation = (
                    subgraph_activations[i] - min_activation
                ) / activation_range
            else:
                normalized_activation = 0.5
            node_colors.append(plt.cm.get_cmap("Blues")(normalized_activation))

    # Get edge weights
    edge_weights = [subgraph[u][v]["weight"] for u, v in subgraph.edges()]

    # Normalize edge weights for thickness
    if edge_weights:  # Only if there are edges
        max_weight = max(edge_weights)  # type: ignore
        min_weight = min(edge_weights)  # type: ignore
        normalized_weights = [
            (w - min_weight) / (max_weight - min_weight) for w in edge_weights
        ]
        edge_thickness = [0.5 + 4.5 * w for w in normalized_weights]
    else:
        edge_thickness = []

    # Draw the graph with adjusted parameters
    nx.draw(
        subgraph,
        pos,
        with_labels=False,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="black",
        linewidths=2,  # Changed to match interactive version
        edge_color=(0.5, 0.5, 0.5, 0.75),
        width=edge_thickness,
        arrows=True,
    )

    # Adjust label positions with scaled coordinates
    label_pos = {
        k: (v[0], v[1] - 75) for k, v in pos.items()
    }  # Adjusted offset for scale
    nx.draw_networkx_labels(subgraph, label_pos, labels, font_size=8)

    plt.axis("off")
    plt.tight_layout()

    if save_figs:
        plt.savefig(f"{output_path}.png", format="png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{output_path}.pdf", format="pdf", dpi=300, bbox_inches="tight")
        plt.savefig(f"{output_path}.svg", format="svg", dpi=300, bbox_inches="tight")
        plt.close()
    elif show_plot:
        plt.show()


def plot_subgraph_interactive_from_nx(
    subgraph: nx.Graph,
    subgraph_df: pd.DataFrame,
    node_info_df: pd.DataFrame | None = None,
    activation_array: np.ndarray | None = None,
    colour_when_inactive: bool = True,
    plot_token_factors: bool = False,
    height: str = "700px",
) -> tuple[Network, str]:
    """
    Plot a static subgraph from a networkx graph, as provided by generate_subgraph_plot_data.
    subgraph: networkx graph
    subgraph_df: pandas DataFrame, node information
    node_info_df: pandas DataFrame | None, node information
    activation_array: numpy array | None, activation array
    colour_when_inactive: bool, plot with colour representing feature density if there is no activation
    plot_token_factors: bool, plot token factors
    height: str, height
    width: str, width
    """

    if not isinstance(subgraph, nx.Graph):
        raise TypeError("subgraph must be a networkx graph")

    if not isinstance(subgraph_df, pd.DataFrame):
        raise TypeError("subgraph_df must be a pandas DataFrame")

    if not {"node_id", "feature_activations"}.issubset(subgraph_df.columns):
        raise ValueError(
            "subgraph_df must contain the columns 'node_id' and 'feature_activations'"
        )

    if subgraph.number_of_nodes() != len(subgraph_df):
        raise IndexError("subgraph and subgraph_df must have the same number of nodes")

    if subgraph.number_of_nodes() == 0:
        raise ValueError("subgraph must contain nodes")

    # Calculate fixed positions using networkx layout
    fixed_pos = nx.spring_layout(subgraph, seed=42, k=0.5)  # Fixed seed for consistency

    # Initialize pyvis network with notebook=True
    net = Network(
        height=height,
        width="100%",
        bgcolor="#ffffff",
        font_color=False,
        notebook=True,
    )

    # Get edge weights for scaling
    edge_weights = [subgraph[u][v]["weight"] for u, v in subgraph.edges()]

    # Rest of your existing configuration
    net.toggle_physics(False)

    # Use provided activation_array if available
    if activation_array is None:
        hover_info = False
        if colour_when_inactive:
            activation_array = np.array(subgraph_df["feature_activations"].values)
        else:
            activation_array = np.zeros(len(list(subgraph.nodes())))

    else:
        hover_info = True
    # Calculate node sizes
    feature_acts = np.array(subgraph_df["feature_activations"].values)
    min_act = feature_acts.min()
    max_act = feature_acts.max()
    act_range = max_act - min_act
    base_size = 25

    if act_range != 0:
        normalized_sizes = (feature_acts - min_act) / act_range
        node_sizes = base_size + (normalized_sizes * base_size)
    else:
        node_sizes = [base_size] * len(feature_acts)

    # Color normalization
    min_activation = min(activation_array)
    max_activation = max(activation_array)
    activation_range = max_activation - min_activation

    # Add nodes with fixed positions
    for i, node in enumerate(subgraph.nodes()):
        node_id = subgraph_df["node_id"].iloc[i]

        # Prepare label
        if node_info_df is not None:
            node_info = node_info_df[node_info_df["node_id"] == node_id].iloc[0]
            if plot_token_factors:
                top_tokens = ast.literal_eval(node_info["top_10_tokens"])
                label = f"ID: {node_id}\n{top_tokens[0]}"
            else:
                label = f"ID: {node_id}"
        else:
            label = f"ID: {node_id}"

        # Calculate color
        if activation_array[i] == 0:
            color = "#ffffff"
        else:
            if activation_range != 0:
                normalized_activation = (
                    activation_array[i] - min_activation
                ) / activation_range
                rgba = plt.cm.get_cmap("Blues")(normalized_activation)
                color = f"#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}"
            else:
                color = "#084594"

        # Get fixed position for this node
        pos = fixed_pos[node]
        x, y = pos[0] * 500, pos[1] * 500  # Scale positions for better visibility

        # Add node
        net.add_node(
            node,
            label=label,
            color=color,
            size=node_sizes[i],
            borderWidth=2,
            font=dict(size=20),
            title=f"Activation: {activation_array[i]:.2f}" if hover_info else None,
            x=float(x),  # Set fixed x position
            y=float(y),  # Set fixed y position
            physics=False,  # Disable physics for this node
        )

    # Add edges
    for u, v in subgraph.edges():
        weight = subgraph[u][v]["weight"]
        width = 1
        if edge_weights:
            width = 1 + 4 * (weight - min(edge_weights)) / (  # type: ignore
                max(edge_weights) - min(edge_weights)  # type: ignore
            )
        net.add_edge(u, v, width=width, color={"color": "rgba(128, 128, 128, 0.75)"})

    html = net.generate_html(notebook=False)
    return net, html
