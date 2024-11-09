import logging
import os
import warnings
from os.path import join as pj

import numpy as np
import torch
from sae_lens import SAE, ActivationsStore
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

from sae_cooccurrence.utils.saving_loading import (
    compress_directory_to_tar,
    load_npz_files,
)
from sae_cooccurrence.utils.set_paths import get_git_root


def neat_sae_id(sae_id: str) -> str:
    return sae_id.replace(".", "_").replace("/", "_")


def setup_logging(
    results_dir: str,
    model_name: str,
    sae_release_short: str,
    sae_id_neat: str,
    context: str,
) -> None:
    log_file = pj(
        results_dir,
        f"script_log_{context}_{model_name}_{sae_release_short}_{sae_id_neat}.txt",
    )
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_special_tokens(model: HookedTransformer) -> set[int | None]:
    if model.tokenizer is None:
        raise ValueError("Model tokenizer is None")
    special_tokens = {
        model.tokenizer.bos_token_id,
        model.tokenizer.eos_token_id,
        model.tokenizer.pad_token_id,
    }
    return special_tokens


def get_sae_release(model_name: str, sae_release_short: str) -> str:
    """
    Determine the SAE release based on the model name.

    Args:
    model_name (str): The name of the model.
    sae_release_short (str): A shortened version of the SAE release.

    Returns:
    str: The full SAE release string.
    """
    if model_name == "gemma-2-2b":
        if sae_release_short == "gemma-scope-2b-pt-res-canonical":
            return "gemma-scope-2b-pt-res-canonical"
        elif sae_release_short == "gemma-scope-2b-pt-res":
            return "gemma-scope-2b-pt-res"
        else:
            raise ValueError("SAE Release Short unsupported.")
    else:
        return f"{model_name}-{sae_release_short}"


def generate_normalised_features(
    model_name: str,
    sae_release_short: str,
    sae_id: str,
    results_dir: str,
    device: str,
    tar_name: str,
    activation_thresholds: list[float | int],
    n_batches: int = 100,
    generate_jaccard: bool = True,
    generate_tar: bool = True,
    n_batches_in_buffer: int = 32,
    save: bool = True,
    remove_special_token_acts: bool = False,
) -> None | dict[str, dict[float, torch.Tensor] | dict[float, np.ndarray]]:
    """
    Generates normalised features co-occurrence matrices for a given model, SAE release, and SAE ID.

    This function sets up and runs the process of generating normalised features co-occurrence matrices for a specified model,
    SAE release, and SAE ID. It sets up the model and SAE, computes co-occurrence matrices and overall feature activations,
    and optionally calculating Jaccard matrices and compressing the results directory to a tar file for easy download from a remote.

    Parameters:
    - model_name (str): The name of the model to use for generating features.
    - sae_release_short (str): A shortened version of the SAE release.
    - sae_id (str): The unique identifier for the SAE.
    - results_dir (str): The directory where the results will be saved.
    - device (str): The device to use for computations (e.g., 'cpu' or 'cuda').
    - tar_name (str): The name of the tar file to which the results directory will be compressed.
    - activation_thresholds (list[Union[float, int]]): A list of thresholds for generating co-occurrence matrices where
    the threshold is the minimum activation value for a feature to be considered active.
    - n_batches (int, optional): The number of batches of the Activations Store to use for computing co-occurrence matrices.
    Defaults to 100.
    - generate_jaccard (bool, optional): Whether to generate Jaccard matrices. Defaults to True.
    - generate_tar (bool, optional): Whether to compress the results directory to a tar file. Defaults to True.
    - n_batches_in_buffer (int, optional): The number of batches to keep in memory for processing. Defaults to 32.
    - save (bool, optional): Whether to save the results to disk. Defaults to True.
    - remove_special_tokens_acts (bool, optional): Whether to remove the activations of special tokens from the batch. Defaults to False.
    Returns:
    - Union[None, dict[str, Union[dict[float, torch.Tensor], dict[float, np.ndarray]]]]:
    If save is False, returns a dictionary containing the total matrices and feature activations. Otherwise, returns None.
    """

    np.random.seed(1234)

    # Convert any integers to floats and ensure all elements are numbers
    activation_thresholds = [float(threshold) for threshold in activation_thresholds]

    ## Directories and paths -------
    git_root = get_git_root()
    results_path = pj(git_root, results_dir)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    ## Device -------
    # We do not want to have the computational overhead of doing inference

    ## Model ------
    # Load model and SAE

    sae_release = get_sae_release(model_name, sae_release_short)

    model = HookedTransformer.from_pretrained(model_name, device=device)
    sae, _, _ = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=device)

    # Normalise the decoder weights
    sae.W_dec.norm(dim=-1).mean()
    sae.fold_W_dec_norm()

    special_tokens = get_special_tokens(model)

    # Set up the activations store
    activation_store = ActivationsStore.from_sae(
        model=model,
        sae=sae,
        streaming=True,
        store_batch_size_prompts=8,
        train_batch_size_tokens=4096,
        n_batches_in_buffer=n_batches_in_buffer,
        device=device,
    )

    # if not load_pregen_feature_cooc:
    # Generate co-occurrence matrices -------

    if not check_all_files_exist(results_path, activation_thresholds, "total"):
        total_matrices = compute_cooccurrence_matrices(
            sae=sae,
            activation_store=activation_store,
            n_batches=n_batches,
            activation_thresholds=activation_thresholds,
            device=device,
            remove_special_tokens_acts=remove_special_token_acts,
            special_tokens=special_tokens,
        )
        logging.info("Co-occurrence matrices calculated.")
        if save:
            save_cooccurrence_matrices(
                total_matrices, "total", activation_thresholds, results_path
            )
        feature_activations = generate_feature_activations(total_matrices)
        check_inf_features_dict(feature_activations)
        if save:
            save_feature_activations(
                feature_activations, activation_thresholds, results_path
            )
        logging.info("Co-occurrence matrices saved.")
    else:
        print("Loading pre-existing total matrices for specified thresholds:")
        total_matrices = {}
        for threshold in activation_thresholds:
            threshold_str = str(threshold).replace(".", "_")
            file_name = f"feature_acts_cooc_total_threshold_{threshold_str}.npz"
            file_path = pj(results_path, file_name)

            if os.path.exists(file_path):
                with np.load(file_path) as data:
                    array_name = list(data.keys())[0]
                    total_matrices[threshold] = torch.tensor(
                        data[array_name], device=device
                    )
                print(f"Loaded matrix for threshold {threshold}")
            else:
                print(f"Warning: No file found for threshold {threshold}")

        # Check if all required thresholds are loaded
        if set(total_matrices.keys()) != set(activation_thresholds):
            missing_thresholds = set(activation_thresholds) - set(total_matrices.keys())
            raise ValueError(
                f"Mismatch in loaded matrices and activation_thresholds. Missing thresholds: {missing_thresholds}"
            )

        print(f"Existing total matrices loaded onto {device} for specified thresholds.")

    if generate_jaccard:
        if not check_all_files_exist(results_path, activation_thresholds, "jaccard"):
            jaccard_matrices = calculate_jaccard_matrices_chunked(
                total_matrices, device=device
            )
            logging.info("Jaccard matrices calculated.")
            if save:
                save_cooccurrence_matrices(
                    jaccard_matrices, "jaccard", activation_thresholds, results_path
                )
            logging.info("Jaccard matrices saved.")
            del jaccard_matrices
            torch.cuda.empty_cache()
        else:
            print("Jaccard matrices already calculated for all thresholds, skipping.")

    print("Co-occurrence matrices calculated and saved for all thresholds.")
    if generate_tar and save:
        print("Beginning compression of results directory.")
        if not os.path.exists(f"{results_path}/{tar_name}"):
            compress_directory_to_tar(results_path, tar_name)
        else:
            print("Tar file already exists, skipping compression.")
        logging.info("Results directory compressed to tar file.")

    if not save:
        feature_activations = generate_feature_activations(total_matrices)
        return {
            "total_matrices": total_matrices,
            "feature_activations": feature_activations,
        }
    else:
        return None


# Functions -------


def check_inf_features(activations: np.ndarray) -> None:
    # Count how many times features are 'inf'
    inf_mask = np.isinf(activations)
    inf_count = np.sum(inf_mask)

    if inf_count > 0:
        raise ValueError(
            f"Found {inf_count} inf values in feature activations. Are you using low precision?"
        )


def check_inf_features_dict(feature_activations: dict[float, np.ndarray]) -> None:
    """
    Check for 'inf' values in the feature activations dictionary.

    This function iterates over the feature activations dictionary and checks for 'inf' values in each activation array.
    If any 'inf' values are found, a ValueError is raised.

    This can occur when using half precision.

    Args:
    - feature_activations (dict[float, np.ndarray]): A dictionary where each key is an activation threshold and
    the value is the corresponding feature activations array.

    Returns:
    - None
    """
    for _, activations in feature_activations.items():
        check_inf_features(activations)
    return None


def apply_activation_threshold(
    feature_acts: torch.Tensor,
    threshold: float,
    internal_threshold: torch.Tensor,
) -> torch.Tensor:
    """
    Applies an activation threshold to a feature activation tensor.

    This function takes a tensor of feature activations, a threshold value, and an array of internal threshold values.
    It returns a tensor where each element is 1.0 if the corresponding element in `feature_acts` is greater than
    the sum of `threshold` and the corresponding value in `internal_threshold`, and 0.0 otherwise.

    The internal threshold allows for correction for the threshold of JumpRELU SAE such as Gemma-2-2b. The lower bound of activation for
    these SAEs is not zero, and so we want an activation threshold that accounts for this. For Gemma-2-2b this information is stored in the
    SAE object as `sae.threshold`.

    Args:
    - feature_acts (torch.Tensor): The tensor of feature activations with shape (N, sae.cfg.d_sae).
    - threshold (float): The minimum value for a feature to be considered active.
    - internal_threshold (torch.Tensor): An array of additional threshold values with shape (sae.cfg.d_sae,).

    Returns:
    - torch.Tensor: A tensor where each element is 1.0 if the corresponding feature activation is above the
    threshold, and 0.0 otherwise.
    """
    if internal_threshold.dim() != 1:
        raise ValueError("internal_threshold must be a 1D tensor")
    if feature_acts.dim() != 2 or feature_acts.size(1) != internal_threshold.size(0):
        raise ValueError(
            f"feature_acts must be a 2D tensor with shape (N, {internal_threshold.size(0)})"
        )

    # Only consider a feature active if it is greater than the threshold and the internal threshold
    feature_acts_bool = (
        feature_acts > (threshold + internal_threshold.unsqueeze(0))
    ).float()
    return feature_acts_bool


def check_if_sae_has_threshold(sae: SAE) -> bool:
    """
    Check if the SAE has a threshold attribute that is a tensor of length sae.cfg.d_sae.
    """
    if sae is None:
        raise AttributeError("SAE is None")

    return (
        hasattr(sae, "threshold")
        and isinstance(sae.threshold, torch.Tensor)
        and sae.threshold.shape == (sae.cfg.d_sae,)
    )


# If the SAE has a threshold attribute, use it, otherwise set it to zero
def get_sae_threshold(sae: SAE, device: str) -> torch.Tensor:
    sae_threshold = (
        sae.threshold
        if check_if_sae_has_threshold(sae)
        else torch.zeros(sae.cfg.d_sae, device=device)
    )
    logging.info(f"Correcting for SAE threshold: {sae_threshold}")
    return sae_threshold


def get_feature_activations_for_batch(
    activation_store: ActivationsStore,
    device: str,
    remove_special_tokens_acts: bool,
    special_tokens: set[int | None],
) -> torch.Tensor:
    """
    Get feature activations for a batch of tokens from an ActivationsStore.

    This function retrieves a batch of activations from the ActivationsStore,
    optionally removes the first token (typically the beginning-of-sequence token),
    and encodes the activations using the provided SAE (Sparse Autoencoder).

    Args:
        activation_store (ActivationsStore): The ActivationsStore object to get activations from.
        sae (SAE): The Sparse Autoencoder used to encode the activations.
        remove_special_tokens_acts (bool, optional): Whether to remove the activations of special tokens from the batch.
                                                     Defaults to False.
        special_tokens (set[int | None]): A set of special tokens to remove from the batch e.g. bos, eos, pad.

    Returns:
        torch.Tensor: Encoded feature activations, shape (batch_size * context_size, d_sae).

    Note:
        - If remove_first_token is True, the function uses get_flattened_activations_wout_first
          to retrieve activations without the first token.
        - The returned tensor is squeezed to remove any singleton dimensions.
    """
    if not remove_special_tokens_acts:
        activations_batch = activation_store.next_batch()
    else:
        activations_batch = get_batch_without_special_token_activations(
            activation_store, special_tokens, device
        )
    return activations_batch


def get_batch_without_special_token_activations(
    activations_store: ActivationsStore,
    special_tokens: set[int | None],
    device: str,
) -> torch.Tensor:
    """
    Get a batch of activations from the ActivationsStore, removing the first token of every prompt.

    Args:
    activations_store (ActivationsStore): An instance of the ActivationsStore class.

    Returns:
    torch.Tensor: A tensor of shape [train_batch_size, 1, d_in] containing activations,
                  with the first token of each prompt removed.
    """
    # Get a batch of tokens
    batch_tokens = activations_store.get_batch_tokens().to(device)

    # Get activations for these tokens
    with torch.no_grad():
        activations = activations_store.get_activations(batch_tokens).to(device)

    non_special_mask = ~torch.isin(
        batch_tokens, torch.tensor(list(special_tokens), device=device)
    )

    # Remove the first token's activation from each prompt
    # activations = activations[non_special_mask, ...]
    activations = activations[non_special_mask]

    # Note I believe that this is necessary because ActivationsStore.next_batch() gets `train_batch_size_tokens` tokens
    # whereas ActivationsStore.get_batch_tokens() gets `store_batch_size_prompts` i.e. n `prompts of context_size` tokens each.
    # However, normally the ActivationsStore is initialised such that
    # train_batch_size_tokens` = `store_batch_size_prompts` * `context_size` and ActivationsStore.get_batch_tokens() defaults to
    # `store_batch_size_prompts` i.e. the same number of tokens as next_batch()

    # Reshape to match the output of next_batch() because the batch of tokens are stored as seperate prompts whereas next_batch()
    # returns a single batch of tokens from the prompts shuffled together
    activations = activations.reshape(-1, 1, activations.shape[-1])

    # If there's any normalization applied in the original next_batch(), apply it here
    if activations_store.normalize_activations == "expected_average_only_in":
        activations = activations_store.apply_norm_scaling_factor(activations)

    # Shuffle the activations
    # activations = activations[torch.randperm(activations.shape[0])]

    # Get the correct batch size
    train_batch_size = activations_store.train_batch_size_tokens

    # Return only the required number of activations
    return activations[:train_batch_size]


def compute_cooccurrence_matrices(
    sae: SAE,
    activation_store: ActivationsStore,
    n_batches: int,
    activation_thresholds: list[float],
    device: str,
    remove_special_tokens_acts: bool,
    special_tokens: set[int | None],
) -> dict[float, torch.Tensor]:
    """
    Computes co-occurrence matrices for given activation thresholds and precision.

    This function iterates over a specified number of batches from the activation store,
    processes each batch through a given SAE model, and computes co-occurrence matrices
    for each specified activation threshold. The matrices are accumulated over all batches.

    Args:
    - sae (SAE): The SAE model to use for encoding activations.
    - sae_id (str): The identifier for the SAE model.
    - activation_store (ActivationsStore): SAE Lens ActivationsStore object.
    - n_batches (int): The number of batches of the activation store to process.
    - activation_thresholds (list[float]): A list of thresholds for which to compute co-occurrence matrices,
    where the threshold is the minimum activation value for a feature to be considered active.
    - device (str): The device on which to perform computations (e.g. 'cpu', 'cuda', 'mps').
    - remove_special_tokens_acts (bool, optional): Whether to remove the activations of special tokens from the batch. Defaults to False.
    - special_tokens (set[int | None]): A set of special tokens to remove from the batch e.g. bos, eos, pad.
    Returns:
    - dict[float, torch.Tensor]: A dictionary where each key is an activation threshold and
    the value is the accumulated co-occurrence matrix for that threshold.
    """

    sae_threshold = get_sae_threshold(sae, device)

    feature_acts_cooc_totals = {
        t: torch.zeros((sae.cfg.d_sae, sae.cfg.d_sae), dtype=torch.float, device=device)
        for t in activation_thresholds
    }
    for _ in tqdm(
        range(n_batches), desc=f"Processing {sae.cfg.neuronpedia_id}", leave=False
    ):
        activations_batch = get_feature_activations_for_batch(
            activation_store, device, remove_special_tokens_acts, special_tokens
        )
        feature_acts = sae.encode(activations_batch).squeeze()

        for threshold in activation_thresholds:
            feature_acts_bool = apply_activation_threshold(
                feature_acts, threshold, sae_threshold
            )
            feature_acts_cooc = feature_acts_bool.T @ feature_acts_bool
            feature_acts_cooc_totals[threshold] += feature_acts_cooc

    return feature_acts_cooc_totals


def generate_feature_activations(
    matrix_dict: dict[float, torch.Tensor],
) -> dict[float, np.ndarray]:
    """
    For a dictionary of co-occurrence matrices, extract the diagonal elements i.e. feature activations.
    Args:
    - matrix_dict (dict[float, torch.Tensor]): A dictionary where each key is a threshold and the value
    is a tensor representing the co-occurrence matrix for that threshold.

    Returns:
    - dict[float, np.ndarray]: A dictionary where each key is a threshold and the value is a NumPy array
    representing the feature activations for that threshold.
    """
    feature_activations: dict[float, np.ndarray] = {}
    for threshold, matrix in matrix_dict.items():
        feature_activations[threshold] = matrix.diagonal().cpu().float().numpy()
    return feature_activations


def save_feature_activations(
    feature_activations: dict[float, np.ndarray],
    activation_thresholds: list[float],
    results_path: str,
) -> None:
    # Convert any integers to floats and ensure all elements are numbers
    activation_thresholds = [float(threshold) for threshold in activation_thresholds]

    for threshold in tqdm(
        activation_thresholds, desc="Saving feature activations", leave=False
    ):
        filepath_safe_threshold = str(threshold).replace(".", "_")
        np.savez_compressed(
            f"{results_path}/feature_acts_cooc_activations_threshold_{filepath_safe_threshold}.npz",
            feature_activations[threshold],
        )


def calculate_jaccard_matrices_chunked(
    feature_acts_totals: dict[float, torch.Tensor],
    device: str,
    chunk_size: int = 10_000,
) -> dict[float, torch.Tensor]:
    """
    Calculate Jaccard similarity matrices for each threshold. Does this in chunks to avoid memory issues.

    This function iterates over each threshold in the `feature_acts_totals` dictionary, calculates
    the total occurrences for each feature, and then iterates over the matrix in chunks to calculate
    the Jaccard similarity. The Jaccard similarity is calculated as the intersection over the union
    of the occurrences. The result is stored in a dictionary with the threshold as the key and the
    Jaccard similarity matrix as the value.

    Args:
    - feature_acts_totals (dict[float, torch.Tensor]): A dictionary where each key is a threshold
      and the value is a tensor representing the total co-occurrences of features for that threshold.
    - device (str): The device to use for calculations. Can be 'cpu', 'cuda' or 'mps'.
    - chunk_size (int, optional): The size of the chunk to use for calculating Jaccard similarity.
      Defaults to 10,000.

    Returns:
    - dict[float, torch.Tensor]: A dictionary where each key is a threshold and the value is the Jaccard
      similarity matrix for that threshold.
    """
    jaccard_matrices = {}
    print(f"Device: {device}")
    print(f"Using chunking, chunk size: {chunk_size}")

    for threshold, feature_acts_cooc_total in tqdm(
        feature_acts_totals.items(), desc="Calculating Jaccard matrices", leave=False
    ):
        n = feature_acts_cooc_total.shape[0]
        jaccard_matrix = torch.zeros((n, n), dtype=torch.float32, device=device)

        # Calculate total occurrences for each feature
        total_occurrences = feature_acts_cooc_total.diagonal()

        for i in tqdm(range(0, n, chunk_size), leave=False):
            end = min(i + chunk_size, n)
            chunk = feature_acts_cooc_total[i:end, :]

            # Calculate Jaccard similarity for the chunk
            intersection = chunk
            union = (
                total_occurrences[i:end].unsqueeze(1)
                + total_occurrences.unsqueeze(0)
                - intersection
            )

            chunk_jaccard = intersection / union

            # Check for inf values
            if torch.isinf(chunk_jaccard).any():
                warnings.warn(
                    f"Infinite value(s) detected in Jaccard calculation for threshold {threshold}, chunk {i}:{end}"
                )

            chunk_jaccard = torch.nan_to_num(
                chunk_jaccard, nan=0.0, posinf=1.0, neginf=0.0
            )

            jaccard_matrix[i:end, :] = chunk_jaccard

            # Clear CUDA cache if using GPU
            if device == "cuda":
                torch.cuda.empty_cache()

        # Move the result to CPU to free GPU memory
        jaccard_matrices[threshold] = jaccard_matrix.cpu()

        # Clear CUDA cache if using GPU
        if device == "cuda":
            torch.cuda.empty_cache()

    return jaccard_matrices


def save_cooccurrence_matrices(
    matrix_dict: dict[float, torch.Tensor],
    mat_type: str,
    activation_thresholds: list[float],
    results_path: str,
) -> None:
    # Convert any integers to floats and ensure all elements are numbers
    activation_thresholds = [float(threshold) for threshold in activation_thresholds]

    for threshold in tqdm(
        activation_thresholds, desc="Saving co-occurrence matrices", leave=False
    ):
        filepath_safe_threshold = str(threshold).replace(".", "_")
        np.savez_compressed(
            f"{results_path}/feature_acts_cooc_{mat_type}_threshold_{filepath_safe_threshold}.npz",
            matrix_dict[threshold].cpu().float().numpy(),
        )


def check_all_files_exist(
    results_path: str, activation_thresholds: list[float], mat_type: str
) -> bool:
    all_files_exist = True

    for threshold in tqdm(
        activation_thresholds, desc="Checking existing files", leave=False
    ):
        filepath_safe_threshold = str(threshold).replace(".", "_")
        filename = f"{results_path}/feature_acts_cooc_{mat_type}_threshold_{filepath_safe_threshold}.npz"

        if not os.path.exists(filename):
            all_files_exist = False
            break  # We can stop checking as soon as we find one missing file

    return all_files_exist


def load_all_feature_matrices(
    directory: str,
) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray]]:
    """
    Load all feature_total and feature_prop npz files from a directory into two dictionaries.

    Args:
    directory (str): Path to the directory containing the npz files.

    Returns:
    tuple: Two dictionaries, one for total matrices and one for proportional matrices.
    """

    total_matrices = load_npz_files(directory, "feature_acts_cooc_total")
    prop_matrices = load_npz_files(directory, "feature_acts_cooc_prop")

    return total_matrices, prop_matrices


def create_results_dir(
    model_name: str, sae_release_short: str, sae_id_neat: str, n_batches: int
) -> str:
    return f"results/{model_name}/{sae_release_short}/{sae_id_neat}/n_batches_{n_batches}"
