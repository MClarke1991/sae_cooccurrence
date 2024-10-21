import logging
import os
import time
from os.path import join as pj

import toml
import torch
from tqdm.auto import tqdm

from sae_cooccurrence.normalised_cooc_functions import (
    generate_normalised_features,
    setup_logging,
)
from sae_cooccurrence.utils.saving_loading import set_device
from sae_cooccurrence.utils.set_paths import get_git_root


# Function to split activation_thresholds into batches of 4
def batch_thresholds(
    thresholds: list[float | int], batch_size: int = 4
) -> list[list[float | int]]:
    """
    This function splits a list of thresholds into batches of a specified size. This is to keep execution within memory on e.g. an A100.

    Parameters:
    thresholds (list): The list of thresholds to be split into batches.
    batch_size (int, optional): The size of each batch. Defaults to 4.

    Returns:
    list: A list of batches, where each batch is a list of thresholds.
    """
    return [
        thresholds[i : i + batch_size] for i in range(0, len(thresholds), batch_size)
    ]


def process_sae(
    sae_id: str,
    model_name: str,
    sae_release_short: str,
    activation_thresholds: list[float | int],
    n_batches: int,
    n_batches_in_buffer: int,
    device: str,
    remove_special_tokens: bool,
) -> None:
    """
    Processes a given SAE ID by generating normalised features for each batch of activation thresholds.

    Parameters:
    sae_id (str): The unique identifier for the SAE in Neuronpedia.
    model_name (str): The name of the model used for processing in Neuronpedia.
    sae_release_short (str): A shortened version of the SAE release in Neuronpedia.
    activation_thresholds (list): A list of thresholds to split into batches for processing.
    n_batches (int): The total number of batches of the Activation Store to process.
    n_batches_in_buffer (int): The number of batches of the Activation Store to keep in memory for processing.
    device (torch.device): The device to use for processing (e.g., GPU or CPU).
    remove_special_tokens (bool): Whether to remove the special tokens from the batch.
    """
    sae_id_neat = sae_id.replace(".", "_").replace("/", "_")

    results_dir = f"results/cooc/{model_name}/{sae_release_short}/{sae_id_neat}"

    os.makedirs(results_dir, exist_ok=True)

    setup_logging(
        results_dir,
        model_name,
        sae_release_short,
        sae_id_neat,
        context="normalised_features",
    )

    start_time = time.time()
    logging.info(f"Script started running for {sae_id}")
    logging.info(
        f"Variables - n_batches: {n_batches}, model_name: {model_name}, sae_release_short: {sae_release_short}, sae_id: {sae_id}"
    )
    tar_name = f"{model_name}_{sae_release_short}_{sae_id_neat}.tar.gz"

    # Split activation_thresholds into batches of 4
    threshold_batches = batch_thresholds(activation_thresholds)

    for batch_index, threshold_batch in tqdm(
        enumerate(threshold_batches), desc="Batches"
    ):
        logging.info(f"Processing threshold batch {batch_index + 1}: {threshold_batch}")
        generate_normalised_features(
            model_name=model_name,
            sae_release_short=sae_release_short,
            sae_id=sae_id,
            results_dir=results_dir,
            n_batches=n_batches,
            device=device,
            tar_name=tar_name,
            activation_thresholds=threshold_batch,
            n_batches_in_buffer=n_batches_in_buffer,
            remove_special_tokens=remove_special_tokens,
            save=True,
        )

    execution_time = time.time() - start_time
    logging.info(
        f"Script finished running for {sae_id}. Execution time: {execution_time:.2f} seconds"
    )


def main():
    torch.set_grad_enabled(False)
    device = set_device()
    git_root = get_git_root()
    config = toml.load(pj(git_root, "src", "config_gemma.toml"))

    n_batches = config["generation"]["n_batches"]
    model_name = config["generation"]["model_name"]
    sae_release_short = config["generation"]["sae_release_short"]
    sae_ids = config["generation"]["sae_ids"]
    activation_thresholds = config["generation"]["activation_thresholds"]
    n_batches_in_buffer = config["generation"]["n_batches_in_buffer"]
    remove_special_tokens = config["generation"]["remove_special_tokens"]

    for sae_id in tqdm(sae_ids, desc="Processing SAE IDs"):
        process_sae(
            sae_id=sae_id,
            model_name=model_name,
            sae_release_short=sae_release_short,
            activation_thresholds=activation_thresholds,
            n_batches=n_batches,
            n_batches_in_buffer=n_batches_in_buffer,
            device=device,
            remove_special_tokens=remove_special_tokens,
        )

    logging.info("Script finished running for all SAE IDs.")


if __name__ == "__main__":
    main()
