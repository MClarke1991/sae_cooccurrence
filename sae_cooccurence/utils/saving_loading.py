import logging
import os
import platform
import subprocess
import tarfile
from sae_lens import SAE
from transformer_lens import HookedTransformer
from typing import Dict

import numpy as np
import torch.backends.mps
import torch.cuda
from tqdm import tqdm

def load_model_and_sae(model_name: str, sae_release: str, sae_id: str, device: str) -> tuple:
    model = HookedTransformer.from_pretrained(model_name, device=device)
    sae, _, _ = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=device)
    sae.W_dec.norm(dim=-1).mean()
    sae.fold_W_dec_norm()
    return model, sae


def notify(title: str) -> None:
    if platform.system() != "Darwin":
        return None

    apple_script = f"""
    display notification "User Notification" with title "{title}"
    """

    subprocess.run(["osascript", "-e", apple_script])
    return None


# def setup_logging(results_path: str, log_name="log") -> None:
#     if not os.path.exists(results_path):
#         raise FileNotFoundError(
#             f"In logging setup, results directory {results_path} does not exist."
#         )
#     else:
#         package_version = version("PIBBSS")
#         logging.basicConfig(
#             filename=pj(results_path, log_name),
#             level=logging.INFO,
#             format="%(asctime)s - %(levelname)s - %(message)s",
#             datefmt="%Y-%m-%d %H:%M:%S",
#         )
#         logging.info(f"Package version: {package_version}")
#     return None


def log_config_variables(config: Dict[str, Dict[str, str]]) -> None:
    # Log all config variables
    logging.info("Configuration variables:")
    for section, variables in config.items():
        logging.info(f"[{section}]")
        for key, value in variables.items():
            logging.info(f"{key}: {value}")
    logging.info("End of configuration variables")
    return None


def set_device(gpu_id: int = 0) -> str:
    if torch.cuda.is_available():
        # Get the number of available GPUs
        available_gpus = torch.cuda.device_count()
        if gpu_id >= available_gpus:
            raise ValueError(
                f"Invalid gpu_id {gpu_id}, only {available_gpus} GPUs are available."
            )
        device = f"cuda:{gpu_id}"
        print(f"Using GPU {gpu_id} out of {available_gpus} available GPUs")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS")
    else:
        device = "cpu"
        print("Using CPU")
    return device


def load_npz_files(directory, file_prefix):
    """
    Load all npz files with a given prefix from a directory into a dictionary.

    Args:
    directory (str): Path to the directory containing the npz files.
    file_prefix (str): Prefix of the files to load (e.g., "feature_acts_cooc_total" or "feature_acts_cooc_prop").

    Returns:
    dict: A dictionary where keys are thresholds and values are the loaded numpy arrays.
    """

    loaded_data = {}
    file_list = [
        filename
        for filename in os.listdir(directory)
        if filename.startswith(file_prefix) and filename.endswith(".npz")
    ]

    if not file_list:
        # Try looking in a subdirectory
        subdirectory = os.path.join(directory, f"raw_cooc_{os.path.basename(directory)}")
        if os.path.exists(subdirectory):
            directory = subdirectory
            file_list = [
                filename
                for filename in os.listdir(directory)
                if filename.startswith(file_prefix) and filename.endswith(".npz")
            ]

    if not file_list:
        raise ValueError(
            f"No files found with prefix {file_prefix} in directory {directory} or its subdirectory"
        )

    with tqdm(total=len(file_list), desc="Loading npz files") as pbar:
        for filename in file_list:
            full_path = os.path.join(directory, filename)
            # Extract threshold from filename
            assert os.path.exists(full_path), f"Path does not exist {full_path}"
            threshold_str = filename.split("threshold_")[-1].split(".")[0]
            threshold = float(threshold_str.replace("_", "."))
            # Load the npz file
            with np.load(full_path) as data:
                # Assume there's only one array in the npz file
                array_name = list(data.keys())[0]
                loaded_data[threshold] = data[array_name]
            pbar.update(1)

    return loaded_data


def compress_directory_to_tar(path_to_dir, tar_gz_name):
    # Change directory to the specified path
    os.chdir(path_to_dir)

    # Create a tar.gz file
    with tarfile.open(tar_gz_name, "w:gz") as tar:
        for root, dirs, files in os.walk("."):
            for file in files:
                tar.add(os.path.join(root, file))
