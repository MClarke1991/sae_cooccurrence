# %%
import json
import os
from os.path import join as pj

import pandas as pd
import torch
from sae_lens import SAE, ActivationsStore
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer

from sae_cooccurrence.normalised_cooc_functions import neat_sae_id
from sae_cooccurrence.utils.saving_loading import set_device
from sae_cooccurrence.utils.set_paths import get_git_root

# %%
# Setup
device = set_device()
git_root = get_git_root()

# Configuration
gpt_model_name = "gpt2-small"
gpt_sae_release_short = "res-jb-feature-splitting"
gpt_sae_ids = [
    "blocks.8.hook_resid_pre_768",
    "blocks.8.hook_resid_pre_1536",
    "blocks.8.hook_resid_pre_3072",
    "blocks.8.hook_resid_pre_6144",
    "blocks.8.hook_resid_pre_12288",
    "blocks.8.hook_resid_pre_24576",
]

gemma_width_model_name = "gemma-2-2b"
gemma_width_sae_release_short = "gemma-scope-2b-pt-res-canonical"
gemma_width_sae_ids = [
    "layer_12/width_16k/canonical",
    "layer_12/width_32k/canonical",
    "layer_12/width_65k/canonical",
    "layer_12/width_262k/canonical",
    "layer_12/width_524k/canonical",
    "layer_12/width_1m/canonical",
]

gemma_l0_model_name = "gemma-2-2b"
gemma_l0_sae_release_short = "gemma-scope-2b-pt-res"
gemma_l0_sae_ids = [
    "layer_12/width_16k/average_l0_176",
    "layer_12/width_16k/average_l0_22",
    "layer_12/width_16k/average_l0_41",
    "layer_12/width_16k/average_l0_445",
    "layer_12/width_16k/average_l0_82",
]

n_batches = 10
activation_threshold = 1.5  # You can adjust this threshold
activation_threshold_safe = str(activation_threshold).replace(".", "_")


# %%
def load_node_info_df(model_name, sae_release_short, sae_id, activation_threshold_safe):
    base_path = pj(get_git_root(), "results", model_name, sae_release_short)
    sae_id_neat = neat_sae_id(sae_id)
    file_path = pj(
        base_path,
        sae_id_neat,
        "dataframes",
        f"node_info_df_{activation_threshold_safe}.csv",
    )
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df
    else:
        print(f"Warning: File not found - {file_path}")
        return None


def load_subgraph_data(csv_path):
    return pd.read_csv(csv_path)


def create_subgraph_dict(node_df):
    subgraph_dict = {}
    for _, row in node_df.iterrows():
        subgraph_id = row["subgraph_id"]
        node_id = row["node_id"]
        if subgraph_id not in subgraph_dict:
            subgraph_dict[subgraph_id] = []
        subgraph_dict[subgraph_id].append(node_id)
    return subgraph_dict


def subgraph_activation_any(feature_acts, subgraph_nodes, activation_threshold):
    return (feature_acts[:, subgraph_nodes] > activation_threshold).any(dim=1)


def calculate_l0_sparsity(
    sae, activation_store, subgraph_dict, n_batches, device, activation_threshold
):
    total_tokens = 0
    feature_l0_sum = 0
    subgraph_l0_any_sum = 0

    for _ in tqdm(range(n_batches), desc="Processing batches"):
        activations_batch = activation_store.next_batch()
        feature_acts = sae.encode(activations_batch).squeeze()

        feature_activations = (feature_acts > activation_threshold).float()
        feature_l0_per_token = feature_activations.sum(dim=1)

        feature_l0_sum += feature_l0_per_token.sum().item()

        subgraph_l0_any_per_token = torch.zeros(feature_acts.shape[0], device=device)

        for _, nodes in subgraph_dict.items():
            sg_activation_any = subgraph_activation_any(
                feature_acts, nodes, activation_threshold
            )
            subgraph_l0_any_per_token += sg_activation_any.float()

        subgraph_l0_any_sum += subgraph_l0_any_per_token.sum().item()

        total_tokens += feature_acts.shape[0]

    feature_l0_mean = feature_l0_sum / total_tokens
    subgraph_l0_any_mean = subgraph_l0_any_sum / total_tokens

    return feature_l0_mean, subgraph_l0_any_mean, total_tokens


def analyze_sae(
    model_name,
    sae_release_short,
    sae_id,
    n_batches,
    device,
    git_root,
    output_dir,
    activation_threshold,
):
    results_dir = f"results/{model_name}/{sae_release_short}"

    model = HookedTransformer.from_pretrained(model_name, device=device)
    sae, _, _ = SAE.from_pretrained(
        release=f"{model_name}-{sae_release_short}", sae_id=sae_id, device=device
    )

    activation_store = ActivationsStore.from_sae(
        model=model,
        sae=sae,
        streaming=True,
        store_batch_size_prompts=8,
        train_batch_size_tokens=4096,
        n_batches_in_buffer=32,
        device=device,
    )

    threshold_safe = str(activation_threshold).replace(".", "_")

    sae_id_neat = neat_sae_id(sae_id)
    node_df = load_subgraph_data(
        pj(
            git_root,
            results_dir,
            sae_id_neat,
            "dataframes",
            f"node_info_df_{threshold_safe}.csv",
        )
    )
    subgraph_dict = create_subgraph_dict(node_df)

    feature_l0, subgraph_l0_any, total_tokens = calculate_l0_sparsity(
        sae, activation_store, subgraph_dict, n_batches, device, activation_threshold
    )

    sae_size = int(sae_id.split("_")[-1])
    num_subgraphs = len(subgraph_dict)

    result = {
        "sae_size": sae_size,
        "feature_l0": feature_l0,
        "subgraph_l0_any": subgraph_l0_any,
        "total_tokens": total_tokens,
        "num_features": sae_size,
        "num_subgraphs": num_subgraphs,
    }

    with open(pj(output_dir, f"l0_comparison_sae_size_{sae_size}.json"), "w") as f:
        json.dump(result, f)

    return result


def main(model_name, sae_release_short, sae_ids, output_dir):
    for sae_id in tqdm(sae_ids, desc="Processing SAEs"):
        result = analyze_sae(
            model_name,
            sae_release_short,
            sae_id,
            n_batches,
            device,
            git_root,
            output_dir,
            activation_threshold,
        )
        print(
            f"SAE size: {result['sae_size']}, "
            f"Feature L0: {result['feature_l0']:.4f}, "
            f"Subgraph L0 (any): {result['subgraph_l0_any']:.4f}"
        )

    print(f"Analysis complete. Results saved in {output_dir}")


# %%
# Create output directories
gpt_output_dir = pj(
    git_root,
    "results",
    "size_effects",
    gpt_model_name,
    gpt_sae_release_short,
    f"l0_comparison_{activation_threshold_safe}",
)
gemma_width_output_dir = pj(
    git_root,
    "results",
    "size_effects",
    gemma_width_model_name,
    gemma_width_sae_release_short,
    f"l0_comparison_{activation_threshold_safe}",
)
gemma_l0_output_dir = pj(
    git_root,
    "results",
    "size_effects",
    gemma_l0_model_name,
    gemma_l0_sae_release_short,
    f"l0_comparison_{activation_threshold_safe}",
)

os.makedirs(gpt_output_dir, exist_ok=True)
os.makedirs(gemma_width_output_dir, exist_ok=True)
os.makedirs(gemma_l0_output_dir, exist_ok=True)

# %%
main(gpt_model_name, gpt_sae_release_short, gpt_sae_ids, gpt_output_dir)
main(
    gemma_width_model_name,
    gemma_width_sae_release_short,
    gemma_width_sae_ids,
    gemma_width_output_dir,
)
main(
    gemma_l0_model_name,
    gemma_l0_sae_release_short,
    gemma_l0_sae_ids,
    gemma_l0_output_dir,
)
