import json
import os
from os.path import join as pj

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sae_lens import SAE, ActivationsStore
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer

from sae_cooccurrence.normalised_cooc_functions import get_sae_release
from sae_cooccurrence.utils.saving_loading import set_device
from sae_cooccurrence.utils.set_paths import get_git_root


def load_subgraph_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def create_subgraph_dict(node_df: pd.DataFrame) -> dict:
    subgraph_dict = {}
    for _, row in node_df.iterrows():
        subgraph_id = row["subgraph_id"]
        node_id = row["node_id"]
        if subgraph_id not in subgraph_dict:
            subgraph_dict[subgraph_id] = []
        subgraph_dict[subgraph_id].append(node_id)
    return subgraph_dict


def subgraph_activation_50percent(
    feature_acts: torch.Tensor, subgraph_nodes: list[int], activation_threshold: float
) -> torch.Tensor:
    return (feature_acts[:, subgraph_nodes] > activation_threshold).float().mean(
        dim=1
    ) >= 0.5


def subgraph_activation_any(
    feature_acts: torch.Tensor, subgraph_nodes: list[int], activation_threshold: float
) -> torch.Tensor:
    return (feature_acts[:, subgraph_nodes] > activation_threshold).any(dim=1)


def calculate_l0_sparsity(
    sae, activation_store, subgraph_dict, n_batches, device, activation_threshold
) -> tuple[float, float, float, int]:
    total_tokens = 0
    feature_l0_sum = 0
    # subgraph_l0_50percent_sum = 0
    subgraph_l0_any_sum = 0

    for _ in tqdm(range(n_batches), desc="Processing batches"):
        activations_batch = activation_store.next_batch()
        feature_acts = sae.encode(activations_batch).squeeze()

        # Count active features per token
        feature_activations = (feature_acts > activation_threshold).float()
        feature_l0_per_token = feature_activations.sum(dim=1)

        feature_l0_sum += feature_l0_per_token.sum().item()

        # Count active subgraphs per token (≥50% definition)
        # subgraph_l0_50percent_per_token = torch.zeros(
        #     feature_acts.shape[0], device=device
        # )
        # Count active subgraphs per token (any active definition)
        subgraph_l0_any_per_token = torch.zeros(feature_acts.shape[0], device=device)

        for _, nodes in subgraph_dict.items():
            # sg_activation_50percent = subgraph_activation_50percent(
            #     feature_acts, nodes, activation_threshold
            # )
            sg_activation_any = subgraph_activation_any(
                feature_acts, nodes, activation_threshold
            )

            # subgraph_l0_50percent_per_token += sg_activation_50percent.float()
            subgraph_l0_any_per_token += sg_activation_any.float()

        # subgraph_l0_50percent_sum += subgraph_l0_50percent_per_token.sum().item()
        subgraph_l0_any_sum += subgraph_l0_any_per_token.sum().item()

        total_tokens += feature_acts.shape[0]

    # Calculate means (average number of active features/subgraphs per token)
    feature_l0_mean = feature_l0_sum / total_tokens
    # subgraph_l0_50percent_mean = subgraph_l0_50percent_sum / total_tokens
    subgraph_l0_any_mean = subgraph_l0_any_sum / total_tokens

    return (
        feature_l0_mean,
        0.0,
        subgraph_l0_any_mean,
        total_tokens,
    )


def extract_sae_size(sae_id: str) -> int:
    """Extract SAE size from different SAE ID formats."""
    # For format like "layer_12/width_16k/canonical"
    if "/canonical" in sae_id:
        width_part = sae_id.split("/")[1]
        size_str = width_part.split("_")[-1]
        return int(size_str.replace("k", "000"))
    # For format like "blocks.8.hook_resid_pre_768"
    elif "hook_resid_pre_" in sae_id:
        return int(sae_id.split("_")[-1])
    # For format like "layer_12/width_16k/average_l0_22"
    elif "/average_l0_" in sae_id:
        width_part = sae_id.split("/")[1]
        size_str = width_part.split("_")[-1]
        return int(size_str.replace("k", "000"))
    else:
        raise ValueError(f"Unrecognized SAE ID format: {sae_id}")


def analyze_sae(
    model_name: str,
    sae_release_short: str,
    sae_id: str,
    n_batches: int,
    n_batches_generation: int,
    device: str,
    git_root: str,
    output_dir: str,
    activation_threshold: float,
) -> dict:
    sae_release_short_safe = sae_release_short.replace("/", "_")
    results_dir = f"results/{model_name}/{sae_release_short_safe}/{sae_id.replace('.', '_').replace('/', '_')}"

    sae_release = get_sae_release(model_name, sae_release_short)

    # Load model and SAE
    model = HookedTransformer.from_pretrained(model_name, device=device)
    sae, _, _ = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=device)

    # Set up the activations store
    activation_store = ActivationsStore.from_sae(
        model=model,
        sae=sae,
        streaming=True,
        store_batch_size_prompts=8,
        train_batch_size_tokens=4096,
        n_batches_in_buffer=4,
        device=device,
    )

    threshold_safe = str(activation_threshold).replace(".", "_")

    # Load subgraph data
    node_df = load_subgraph_data(
        pj(
            git_root,
            results_dir,
            f"n_batches_{n_batches_generation}",
            "dataframes",
            f"node_info_df_{threshold_safe}.csv",
        )
    )
    subgraph_dict = create_subgraph_dict(node_df)

    # Calculate L0 sparsity
    feature_l0, subgraph_l0_50percent, subgraph_l0_any, total_tokens = (
        calculate_l0_sparsity(
            sae,
            activation_store,
            subgraph_dict,
            n_batches,
            device,
            activation_threshold,
        )
    )

    sae_size = extract_sae_size(sae_id)
    num_subgraphs = len(subgraph_dict)

    # Extract average_l0 from sae_id if present
    if "average_l0_" in sae_id:
        average_l0 = int(sae_id.split("average_l0_")[-1])
    else:
        average_l0 = None

    # Save results
    result = {
        "sae_size": sae_size,
        "average_l0": average_l0,
        "feature_l0": feature_l0,
        "subgraph_l0_50percent": subgraph_l0_50percent,
        "subgraph_l0_any": subgraph_l0_any,
        "total_tokens": total_tokens,
        "num_features": sae_size,
        "num_subgraphs": num_subgraphs,
    }

    if average_l0 is not None:
        with open(pj(output_dir, f"l0_comparison_sae_l0_{average_l0}.json"), "w") as f:
            json.dump(result, f)
    else:
        with open(pj(output_dir, f"l0_comparison_sae_size_{sae_size}.json"), "w") as f:
            json.dump(result, f)

    return result


def plot_l0_comparison_matplotlib(
    results: list[dict],
    output_dir: str,
    x_axis_key: str = "sae_size",
    activation_threshold_safe: str = "1_5",
):
    # Use either sae_size or average_l0 for x-axis
    x_values = [r[x_axis_key] for r in results]
    x_axis_label = (
        "SAE Size (number of features)"
        if x_axis_key == "sae_size"
        else "Target Average L0"
    )

    feature_l0s = [r["feature_l0"] for r in results]
    subgraph_l0s_any = [r["subgraph_l0_any"] for r in results]

    # Normalized values
    feature_l0s_norm = [r["feature_l0"] / r["num_features"] for r in results]
    subgraph_l0s_any_norm = [r["subgraph_l0_any"] / r["num_subgraphs"] for r in results]

    # Define colors and line styles
    color_map = {
        "res-jb-feature-splitting": "blue",
        "gemma-scope-2b-pt-res-canonical": "red",
        "gemma-scope-2b-pt-res": "green",
    }

    # Determine color based on sae_release
    sae_release = results[0].get("sae_release", "default")
    color = color_map.get(sae_release, "blue")

    # Plot normalized values
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, feature_l0s_norm, "o-", color=color, label="Feature-based L0")
    plt.plot(
        x_values,
        subgraph_l0s_any_norm,
        "o--",
        color=color,
        label="Subgraph-based L0",
        alpha=0.7,
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(x_axis_label)
    plt.ylabel("Proportion of Features/Subgraphs Active")
    plt.xticks(x_values, x_values)  # Set specific x-tick locations and labels
    plt.title(
        f"Normalized L0 Sparsity (Activation Threshold = {activation_threshold_safe.replace('_', '.')})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        pj(output_dir, f"l0_comparison_normalized_{activation_threshold_safe}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Plot raw values
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, feature_l0s, "o-", color=color, label="Feature-based L0")
    plt.plot(
        x_values,
        subgraph_l0s_any,
        "o--",
        color=color,
        label="Subgraph-based L0",
        alpha=0.7,
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(x_axis_label)
    plt.ylabel("Average Active Features/Subgraphs per Token")
    plt.xticks(x_values, x_values)  # Set specific x-tick locations and labels
    plt.title(
        f"Raw L0 Sparsity (Activation Threshold = {activation_threshold_safe.replace('_', '.')})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        pj(output_dir, f"l0_comparison_raw_{activation_threshold_safe}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def load_or_generate_data(
    model_name: str,
    sae_release_short,
    sae_ids: list[str],
    n_batches: int,
    n_batches_generation: int,
    device: str,
    git_root: str,
    output_dir: str,
    activation_threshold: float,
    regenerate: bool = False,
) -> list[dict]:
    results = []
    for sae_id in tqdm(sae_ids, desc="Processing SAEs"):
        # Determine file name based on presence of average_l0
        if "average_l0_" in sae_id:
            average_l0 = int(sae_id.split("average_l0_")[-1])
            result_file = pj(output_dir, f"l0_comparison_sae_l0_{average_l0}.json")
        else:
            sae_size = extract_sae_size(sae_id)
            result_file = pj(output_dir, f"l0_comparison_sae_size_{sae_size}.json")

        if os.path.exists(result_file) and not regenerate:
            with open(result_file) as f:
                result = json.load(f)
        else:
            result = analyze_sae(
                model_name,
                sae_release_short,
                sae_id,
                n_batches,
                n_batches_generation,
                device,
                git_root,
                output_dir,
                activation_threshold,
            )

        results.append(result)
        print(
            f"SAE size: {result['sae_size']}, "
            f"Feature L0: {result['feature_l0']:.4f}, "
            f"Subgraph L0 (≥50%): {result['subgraph_l0_50percent']:.4f}, "
            f"Subgraph L0 (any): {result['subgraph_l0_any']:.4f}"
        )

    return results


def main():
    # Setup
    device = set_device()
    git_root = get_git_root()

    # Configuration for multiple models
    model_configs = [
        {
            "model_name": "gpt2-small",
            "sae_release_short": "res-jb-feature-splitting",
            "sae_ids": [
                "blocks.8.hook_resid_pre_768",
                "blocks.8.hook_resid_pre_1536",
                "blocks.8.hook_resid_pre_3072",
                "blocks.8.hook_resid_pre_6144",
                "blocks.8.hook_resid_pre_12288",
                "blocks.8.hook_resid_pre_24576",
                "blocks.8.hook_resid_pre_49152",
                "blocks.8.hook_resid_pre_98304",
            ],
            "n_batches_generation": 500,
        },
        {
            "model_name": "gemma-2-2b",
            "sae_release_short": "gemma-scope-2b-pt-res-canonical",
            "sae_ids": [
                "layer_12/width_16k/canonical",
                "layer_12/width_32k/canonical",
                "layer_12/width_65k/canonical",
            ],
            "n_batches_generation": 100,
        },
        {
            "model_name": "gemma-2-2b",
            "sae_release_short": "gemma-scope-2b-pt-res",
            "sae_ids": [
                "layer_12/width_16k/average_l0_22",
                "layer_12/width_16k/average_l0_41",
                "layer_12/width_16k/average_l0_82",
                "layer_12/width_16k/average_l0_176",
                "layer_12/width_16k/average_l0_445",
            ],
            "n_batches_generation": 100,
        },
        # Add more model configurations as needed
    ]

    n_batches_profiling = 10
    activation_threshold = 1.5  # You can adjust this threshold
    activation_threshold_safe = str(activation_threshold).replace(".", "_")

    for config in model_configs:
        model_name = config["model_name"]
        sae_release_short = config["sae_release_short"]
        sae_ids = config["sae_ids"]
        n_batches_generation = config["n_batches_generation"]

        # Create output directory
        output_dir = pj(
            git_root,
            "results",
            "size_effects",
            model_name,
            sae_release_short,
            f"l0_of_feature_and_graph_comparison_{activation_threshold_safe}",
        )
        os.makedirs(output_dir, exist_ok=True)

        # Check if data already exists
        existing_data = [f for f in os.listdir(output_dir) if f.endswith(".json")]
        regenerate = False

        if existing_data:
            print(f"Existing data found for {model_name}.")
            regenerate_input = (
                input("Do you want to regenerate the data? (y/n): ").lower().strip()
            )
            regenerate = regenerate_input == "y"
            if regenerate:
                print("Regenerating data...")
                for file in existing_data:
                    os.remove(pj(output_dir, file))
            else:
                print("Using existing data...")
        else:
            print(f"No existing data found for {model_name}. Generating new data...")
            regenerate = True

        # Determine x-axis based on release
        x_axis_key = (
            "average_l0" if sae_release_short == "gemma-scope-2b-pt-res" else "sae_size"
        )

        results = load_or_generate_data(
            model_name,
            sae_release_short,
            sae_ids,
            n_batches_profiling,
            n_batches_generation,
            device,
            git_root,
            output_dir,
            activation_threshold,
            regenerate,
        )
        for result in results:
            result["model_name"] = model_name

        # Sort by appropriate key
        results.sort(key=lambda x: x[x_axis_key])
        plot_l0_comparison_matplotlib(
            results, output_dir, x_axis_key, activation_threshold_safe
        )

        print(
            f"Analysis complete for {model_name}. Results plotted in {output_dir}/l0_comparison_combined_{activation_threshold_safe}.png"
        )


if __name__ == "__main__":
    main()
