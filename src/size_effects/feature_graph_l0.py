import json
import os
from os.path import join as pj

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
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
    subgraph_l0_50percent_sum = 0
    subgraph_l0_any_sum = 0

    for _ in tqdm(range(n_batches), desc="Processing batches"):
        activations_batch = activation_store.next_batch()
        feature_acts = sae.encode(activations_batch).squeeze()

        # Count active features per token
        feature_activations = (feature_acts > activation_threshold).float()
        feature_l0_per_token = feature_activations.sum(dim=1)

        feature_l0_sum += feature_l0_per_token.sum().item()

        # Count active subgraphs per token (≥50% definition)
        subgraph_l0_50percent_per_token = torch.zeros(
            feature_acts.shape[0], device=device
        )
        # Count active subgraphs per token (any active definition)
        subgraph_l0_any_per_token = torch.zeros(feature_acts.shape[0], device=device)

        for _, nodes in subgraph_dict.items():
            sg_activation_50percent = subgraph_activation_50percent(
                feature_acts, nodes, activation_threshold
            )
            sg_activation_any = subgraph_activation_any(
                feature_acts, nodes, activation_threshold
            )

            subgraph_l0_50percent_per_token += sg_activation_50percent.float()
            subgraph_l0_any_per_token += sg_activation_any.float()

        subgraph_l0_50percent_sum += subgraph_l0_50percent_per_token.sum().item()
        subgraph_l0_any_sum += subgraph_l0_any_per_token.sum().item()

        total_tokens += feature_acts.shape[0]

    # Calculate means (average number of active features/subgraphs per token)
    feature_l0_mean = feature_l0_sum / total_tokens
    subgraph_l0_50percent_mean = subgraph_l0_50percent_sum / total_tokens
    subgraph_l0_any_mean = subgraph_l0_any_sum / total_tokens

    return (
        feature_l0_mean,
        subgraph_l0_50percent_mean,
        subgraph_l0_any_mean,
        total_tokens,
    )


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
    results_dir = f"results/{model_name}/{sae_release_short}/{sae_id.replace('.', '_')}"

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
        n_batches_in_buffer=32,
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

    sae_size = int(sae_id.split("_")[-1])
    num_subgraphs = len(subgraph_dict)

    # Save results
    result = {
        "sae_size": sae_size,
        "feature_l0": feature_l0,
        "subgraph_l0_50percent": subgraph_l0_50percent,
        "subgraph_l0_any": subgraph_l0_any,
        "total_tokens": total_tokens,
        "num_features": sae_size,
        "num_subgraphs": num_subgraphs,
    }

    with open(pj(output_dir, f"l0_comparison_sae_size_{sae_size}.json"), "w") as f:
        json.dump(result, f)

    return result


def plot_l0_comparison(results: list[dict], output_dir: str):
    # Create separate figures for normalized and raw plots
    fig_norm = go.Figure()
    fig_raw = go.Figure()

    sae_sizes = [r["sae_size"] for r in results]
    feature_l0s = [r["feature_l0"] for r in results]
    # subgraph_l0s_50percent = [r['subgraph_l0_50percent'] for r in results]
    subgraph_l0s_any = [r["subgraph_l0_any"] for r in results]

    # Normalized values
    feature_l0s_norm = [r["feature_l0"] / r["num_features"] for r in results]
    # subgraph_l0s_50percent_norm = [r['subgraph_l0_50percent'] / r['num_subgraphs'] for r in results]
    subgraph_l0s_any_norm = [r["subgraph_l0_any"] / r["num_subgraphs"] for r in results]

    # Define colors
    colors = ["blue", "red", "green"]

    # Normalized plot
    fig_norm.add_trace(
        go.Scatter(
            x=sae_sizes,
            y=feature_l0s_norm,
            mode="markers+lines",
            name="Feature-based L0",
            line=dict(color=colors[0]),
        )
    )
    fig_norm.add_trace(
        go.Scatter(
            x=sae_sizes,
            y=subgraph_l0s_any_norm,
            mode="markers+lines",
            name="Subgraph-based L0 (any active)",
            line=dict(color=colors[2]),
        )
    )

    # Raw plot
    fig_raw.add_trace(
        go.Scatter(
            x=sae_sizes,
            y=feature_l0s,
            mode="markers+lines",
            name="Feature-based L0",
            line=dict(color=colors[0]),
        )
    )
    fig_raw.add_trace(
        go.Scatter(
            x=sae_sizes,
            y=subgraph_l0s_any,
            mode="markers+lines",
            name="Subgraph-based L0 (any active)",
            line=dict(color=colors[2]),
        )
    )

    # Update layout for normalized plot
    fig_norm.update_layout(
        title_text="Normalized L0 Sparsity vs SAE Size",
        xaxis_title="SAE Size",
        yaxis_title="Proportion of Features/Subgraphs Active",
        xaxis_type="log",
        yaxis_type="log",
        height=600,
        width=600,
        showlegend=True,
    )

    # Update layout for raw plot
    fig_raw.update_layout(
        title_text="Raw L0 Sparsity vs SAE Size",
        xaxis_title="SAE Size",
        yaxis_title="Average Active Features/Subgraphs per Token",
        xaxis_type="log",
        yaxis_type="log",
        height=600,
        width=600,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    # Set y-axis ranges
    y_min_norm, y_max_norm = (
        min(min(feature_l0s_norm), min(subgraph_l0s_any_norm)),
        max(max(feature_l0s_norm), max(subgraph_l0s_any_norm)),
    )
    y_min_raw, y_max_raw = (
        min(min(feature_l0s), min(subgraph_l0s_any)),
        max(max(feature_l0s), max(subgraph_l0s_any)),
    )

    fig_norm.update_yaxes(
        range=[np.log10(y_min_norm) - 0.1, np.log10(y_max_norm) + 0.1]
    )
    fig_raw.update_yaxes(range=[np.log10(y_min_raw) - 0.1, np.log10(y_max_raw) + 0.1])

    # Save separate plots
    fig_norm.write_html(pj(output_dir, "l0_comparison_normalized.html"))
    fig_norm.write_image(pj(output_dir, "l0_comparison_normalized.png"), scale=3.0)
    fig_raw.write_html(pj(output_dir, "l0_comparison_raw.html"))
    fig_raw.write_image(pj(output_dir, "l0_comparison_raw.png"), scale=3.0)

    # Create combined plot
    fig_combined = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Normalized L0 Sparsity", "Raw L0 Sparsity"),
        shared_yaxes=False,
    )

    # Add traces to combined plot
    for trace in fig_norm.data:
        fig_combined.add_trace(trace, row=1, col=1)
    for trace in fig_raw.data:
        fig_combined.add_trace(trace, row=1, col=2)

    # Update layout for combined plot
    fig_combined.update_layout(
        title_text="Comparison of Feature-based and Subgraph-based L0 Sparsity vs SAE Size",
        height=600,
        width=1200,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
    )

    # Update axes for combined plot
    fig_combined.update_xaxes(title_text="SAE Size", type="log", row=1, col=1)
    fig_combined.update_xaxes(title_text="SAE Size", type="log", row=1, col=2)
    fig_combined.update_yaxes(
        title_text="Proportion of Features/Subgraphs Active", type="log", row=1, col=1
    )
    fig_combined.update_yaxes(
        title_text="Average Active Features/Subgraphs per Token",
        type="log",
        row=1,
        col=2,
    )

    # Set y-axis ranges for combined plot
    fig_combined.update_yaxes(
        range=[np.log10(y_min_norm) - 0.1, np.log10(y_max_norm) + 0.1], row=1, col=1
    )
    fig_combined.update_yaxes(
        range=[np.log10(y_min_raw) - 0.1, np.log10(y_max_raw) + 0.1], row=1, col=2
    )

    # Save combined plot
    fig_combined.write_html(pj(output_dir, "l0_comparison_combined.html"))
    fig_combined.write_image(pj(output_dir, "l0_comparison_combined.png"), scale=3.0)


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
) -> list[dict]:
    results = []
    for sae_id in tqdm(sae_ids, desc="Processing SAEs"):
        result_file = pj(
            output_dir, f"l0_comparison_sae_size_{sae_id.split('_')[-1]}.json"
        )

        if os.path.exists(result_file):
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

        if existing_data:
            print(f"Existing data found for {model_name}.")
            regenerate = (
                input("Do you want to regenerate the data? (y/n): ").lower().strip()
            )
            if regenerate == "y":
                print("Regenerating data...")
                for file in existing_data:
                    os.remove(pj(output_dir, file))
            else:
                print("Using existing data...")
        else:
            print(f"No existing data found for {model_name}. Generating new data...")

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
        )

        results.sort(key=lambda x: x["sae_size"])  # Sort by SAE size
        plot_l0_comparison(results, output_dir)

        print(
            f"Analysis complete for {model_name}. Results plotted in {output_dir}/l0_comparison_by_sae_size.html and .png"
        )


if __name__ == "__main__":
    main()
