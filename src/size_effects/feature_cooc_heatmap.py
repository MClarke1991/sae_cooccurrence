import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sae_lens import SAE, ActivationsStore
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer

from sae_cooccurrence.utils.set_paths import get_git_root


def plot_clustered_heatmap(matrix, save_path):
    """Plot and save a clustered heatmap of the feature occurrences matrix."""
    plt.figure(figsize=(20, 16))
    sns.set_theme(font_scale=0.8)

    # Create a clustered heatmap
    _ = sns.clustermap(
        matrix / matrix.max(),  # Normalize the matrix
        cmap="viridis",
        xticklabels=[],
        yticklabels=[],
        figsize=(20, 16),
        dendrogram_ratio=0.1,
        cbar_pos=(0.02, 0.8, 0.05, 0.18),
    )

    plt.title("Feature Occurrences Matrix (Clustered)", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved to {save_path}")


def calculate_feature_occurrences(sae, activation_store, n_batches, device):
    """Calculate feature co-occurrences matrix from model activations."""
    feature_occurrences = torch.zeros((sae.cfg.d_sae, sae.cfg.d_sae), device=device)

    for _ in tqdm(range(n_batches), desc="Processing batches"):
        # Get next batch of activations and encode them
        activations_batch = activation_store.next_batch()
        feature_acts = sae.encode(activations_batch).squeeze()

        # Calculate feature occurrences for this batch
        feature_activations = (feature_acts > 0).float()
        batch_feature_occurrences = torch.matmul(
            feature_activations.T, feature_activations
        )
        feature_occurrences += batch_feature_occurrences

    return feature_occurrences


def main():
    # Configuration
    device = str(
        torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )
    model_name = "gpt2-small"
    sae_release_short = "res-jb-feature-splitting"
    sae_id = "blocks.8.hook_resid_pre_768"
    n_batches = 10
    out_dir = os.path.join(
        get_git_root(),
        "results",
        "size_effects",
        model_name,
        "feature_occur_heatmap",
    )
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "feature_occurrences_matrix_clustered.png")

    # Load model and SAE
    print("Loading model and SAE...")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    sae, _, _ = SAE.from_pretrained(
        release=f"{model_name}-{sae_release_short}", sae_id=sae_id, device=device
    )

    # Set up activation store
    print("Setting up activation store...")
    activation_store = ActivationsStore.from_sae(
        model=model,
        sae=sae,
        streaming=True,
        store_batch_size_prompts=8,
        train_batch_size_tokens=4096,
        n_batches_in_buffer=32,
        device=device,
    )

    # Calculate feature occurrences
    print("Calculating feature occurrences...")
    feature_occurrences = calculate_feature_occurrences(
        sae, activation_store, n_batches, device
    )

    # Plot and save the heatmap
    print("Creating and saving heatmap...")
    plot_clustered_heatmap(feature_occurrences.cpu().numpy(), save_path)


if __name__ == "__main__":
    main()
