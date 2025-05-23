import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import toml
from adjustText import adjust_text
from matplotlib.patches import Rectangle

from sae_cooccurrence.utils.set_paths import get_git_root


def plot_feature_cooccurrence(
    model_name: str,
    sae_release: str,
    sae_id: str,
    # layer_idx: int,
    n_examples: int,
    highlight_indices: list[int] | None = None,
    top_k: int = 10,
):
    """Plot cooccurrence matrix for selected SAE features.

    Args:
        model_name: Name of the model
        sae_release: SAE release name
        sae_id: SAE ID
        layer_idx: Layer index
        n_examples: Number of examples used in linear probe
        highlight_indices: Optional list of feature indices to highlight
    """
    # Prepare paths
    sae_id_safe = sae_id.replace("/", "_").replace(".", "_")

    # Load top SAE features from linear probe results
    top_sae_indices_path = os.path.join(
        get_git_root(),
        "results",
        "linear_probes",
        model_name,
        sae_release,
        sae_id_safe,
        f"n_examples_{n_examples}",
        f"top_similar_sae_features_n_examples_{n_examples}.csv",
    )

    feature_indices_df = pd.read_csv(top_sae_indices_path)
    top_sae_indices = feature_indices_df["Feature Index"].tolist()
    top_sae_indices = top_sae_indices[:top_k]

    # Load cooccurrence matrix
    cooc_path = os.path.join(
        get_git_root(),
        "results",
        model_name,
        sae_release,
        f"{sae_id_safe}",
        "n_batches_100",
        "feature_acts_cooc_total_threshold_1_5.npz",
    )

    jaccard_path = os.path.join(
        get_git_root(),
        "results",
        model_name,
        sae_release,
        f"{sae_id_safe}",
        "n_batches_100",
        "feature_acts_cooc_jaccard_threshold_1_5.npz",
    )

    data = np.load(cooc_path)
    matrix = data["arr_0"]

    # Extract submatrix for top features
    submatrix = matrix[top_sae_indices][:, top_sae_indices]

    # Create mask for lower triangle and diagonal
    mask = np.tril(np.ones_like(submatrix))

    # Create plot
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        submatrix,
        cmap="viridis",
        xticklabels=[str(idx) for idx in top_sae_indices],
        yticklabels=[str(idx) for idx in top_sae_indices],
        annot=True,
        fmt=".2f",
        square=True,
        mask=mask,
    )

    # Add rectangles around highlighted features if specified
    if highlight_indices:
        highlight_positions = [top_sae_indices.index(idx) for idx in highlight_indices]
        for i in highlight_positions:
            for j in highlight_positions:
                if j > i:  # Changed from >= to > to exclude diagonal
                    ax.add_patch(
                        Rectangle((j, i), 1, 1, fill=False, edgecolor="red", lw=2)
                    )

    plt.title(
        f"Feature Co-occurrence Matrix for Top {len(top_sae_indices)} Features\n"
        f"Red boxes highlight selected features"
    )

    # Save plot
    out_dir = os.path.join(
        get_git_root(),
        "results",
        "linear_probes",
        model_name,
        sae_release,
        sae_id_safe,
        f"n_examples_{n_examples}",
    )
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(
        os.path.join(out_dir, "feature_cooccurrence.png"), bbox_inches="tight", dpi=300
    )
    plt.close()

    # Load Jaccard similarity matrix
    data_jaccard = np.load(jaccard_path)
    jaccard_matrix = data_jaccard["arr_0"]

    # Extract submatrix for top features
    jaccard_submatrix = jaccard_matrix[top_sae_indices][:, top_sae_indices]

    # Create mask for lower triangle and diagonal
    jaccard_mask = np.tril(np.ones_like(jaccard_submatrix))

    # Create Jaccard similarity plot
    plt.figure(figsize=(12, 10))
    ax_jaccard = sns.heatmap(
        jaccard_submatrix,
        cmap="viridis",
        xticklabels=[str(idx) for idx in top_sae_indices],
        yticklabels=[str(idx) for idx in top_sae_indices],
        annot=True,
        fmt=".2f",
        square=True,
        mask=jaccard_mask,
    )

    # Add rectangles around highlighted features if specified
    if highlight_indices:
        highlight_positions = [top_sae_indices.index(idx) for idx in highlight_indices]
        for i in highlight_positions:
            for j in highlight_positions:
                if j > i:
                    ax_jaccard.add_patch(
                        Rectangle((j, i), 1, 1, fill=False, edgecolor="red", lw=2)
                    )

    plt.title(
        f"Jaccard Similarity Matrix for Top {len(top_sae_indices)} Features\n"
        f"Red boxes highlight selected features"
    )

    # Save Jaccard plot
    plt.savefig(
        os.path.join(out_dir, "jaccard_similarity.png"), bbox_inches="tight", dpi=300
    )
    plt.close()

    # Create correlation plots
    cosine_sims = feature_indices_df.set_index("Feature Index")["Cosine Similarity"]

    # Define specific pairs to highlight
    highlight_pairs = {
        (1469, 8129),
        (1469, 6449),
        (6449, 8129),
        (6449, 13989),
        (8129, 13989),
    }

    # Get all pairs of features and their correlations
    pairs = []
    for i, idx1 in enumerate(top_sae_indices):
        for j, idx2 in enumerate(top_sae_indices[i + 1 :], i + 1):
            # Ensure pairs are ordered (smaller index first)
            pair = tuple(sorted([idx1, idx2]))
            pairs.append(
                {
                    "idx1": idx1,
                    "idx2": idx2,
                    "cooc": submatrix[i, j],
                    "jaccard": jaccard_submatrix[i, j],
                    "cosine_avg": (cosine_sims[idx1] + cosine_sims[idx2]) / 2,
                    "highlighted": pair in highlight_pairs,
                }
            )

    pairs_df = pd.DataFrame(pairs)

    # Get top 5 features by cosine similarity
    top_5_features = feature_indices_df["Feature Index"].head(5).tolist()

    # Plot cosine similarity vs cooccurrence
    plt.figure(figsize=(8, 8))
    _ = plt.scatter(
        pairs_df[~pairs_df["highlighted"]]["cosine_avg"],
        pairs_df[~pairs_df["highlighted"]]["cooc"],
        alpha=0.5,
        label="Regular pairs",
    )
    if highlight_indices:
        _ = plt.scatter(
            pairs_df[pairs_df["highlighted"]]["cosine_avg"],
            pairs_df[pairs_df["highlighted"]]["cooc"],
            color="red",
            alpha=0.7,
            label="Pairs in subgraph",
        )

    # Collect texts for adjustText
    texts = []
    for _, row in pairs_df.iterrows():
        if row["idx1"] in top_5_features and row["idx2"] in top_5_features:
            texts.append(
                plt.text(
                    float(row["cosine_avg"]),
                    float(row["cooc"]),
                    f"{row['idx1']},{row['idx2']}",
                    fontsize=8,
                )
            )

    # Adjust text positions to prevent overlap
    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        expand_points=(1.5, 1.5),
    )

    plt.xlabel("Average Cosine Similarity")
    plt.ylabel("Co-occurrence")
    plt.title("Cosine Similarity vs Co-occurrence")
    plt.legend()
    plt.savefig(
        os.path.join(out_dir, "cosine_vs_cooccurrence.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    # Plot cosine similarity vs Jaccard
    plt.figure(figsize=(8, 8))
    plt.scatter(
        pairs_df[~pairs_df["highlighted"]]["cosine_avg"],
        pairs_df[~pairs_df["highlighted"]]["jaccard"],
        alpha=0.5,
        label="Regular pairs",
    )
    if highlight_indices:
        plt.scatter(
            pairs_df[pairs_df["highlighted"]]["cosine_avg"],
            pairs_df[pairs_df["highlighted"]]["jaccard"],
            color="red",
            alpha=0.7,
            label="Pairs in subgraph",
        )

    # Collect texts for adjustText
    texts = []
    for _, row in pairs_df.iterrows():
        if row["idx1"] in top_5_features and row["idx2"] in top_5_features:
            texts.append(
                plt.text(
                    float(row["cosine_avg"]),
                    float(row["jaccard"]),
                    f"{row['idx1']},{row['idx2']}",
                    fontsize=8,
                )
            )

    # Adjust text positions to prevent overlap
    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
        expand_points=(1.5, 1.5),
    )

    plt.xlabel("Average Cosine Similarity")
    plt.ylabel("Jaccard Similarity")
    plt.title("Cosine Similarity vs Jaccard Similarity")
    plt.legend()
    plt.savefig(
        os.path.join(out_dir, "cosine_vs_jaccard.png"), bbox_inches="tight", dpi=300
    )
    plt.close()

    # Create a bar plot for the top 10 highest cosine similarity features
    top_cosine_features = feature_indices_df.nlargest(10, "Cosine Similarity")
    plt.figure(figsize=(8, 8))

    # Set colors for highlighted and non-highlighted features
    colors = [
        "red" if idx in highlight_indices else "blue"
        for idx in top_cosine_features["Feature Index"]
    ]

    plt.bar(
        top_cosine_features["Feature Index"].astype(str),
        top_cosine_features["Cosine Similarity"],
        color=colors,
    )

    plt.xlabel("Feature Index")
    plt.ylabel("Cosine Similarity")
    plt.title("Top 10 Highest Cosine Similarity Features")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "top_cosine_similarity_features.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


if __name__ == "__main__":
    # Load config
    config_path = os.path.join(
        get_git_root(), "src", "linear_probes", "config_linear_gemma_counting.toml"
    )
    config = toml.load(config_path)

    # Extract config values
    model_name = config["model"]["model_name"]
    sae_release = config["sae"]["sae_release"]
    sae_id = config["sae"]["sae_id"]
    layer_idx = config["layer"]["layer_idx"]
    n_examples = config["examples"]["n_examples"]

    # Example highlight indices (modify as needed)
    highlight_indices = [1469, 6449, 8129, 13989]

    # Create plot
    plot_feature_cooccurrence(
        model_name=model_name,
        sae_release=sae_release,
        sae_id=sae_id,
        # layer_idx=layer_idx,
        n_examples=n_examples,
        highlight_indices=highlight_indices,
    )
