import ast
import os
from os.path import join as pj

import matplotlib.pyplot as plt
import networkx as nx

from sae_cooccurrence.graph_generation import plot_subgraph_static
from sae_cooccurrence.pca import (
    assign_category,
    create_bar_plot,
    create_pie_charts,
    get_active_subgraphs,
    prepare_data,
    print_statistics,
)

# Old version of barplot with pie charts


def create_combined_subgraph_bar_plot(
    subgraph,
    node_df,
    activation_array,
    df,
    context,
    fs_splitting_cluster,
    color_other_subgraphs=True,
    order_other_subgraphs=True,
):
    # Create a new figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))  # type: ignore

    # Plot subgraph on the left subplot
    pos = nx.spring_layout(subgraph, k=0.5, iterations=50, seed=1234)
    # subgraph_activations = [activation_array[node] for node in subgraph.nodes()]
    min_activation = min(activation_array)
    max_activation = max(activation_array)
    activation_range = max_activation - min_activation

    labels = {}
    node_colors = []
    for node in subgraph.nodes():
        node_info = node_df[node_df["node_id"] == node].iloc[0]
        node_id = node_info["node_id"]
        top_tokens = ast.literal_eval(node_info["top_10_tokens"])
        top_token = top_tokens[0]
        labels[node] = f"ID: {node_id}\n{top_token}"

        if activation_array[node] == 0:
            node_colors.append("white")
        else:
            normalized_activation = (
                (activation_array[node] - min_activation) / activation_range
                if activation_range != 0
                else 0.5
            )
            node_colors.append(plt.cm.viridis(normalized_activation))  # type: ignore

    edge_weights = [subgraph[u][v]["weight"] for u, v in subgraph.edges()]
    max_weight = max(edge_weights)
    min_weight = min(edge_weights)
    normalized_weights = [
        (w - min_weight) / (max_weight - min_weight) for w in edge_weights
    ]
    edge_thickness = [0.5 + 4.5 * w for w in normalized_weights]

    nx.draw(
        subgraph,
        pos,
        ax=ax1,
        with_labels=False,
        node_size=300,
        node_color=node_colors,
        edgecolors="black",
        linewidths=1,
        edge_color="gray",
        width=edge_thickness,
        arrows=True,
    )

    label_pos = {k: (v[0], v[1] - 0.1) for k, v in pos.items()}
    nx.draw_networkx_labels(subgraph, label_pos, labels, font_size=8, ax=ax1)

    ax1.set_title("Subgraph Visualization", fontsize=16)
    ax1.axis("off")

    # Create bar plot on the right subplot
    plot_df = df.copy()

    plot_df["Category"] = plot_df.apply(
        lambda row: assign_category(row, fs_splitting_cluster, order_other_subgraphs),
        axis=1,
    )

    subgraph_max_activations = (
        plot_df[plot_df["Category"] == 1].groupby("subgraph_id")["Activation"].max()
    )
    subgraph_order = subgraph_max_activations.sort_values(ascending=False).index

    subgraph_order_map = {
        subgraph: order for order, subgraph in enumerate(subgraph_order)
    }

    plot_df["SubgraphOrder"] = plot_df["subgraph_id"].map(subgraph_order_map)
    plot_df["SubgraphOrder"] = plot_df["SubgraphOrder"].fillna(len(subgraph_order_map))

    plot_df = plot_df.sort_values(
        ["Category", "SubgraphOrder", "Activation"], ascending=[True, True, False]
    )
    plot_df = plot_df.reset_index(drop=True)

    def assign_color(row, color_other_subgraphs):
        if row["subgraph_id"] == fs_splitting_cluster:
            return "red"
        elif row["Category"] == 1 and color_other_subgraphs:
            return "blue"
        else:
            return "grey"

    plot_df["Color"] = plot_df.apply(
        lambda row: assign_color(row, color_other_subgraphs), axis=1
    )

    ax2.bar(plot_df.index, plot_df["Activation"], color=plot_df["Color"])
    ax2.set_xlabel("Feature Index (Sorted)", fontsize=12)
    ax2.set_ylabel("Feature Activation", fontsize=12)
    ax2.set_title(f"Feature Activations\n{context}", fontsize=16)

    mean_activation = plot_df["Activation"].mean()
    ax2.axhline(
        y=mean_activation, color="green", linestyle="--", label="Mean activation"
    )
    ax2.legend()

    plt.tight_layout()
    return fig


def plot_feature_activations_combined(
    results,
    fs_splitting_nodes,
    fs_splitting_cluster,
    activation_threshold,
    node_df,
    results_path,
    pca_path,
    save_figs=False,
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

        # Create and save combined plot
        combined_fig = create_combined_subgraph_bar_plot(
            subgraph,
            node_df,
            activation_threshold,
            activation_array,
            df,
            context,
            fs_splitting_nodes,
            fs_splitting_cluster,
        )
        if save_figs and pca_path:
            combined_fig.savefig(
                pj(pca_path, f"combined_plot_subgraph_{subgraph_id}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            print(pj(pca_path, f"combined_plot_subgraph_{subgraph_id}.png"))
            combined_fig.savefig(
                pj(pca_path, f"combined_plot_subgraph_{subgraph_id}.pdf"),
                format="pdf",
                dpi=300,
                bbox_inches="tight",
            )
            combined_fig.savefig(
                pj(pca_path, f"combined_plot_subgraph_{subgraph_id}.svg"),
                format="svg",
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(combined_fig)

    bar_fig.show()
    pie_fig.show()

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
