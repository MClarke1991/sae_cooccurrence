import ast
import os
from os.path import join as pj

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sae_cooccurrence.graph_generation import plot_subgraph_static
from sae_cooccurrence.pca import (
    ProcessedExamples,
    assign_category,
    create_bar_plot,
    create_pie_charts,
    get_active_subgraphs,
    get_point_result,
    prepare_data,
    print_statistics,
    select_representative_points,
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


def create_comprehensive_plot(
    pca_df,
    highlighted_indices,
    contexts,
    results,
    fs_splitting_nodes,
    node_df,
    activation_threshold,
):
    """
    Create a comprehensive plot with PCA, bar plots, and pie charts for user-specified points.

    Args:
    pca_df (pd.DataFrame): DataFrame containing PCA results
    highlighted_indices (list): Indices of points to highlight
    contexts (list): list of context strings for highlighted points
    results (ProcessedExamples): Results from process_examples function
    fs_splitting_nodes (list): list of feature splitting node indices
    activation_threshold (float): Threshold for considering a feature as active

    Returns:
    go.Figure: Plotly figure object
    """
    # Calculate the number of rows needed (1 for PCA, 1 for each user-specified point)
    n_rows = 1 + len(highlighted_indices)

    # Create subplot layout
    fig = make_subplots(
        rows=n_rows,
        cols=3,
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]]  # PCA plots
        + [
            [{"type": "xy"}, {"type": "domain"}, {"type": "domain"}]
            for _ in range(len(highlighted_indices))
        ],  # Bar and Pie charts
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
    )

    # Color palette for highlighted points
    colors = px.colors.qualitative.Set1[: len(highlighted_indices)]

    # Plot PCA
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
                showlegend=False,
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

    # Plot bar charts and pie charts for each user-specified point
    for i, idx in enumerate(highlighted_indices):
        # Extract data for this point
        point_result = get_point_result(results, idx)

        df, context = prepare_data(point_result, fs_splitting_nodes, node_df)

        # Bar plot
        color_map = {True: colors[i], False: "lightgrey"}
        for _, row in df.iterrows():
            fig.add_trace(
                go.Bar(
                    x=[row["Feature Index"]],
                    y=[row["Activation"]],
                    marker_color=color_map[row["Is Feature Splitting"]],  # type: ignore
                    showlegend=False,
                ),
                row=i + 2,
                col=1,
            )

        fig.update_xaxes(title_text="Feature Index", row=i + 2, col=1)
        fig.update_yaxes(title_text="Activation", row=i + 2, col=1)

        # Pie charts
        active_splitting = sum(
            (df["Is Feature Splitting"]) & (df["Activation"] > activation_threshold)
        )
        active_non_splitting = sum(
            (~df["Is Feature Splitting"]) & (df["Activation"] > activation_threshold)
        )
        sum_splitting = df[df["Is Feature Splitting"]]["Activation"].sum()
        sum_non_splitting = df[~df["Is Feature Splitting"]]["Activation"].sum()

        fig.add_trace(
            go.Pie(
                labels=["Splitting", "Non-splitting"],
                values=[active_splitting, active_non_splitting],
                marker_colors=[colors[i], "lightgrey"],
                domain=dict(x=[0, 0.45], y=[0, 1]),
                name="Active Features",
                title="N. Active Features",
                showlegend=False,
            ),
            row=i + 2,
            col=2,
        )

        fig.add_trace(
            go.Pie(
                labels=["Splitting", "Non-splitting"],
                values=[sum_splitting, sum_non_splitting],
                marker_colors=[colors[i], "lightgrey"],
                domain=dict(x=[0.55, 1], y=[0, 1]),
                name="Activation Sum",
                title="Activation Sum",
                showlegend=False,
            ),
            row=i + 2,
            col=3,
        )

        # Add context annotation
        fig.add_annotation(
            text=contexts[i],
            xref="paper",
            yref="paper",
            x=0,
            y=1 - (i + 1.5) / n_rows,  # Adjust y position for each row
            xanchor="left",
            yanchor="bottom",
            font=dict(size=10, color=colors[i]),
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.8)",  # Semi-transparent white background
            bordercolor=colors[i],
            borderwidth=2,
            borderpad=4,
        )

    # Update layout
    fig.update_layout(
        height=300 * n_rows,
        width=1500,
        title_text="PCA and Feature Activation Analysis for User-Specified Points",
    )

    return fig


def analyze_representative_points_comp(
    results,
    fs_splitting_nodes,
    activation_threshold,
    node_df,
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

    # Create comprehensive plot
    comp_fig = create_comprehensive_plot(
        pca_df,
        rep_indices,
        contexts,
        results,
        fs_splitting_nodes,
        node_df,
        activation_threshold,
    )
    comp_fig.show()

    if save_figs and pca_path:
        comp_fig.write_image(pj(pca_path, "comprehensive_analysis.png"), scale=4.0)
        comp_fig.write_image(pj(pca_path, "comprehensive_analysis.svg"))
        comp_fig.write_image(pj(pca_path, "comprehensive_analysis.pdf"))
        comp_fig.write_html(pj(pca_path, "comprehensive_analysis.html"))

    # Print statistics for each point
    for i, idx in enumerate(rep_indices):
        print(f"\nStatistics for representative point {i+1}:")
        point_result = get_point_result(results, idx)
        df, _ = prepare_data(point_result, fs_splitting_nodes, node_df)
        print_statistics(df, fs_splitting_nodes, activation_threshold)


def analyze_user_specified_points_comp(
    results: ProcessedExamples,
    fs_splitting_nodes: list[int],
    activation_threshold: float,
    node_df: pd.DataFrame,
    pca_df: pd.DataFrame,
    user_specified_ids: list[int],
    save_figs=False,
    pca_path=None,
) -> None:
    """
    Analyze and visualize user-specified points from the PCA plot.

    Args:
    results (ProcessedExamples): Results from process_examples function
    fs_splitting_nodes (list): list of feature splitting node indices
    activation_threshold (float): Threshold for considering a feature as active
    node_df (pd.DataFrame): DataFrame containing node information
    pca_df (pd.DataFrame): DataFrame containing PCA results
    user_specified_ids (list): list of point IDs specified by the user
    save_figs (bool): Whether to save the figures
    pca_path (str): Path to save the figures
    """
    # Get contexts for user-specified points
    contexts = [
        results.all_token_dfs["context"].iloc[idx] for idx in user_specified_ids
    ]

    # concatenate user specified ids with underscores
    all_ids = "_".join([str(i) for i in user_specified_ids])

    # Create comprehensive plot
    comp_fig = create_comprehensive_plot(
        pca_df,
        user_specified_ids,
        contexts,
        results,
        fs_splitting_nodes,
        node_df,
        activation_threshold,
    )
    comp_fig.show()

    if save_figs and pca_path:
        comp_fig.write_image(
            pj(pca_path, f"user_specified_comprehensive_analysis{all_ids}.png"),
            scale=4.0,
        )
        comp_fig.write_image(
            pj(pca_path, f"user_specified_comprehensive_analysis{all_ids}.svg")
        )
        comp_fig.write_image(
            pj(pca_path, f"user_specified_comprehensive_analysis{all_ids}.pdf")
        )
        comp_fig.write_html(
            pj(pca_path, f"user_specified_comprehensive_analysis{all_ids}.html")
        )

    # Print statistics for each point
    for i, idx in enumerate(user_specified_ids):
        print(f"\nStatistics for user-specified point {i+1} (ID: {idx}):")
        point_result = get_point_result(results, idx)
        df, _ = prepare_data(point_result, fs_splitting_nodes, node_df)
        print_statistics(df, fs_splitting_nodes, activation_threshold)
