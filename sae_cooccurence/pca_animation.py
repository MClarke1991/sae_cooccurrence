import plotly.express as px
import plotly.graph_objects as go

from sae_cooccurence.pca import ProcessedExamples, prepare_data


def get_point_result(results, idx):
    point_result = ProcessedExamples(
        all_token_dfs=results.all_token_dfs.iloc[[idx]],
        all_fired_tokens=[results.all_fired_tokens[idx]],
        all_reconstructions=results.all_reconstructions[[idx]],
        all_graph_feature_acts=results.all_graph_feature_acts[[idx]],
        all_feature_acts=results.all_feature_acts[[idx]],
        all_max_feature_info=results.all_max_feature_info[[idx]],
        all_examples_found=1,
    )
    return point_result


def create_animated_plots(plot_data):
    """
    Create animated Plotly figures for PCA and bar plots.

    Args:
    plot_data (list): List of dictionaries containing plot data for each point.

    Returns:
    tuple: Two Plotly figure objects (pca_fig, bar_fig) with animations.
    """
    # Create figures
    pca_fig = go.Figure()
    bar_fig = go.Figure()

    # Add traces for each point
    for frame_data in plot_data:
        point_id = frame_data["pointId"]

        # PCA Plot
        pca_frame_data = frame_data["pcaPlotData"]
        pca_fig.add_trace(
            go.Scatter(
                x=[point["x"] for point in pca_frame_data],
                y=[point["y"] for point in pca_frame_data],
                mode="markers",
                marker=dict(
                    color=[point["color"] for point in pca_frame_data],
                    size=[point["size"] for point in pca_frame_data],
                    symbol=[point["symbol"] for point in pca_frame_data],
                ),
                text=[point["text"] for point in pca_frame_data],
                name=f"Point {point_id}",
                visible=False,
            )
        )

        # Bar Plot
        bar_frame_data = frame_data["barPlotData"]
        bar_fig.add_trace(
            go.Bar(
                x=[data["Feature Index"] for data in bar_frame_data],
                y=[data["Activation"] for data in bar_frame_data],
                name=f"Point {point_id}",
                visible=False,
            )
        )

    # Make the first trace visible
    pca_fig.data[0].visible = True  # type: ignore
    bar_fig.data[0].visible = True  # type: ignore

    # Create and add slider steps
    steps = []
    for i in range(len(plot_data)):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(pca_fig.data)},  # type: ignore
                {"title": f"PCA Plot - Point {plot_data[i]['pointId']}"},
            ],
            label=str(plot_data[i]["pointId"]),
        )
        step["args"][0]["visible"][i] = True  # type: ignore
        steps.append(step)

    sliders = [
        dict(
            active=0, currentvalue={"prefix": "Point ID: "}, pad={"t": 50}, steps=steps
        )
    ]

    # Update layout for both figures
    pca_fig.update_layout(
        sliders=sliders,
        title="PCA Plot Animation",
        xaxis_title="PC2",
        yaxis_title="PC3",
        height=600,
        width=800,
    )

    bar_fig.update_layout(
        sliders=sliders,
        title="Feature Activation Bar Plot Animation",
        xaxis_title="Feature Index",
        yaxis_title="Activation",
        height=600,
        width=800,
    )

    return pca_fig, bar_fig


def analyze_specific_points_animation(
    results,
    fs_splitting_nodes,
    node_df,
    pca_df,
    point_ids,
):
    """
    Analyze specific points and return plot data for animation.
    """
    plot_data = []
    colors = px.colors.qualitative.Safe[: len(point_ids)]

    for i, point_id in enumerate(point_ids):
        point_result = get_point_result(results, point_id)
        df, context = prepare_data(point_result, fs_splitting_nodes, node_df)

        # Prepare bar plot data
        bar_plot_data = df[["Feature Index", "Activation"]].to_dict("records")  # type: ignore

        # Prepare PCA plot data
        pca_plot_data = [
            {
                "x": pca_df.loc[id, "PC2"],
                "y": pca_df.loc[id, "PC3"],
                "color": "lightgrey" if id != point_id else colors[i],
                "size": 10 if id != point_id else 15,
                "symbol": "circle" if id != point_id else "star",
                "text": pca_df.loc[id, "context"] if id == point_id else None,
            }
            for id in pca_df.index
        ]

        plot_data.append(
            {
                "pointId": point_id,
                "context": context,
                "barPlotData": bar_plot_data,
                "pcaPlotData": pca_plot_data,
            }
        )

    return plot_data
