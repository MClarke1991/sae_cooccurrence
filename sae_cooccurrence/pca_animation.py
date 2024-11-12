import io
import os
import re
from typing import Mapping

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from plotly.subplots import make_subplots

from sae_cooccurrence.pca import (
    ProcessedExamples,
    ReprocessedResults,
    generate_subgraph_plot_data,
    get_point_result,
    load_subgraph,
    prepare_data,
)


def analyze_specific_points_animated(
    results: ProcessedExamples | ReprocessedResults,
    fs_splitting_nodes: list[int],
    fs_splitting_cluster: int,
    activation_threshold: float,
    node_df: pd.DataFrame,
    results_path: str,
    pca_df: pd.DataFrame,
    point_ids: list[int],
    plot_only_fs_nodes: bool = False,
    save_gif: bool = False,
    gif_path: str = "animation.gif",
):
    """
    Animate the PCA plot, feature activation, and subgraph visualization for a set of points through the PCA.

    Note: This function is deprecated in favour of analyze_specific_points_animated_from_thresholded. (This is a
    legacy function that depends on subgraphs being saved as pickles which is not current practice see equivalent
    function with from_thresholded instead).
    """

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("PCA Plot", "Feature Activation", "Subgraph Visualization"),
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
        horizontal_spacing=0.1,
    )

    # Calculate fixed node positions for the subgraph
    subgraph = load_subgraph(results_path, activation_threshold, fs_splitting_cluster)
    fixed_pos = nx.spring_layout(subgraph, seed=42)  # Use a fixed seed for consistency

    # Calculate global maximum activation
    global_max_activation = 0
    for point_id in point_ids:
        point_result = get_point_result(results, point_id)
        df, _ = prepare_data(point_result, fs_splitting_nodes, node_df)
        global_max_activation = max(global_max_activation, df["Activation"].max())

    # Create frames for animation
    frames = []
    for point_id in point_ids:
        frame_data, context = create_frame_data(
            results=results,
            fs_splitting_nodes=fs_splitting_nodes,
            fs_splitting_cluster=fs_splitting_cluster,
            activation_threshold=activation_threshold,
            node_df=node_df,
            results_path=results_path,
            pca_df=pca_df,
            point_id=point_id,
            plot_only_fs_nodes=plot_only_fs_nodes,
            fixed_pos=fixed_pos,
        )
        frame = go.Frame(
            data=frame_data,
            name=str(point_id),
            layout=go.Layout(title=f"Point ID: {point_id}<br>{context}"),
        )
        frames.append(frame)

    # Add traces for initial state (first point)
    initial_frame_data = frames[0].data
    for trace in initial_frame_data:
        fig.add_trace(trace)

    # Update layout
    fig.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 500, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ],
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Point ID: ",
                    "visible": True,
                    "xanchor": "right",
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [str(point_id)],
                            {
                                "frame": {"duration": 300, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 300},
                            },
                        ],
                        "label": str(point_id),
                        "method": "animate",
                    }
                    for point_id in point_ids
                ],
            }
        ],
        height=600,
        width=1500,
        title=f"Point ID: {point_ids[0]}<br>{frames[0].layout.title.text.split('<br>')[1]}",
        yaxis2=dict(
            range=[0, global_max_activation]
        ),  # Set fixed y-axis range for bar plot
    )

    # Add frames to the figure
    fig.frames = frames

    if save_gif:
        # Create a folder to save individual frames
        gif_name = os.path.splitext(os.path.basename(gif_path))[0]
        # if gif_name does not end in .gif append this
        if not gif_name.endswith(".gif"):
            gif_name += ".gif"
        gif_dir = os.path.dirname(gif_path)
        frames_folder = os.path.join(gif_dir, f"{gif_name}_gif_frames")
        print(frames_folder)
        os.makedirs(frames_folder, exist_ok=True)

        # Generate images for each frame in the animation
        gif_frames = []
        for i, frame in enumerate(fig.frames):
            # Set main traces to appropriate traces within plotly frame
            fig.update(data=frame.data)
            # Update the title with the context for this frame
            fig.update_layout(title=repr(frame.layout.title.text))
            # Generate image of current state with higher resolution
            img_bytes = fig.to_image(format="png", scale=4.0)

            # Save the frame as PNG
            frame_path = os.path.join(frames_folder, f"frame_{i:04d}.png")
            with open(frame_path, "wb") as f:
                f.write(img_bytes)

            gif_frames.append(Image.open(io.BytesIO(img_bytes)))

        # Create animated GIF
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            optimize=True,
            duration=1000,
            loop=0,
        )
        print(f"Animation saved as GIF: {gif_path}")
        print(f"Individual frames saved in folder: {frames_folder}")

    return fig


def analyze_specific_points_animated_from_thresholded(
    results: ProcessedExamples | ReprocessedResults,
    thresholded_matrix: np.ndarray,
    fs_splitting_nodes: list[int],
    fs_splitting_cluster: int,
    # activation_threshold: float,
    node_df: pd.DataFrame,
    # results_path: str,
    pca_df: pd.DataFrame,
    point_ids: list[int],
    plot_only_fs_nodes: bool = False,
    save_gif: bool = False,
    gif_path: str = "animation.gif",
    gif_filename: str = "animation.gif",
    frame_folder_name: str = "gif_frames",
):
    """
        Animate the PCA plot, feature activation, and subgraph visualization for a set of points through the PCA.

        results: ProcessedExamples | ReprocessedResults
        thresholded_matrix: np.ndarray
        fs_splitting_nodes: list[int]
        fs_splitting_cluster: int
        node_df: pd.DataFrame
        pca_df: pd.DataFrame
        point_ids: list[int]
        plot_only_fs_nodes: bool = False
        This decides whether or not to plot subgraphs that are active but are not the subgraph of interest.
        This can only be done if you are using data where all feature activations were saved which is not the standard
        practice when running the script that generates data for the Streamlit app.
        save_gif: bool = False
        gif_path: str = "animation.gif"
        gif_filename: str = "animation.gif"
        frame_folder_name: str = "gif_frames"
    ):
    """

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("PCA Plot", "Feature Activation", "Subgraph Visualization"),
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
        horizontal_spacing=0.1,
    )

    # Calculate fixed node positions for the subgraph
    subgraph, subgraph_df = generate_subgraph_plot_data(
        thresholded_matrix, node_df, fs_splitting_cluster
    )
    fixed_pos = nx.spring_layout(
        subgraph, seed=42, k=0.5
    )  # Use a fixed seed for consistency

    # Calculate global maximum activation
    global_max_activation = 0
    for point_id in point_ids:
        point_result = get_point_result(results, point_id)
        df, _ = prepare_data(point_result, fs_splitting_nodes, node_df)
        global_max_activation = max(global_max_activation, df["Activation"].max())

    # Create frames for animation
    frames = []
    for point_id in point_ids:
        frame_data, context = create_frame_data_from_thresholded(
            results=results,
            thresholded_matrix=thresholded_matrix,
            fs_splitting_nodes=fs_splitting_nodes,
            fs_splitting_cluster=fs_splitting_cluster,
            # activation_threshold=activation_threshold,
            node_df=node_df,
            # results_path=results_path,
            pca_df=pca_df,
            point_id=point_id,
            plot_only_fs_nodes=plot_only_fs_nodes,
            fixed_pos=fixed_pos,
        )
        # context = "TESTTESTTEST"
        frame = go.Frame(
            data=frame_data,
            name=str(point_id),
            layout=go.Layout(
                title=f"Point ID: {point_id}<br>{sanitise_context(context)}"
            ),
        )
        frames.append(frame)

    # Add traces for initial state (first point)
    initial_frame_data = frames[0].data
    for trace in initial_frame_data:
        fig.add_trace(trace)
    print(frames[0].layout.title.text)
    # Update layout
    fig.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 500, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ],
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Point ID: ",
                    "visible": True,
                    "xanchor": "right",
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [str(point_id)],
                            {
                                "frame": {"duration": 300, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 300},
                            },
                        ],
                        "label": str(point_id),
                        "method": "animate",
                    }
                    for point_id in point_ids
                ],
            }
        ],
        height=600,
        width=1500,
        # title=f"Point ID: {point_ids[0]}<br>{frames[0].layout.title.text.split('<br>')[1]}",
        yaxis2=dict(
            range=[0, global_max_activation]
        ),  # Set fixed y-axis range for bar plot
        paper_bgcolor="white",  # Set paper background to white
        plot_bgcolor="white",
    )

    # Add frames to the figure
    fig.frames = frames

    if save_gif:
        if not gif_filename.endswith(".gif"):
            gif_filename += ".gif"
        print(gif_filename)
        # Create a folder to save individual frames
        frames_folder = os.path.join(gif_path, frame_folder_name)
        print(frames_folder)
        os.makedirs(frames_folder, exist_ok=True)

        # Generate images for each frame in the animation
        gif_frames = []
        for i, frame in enumerate(fig.frames):
            # Set main traces to appropriate traces within plotly frame
            fig.update(data=frame.data)
            # Update the title with the context for this frame
            fig.update_layout(title=repr(frame.layout.title.text))
            # Generate image of current state with higher resolution
            img_bytes = fig.to_image(format="png", scale=4.0)

            # Save the frame as PNG
            frame_path = os.path.join(frames_folder, f"frame_{i:04d}.png")
            with open(frame_path, "wb") as f:
                f.write(img_bytes)

            gif_frames.append(Image.open(io.BytesIO(img_bytes)))

        # Create animated GIF
        gif_frames[0].save(
            os.path.join(gif_path, gif_filename),
            save_all=True,
            append_images=gif_frames[1:],
            optimize=True,
            duration=1000,
            loop=0,
        )
        print(f"Animation saved as GIF: {gif_path}")
        print(f"Individual frames saved in folder: {frames_folder}")

    return fig


def create_frame_data(
    results: ProcessedExamples | ReprocessedResults,
    fs_splitting_nodes: list[int],
    fs_splitting_cluster: int,
    activation_threshold: float,
    node_df: pd.DataFrame,
    results_path: str,
    pca_df: pd.DataFrame,
    point_id: int,
    plot_only_fs_nodes: bool = False,
    fixed_pos: Mapping[int, tuple[float, float]] | None = None,
):
    """
    Create the data for a single frame of the animation including the PCA plot, feature activation, and subgraph visualization.

    This is a legacy function that depends on subgraphs being saved as pickles which is not current practice see equivalent function with from thresholded instead

    results: ProcessedExamples | ReprocessedResults
    fs_splitting_nodes: list[int]
    fs_splitting_cluster: int
    activation_threshold: float
    node_df: pd.DataFrame
    results_path: str
    pca_df: pd.DataFrame
    point_id: int
    plot_only_fs_nodes: bool = False
    fixed_pos: Mapping[int, tuple[float, float]] | None = None
    """

    frame_data = []

    # PCA Plot
    pca_trace = go.Scatter(
        x=pca_df["PC2"],
        y=pca_df["PC3"],
        mode="markers",
        marker=dict(
            color=["red" if idx == point_id else "lightgrey" for idx in pca_df.index],
            size=[15 if idx == point_id else 5 for idx in pca_df.index],
        ),
        # text=[
        #     context if idx == point_id else None
        #     for idx, context in zip(pca_df.index, pca_df["context"])
        # ],
        # hoverinfo="text",
        showlegend=False,
        xaxis="x",
        yaxis="y",
    )
    frame_data.append(pca_trace)

    # Bar Plot
    point_result = get_point_result(results, point_id)

    if isinstance(point_result, ReprocessedResults):
        raise ValueError(
            "Passed ReprocessedResults, but cannot plot other subgraphs from Streamlit data. "
            "Please set plot_without_other_subgraphs=True when loading data from generation or pass ProcessedExamples."
        )

    df, context = prepare_data(point_result, fs_splitting_nodes, node_df)

    # Add missing fs_splitting_nodes with activity 0
    missing_nodes = set(fs_splitting_nodes) - set(df["Feature Index"])
    if missing_nodes:
        missing_df = pd.DataFrame(
            {
                "Feature Index": list(missing_nodes),
                "Activation": [0] * len(missing_nodes),
                "subgraph_id": [None] * len(missing_nodes),
                "subgraph_size": [None] * len(missing_nodes),
            }
        )
        df = pd.concat([df, missing_df], ignore_index=True)

    if plot_only_fs_nodes:
        df["Feature Index"] = df["Feature Index"].astype(int)
        df = df[df["Feature Index"].isin(fs_splitting_nodes)]
        df = df.sort_values("Feature Index")  # type: ignore
        df["Feature Index"] = df["Feature Index"].astype(str)
    else:  # Sort by activation
        df = df.sort_values("Activation", ascending=False)

    bar_trace = go.Bar(
        x=df["Feature Index"].astype(str),
        y=df["Activation"],
        marker_color=[
            "red" if idx == fs_splitting_cluster else "blue"
            for idx in df["subgraph_id"]
        ],
        showlegend=False,
        xaxis="x2",
        yaxis="y2",
    )
    frame_data.append(bar_trace)

    # Subgraph Visualization
    subgraph = load_subgraph(results_path, activation_threshold, fs_splitting_cluster)
    activation_array = point_result.all_feature_acts.flatten().cpu().numpy()

    edge_trace, node_trace = create_subgraph_traces(
        subgraph, activation_array, fixed_pos
    )
    frame_data.extend([edge_trace, node_trace])

    return frame_data, context


def escape_latex(string: str) -> str:
    string = string.replace("$", "\\$")
    return string


def remove_markdown(string: str) -> str:
    # Remove markdown formatting
    # Bold and italic
    string = re.sub(r"\*\*.*?\*\*", lambda m: m.group(0).replace("**", ""), string)
    string = re.sub(r"\*.*?\*", lambda m: m.group(0).replace("*", ""), string)
    string = re.sub(r"__.*?__", lambda m: m.group(0).replace("__", ""), string)
    string = re.sub(r"_.*?_", lambda m: m.group(0).replace("_", ""), string)

    # Headers
    string = re.sub(r"#{1,6}\s", "", string)

    # Links
    string = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", string)

    # Backticks for code
    string = re.sub(r"`([^`]+)`", r"\1", string)

    return string


def sanitise_context(context: str) -> str:
    """Sanitize context string by escaping special characters, unless already quoted."""
    # context = escape(context, quote=True)
    # context = remove_markdown(context)
    # context = escape_latex(context)
    if context.startswith('"') and context.endswith('"'):
        return context
    return repr(str(context))[1:-1]


def create_frame_data_from_thresholded(
    results: ProcessedExamples | ReprocessedResults,
    thresholded_matrix: np.ndarray,
    fs_splitting_nodes: list[int],
    fs_splitting_cluster: int,
    # activation_threshold: float,
    node_df: pd.DataFrame,
    # results_path: str,
    pca_df: pd.DataFrame,
    point_id: int,
    plot_only_fs_nodes: bool = False,
    fixed_pos: Mapping[int, tuple[float, float]] | None = None,
):
    """
    Create the data for a single frame of the animation including the PCA plot, feature activation, and subgraph visualization.
    """

    frame_data = []

    # Convert the context to a string
    # Use repr() to escape special characters as plaintext (e.g., \n becomes \\n)
    # Remove the quotes that repr() adds by slicing [1:-1]
    # title = [
    #         sanitise_context(context) if idx == point_id else None
    #         for idx, context in zip(pca_df.index, pca_df["context"])
    #     ]

    # PCA Plot
    pca_trace = go.Scatter(
        x=pca_df["PC2"],
        y=pca_df["PC3"],
        mode="markers",
        marker=dict(
            color=["red" if idx == point_id else "lightgrey" for idx in pca_df.index],
            size=[15 if idx == point_id else 5 for idx in pca_df.index],
        ),
        # text=title,
        # hoverinfo="text",
        showlegend=False,
        xaxis="x",
        yaxis="y",
    )
    frame_data.append(pca_trace)

    # Bar Plot
    point_result = get_point_result(results, point_id)

    df, context = prepare_data(point_result, fs_splitting_nodes, node_df)
    # context = sanitise_context(context)

    # Add missing fs_splitting_nodes with activity 0
    missing_nodes = set(fs_splitting_nodes) - set(df["Feature Index"])
    if missing_nodes:
        missing_df = pd.DataFrame(
            {
                "Feature Index": list(missing_nodes),
                "Activation": [0] * len(missing_nodes),
                "subgraph_id": [None] * len(missing_nodes),
                "subgraph_size": [None] * len(missing_nodes),
            }
        )
        df = pd.concat([df, missing_df], ignore_index=True)

    if plot_only_fs_nodes:
        df["Feature Index"] = df["Feature Index"].astype(int)
        df = df[df["Feature Index"].isin(fs_splitting_nodes)]
        df = df.sort_values("Feature Index")  # type: ignore
        df["Feature Index"] = df["Feature Index"].astype(str)
    else:  # Sort by activation
        df = df.sort_values("Activation", ascending=False)

    bar_trace = go.Bar(
        x=df["Feature Index"].astype(str),
        y=df["Activation"],
        marker_color=[
            "red" if idx == fs_splitting_cluster else "blue"
            for idx in df["subgraph_id"]
        ],
        showlegend=False,
        xaxis="x2",
        yaxis="y2",
    )
    frame_data.append(bar_trace)

    # Subgraph Visualization
    subgraph, subgraph_df = generate_subgraph_plot_data(
        thresholded_matrix, node_df, fs_splitting_cluster
    )

    # Relabel nodes with their actual node_ids from subgraph_df
    node_mapping = {i: node_id for i, node_id in enumerate(subgraph_df["node_id"])}
    subgraph = nx.relabel_nodes(subgraph, node_mapping)
    activation_array = point_result.all_graph_feature_acts.flatten().cpu().numpy()
    fixed_pos_seq = fixed_pos
    fixed_pos = (
        {node_mapping[idx]: pos for idx, pos in fixed_pos_seq.items()}
        if fixed_pos_seq
        else None
    )

    if fixed_pos is None:
        raise ValueError("fixed_pos is None")

    edge_trace, node_trace = create_subgraph_traces(
        subgraph, activation_array, fixed_pos, short_array=True
    )
    frame_data.extend([edge_trace, node_trace])

    return frame_data, context


def create_subgraph_traces(
    subgraph: nx.Graph,
    activation_array: np.ndarray,
    pos: Mapping[int, tuple[float, float]] | None,
    short_array: bool = False,
):
    """
    Create the traces for the subgraph visualization.

    This function is used for both the legacy function that depends on subgraphs being saved as pickles which is not current practice see equivalent function with from thresholded instead
    and the function that creates the subgraph traces from the thresholded matrix.

    """

    if pos is None:
        raise ValueError("pos is None")

    edge_x, edge_y = [], []
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
        showlegend=False,
        xaxis="x3",
        yaxis="y3",
    )

    node_x, node_y = [], []
    for node in subgraph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    # Normalize activation values for color scaling
    if not short_array:
        node_activations = [activation_array[node] for node in subgraph.nodes()]
    else:
        node_activations = activation_array
    normalized_activations = (
        np.array(node_activations) - np.min(node_activations)  # type: ignore
    ) / (np.max(node_activations) - np.min(node_activations))  # type: ignore

    # Prepare the color map
    cmap = plt.cm.get_cmap("Blues")
    n_colors = 256

    # Get the colormap in RGB
    colormap_RGB = cmap(np.arange(cmap.N))

    # Set the color for zero values to be white
    colormap_RGB[0] = (
        1,
        1,
        1,
        1,
    )  # This line sets the first color in the colormap to white

    # Prepare custom color scale (in Plotly format)
    colorscale = [
        [i / (n_colors - 1), mcolors.rgb2hex(colormap_RGB[i])] for i in range(n_colors)
    ]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale=colorscale,
            reversescale=False,
            color=normalized_activations,
            size=15,
            colorbar=dict(
                thickness=15,
                title="Normalized Activation",
                xanchor="left",
                titleside="right",
                bgcolor="white",
            ),
            line_width=2,
        ),
        text=[str(node) for node in subgraph.nodes()],  # Add feature number as text
        textposition="top center",  # Position the text above the node
        textfont=dict(size=10, color="black"),  # Customize text appearance
        showlegend=False,
        xaxis="x3",
        yaxis="y3",
    )

    # node_hover_text = []
    # for node in subgraph.nodes():
    #     node_info = node_df[node_df["node_id"] == node].iloc[0]
    #     top_tokens = ast.literal_eval(node_info["top_10_tokens"])
    #     node_hover_text.append(
    #         f"Feature: {node}<br>Activation: {node_activations[node]:.4f}<br>Top token: {top_tokens[0]}"
    #     )

    # node_trace.hovertext = node_hover_text

    return edge_trace, node_trace
