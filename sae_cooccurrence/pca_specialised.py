import os
import re

import pandas as pd
import plotly.graph_objects as go


def plot_pca_weekdays(
    pca_df, pca_path, fs_splitting_cluster, plot_inner=False, save_figs=False
):
    # Define colors for each day and gray for others
    if not plot_inner:
        color_map = {
            "Monday": "#FF9999",
            "Tuesday": "#66B2FF",
            "Wednesday": "#99FF99",
            "Thursday": "#FFCC99",
            "Friday": "#FF99FF",
            "Saturday": "#99FFFF",
            "Sunday": "#FFFF99",
            "Other": "#CCCCCC",
        }
    else:
        color_map = {
            "Mon": "#FF9999",
            "Tues": "#66B2FF",
            "Wed": "#99FF99",
            "Thurs": "#FFCC99",
            "Fri": "#FF99FF",
            "Sat": "#99FFFF",
            "Sun": "#FFFF99",
            "Other": "#CCCCCC",
        }

    # Function to determine color
    def get_color(token):
        token_lower = token.lower()
        for day in color_map.keys():
            if day.lower() in token_lower:
                return color_map[day]
        return color_map["Other"]

    # Apply the function to get colors
    pca_df["color"] = pca_df["tokens"].apply(get_color)

    # Create three figures for different PC combinations
    figs = []
    pc_combinations = [("PC1", "PC2"), ("PC1", "PC3"), ("PC2", "PC3")]

    for pc_x, pc_y in pc_combinations:
        fig = go.Figure()

        # Add traces for colors (days)
        for day in list(color_map.keys()):
            df_day = pca_df[pca_df["color"] == color_map[day]]
            fig.add_trace(
                go.Scatter(
                    x=df_day[pc_x],
                    y=df_day[pc_y],
                    mode="markers",
                    marker=dict(color=color_map[day], size=12, line=dict(width=0)),
                    name=day,
                    text=[
                        f"Token: {t}<br>Context: {c}"
                        for t, c in zip(df_day["tokens"], df_day["context"])
                    ],
                    hoverinfo="text",
                )
            )

        # Update layout
        fig.update_layout(
            height=800,
            width=800,
            title_text=f"PCA Analysis - Cluster {fs_splitting_cluster} ({pc_x} vs {pc_y})",
            xaxis_title=pc_x,
            yaxis_title=pc_y,
            legend=dict(groupclick="toggleitem", tracegroupgap=20),
        )

        figs.append(fig)

    outer_suffix = "" if not plot_inner else "_inner"

    if save_figs:
        for i, (pc_x, pc_y) in enumerate(pc_combinations):
            # Save as PNG
            png_path = os.path.join(
                pca_path,
                f"pca_plot_weekdays_{fs_splitting_cluster}_{pc_x}_{pc_y}{outer_suffix}.png",
            )
            figs[i].write_image(png_path, scale=3.0)

            # Save as HTML
            html_path = os.path.join(
                pca_path,
                f"pca_plot_weekdays_{fs_splitting_cluster}_{pc_x}_{pc_y}{outer_suffix}.html",
            )
            figs[i].write_html(html_path)
    else:
        for fig in figs:
            fig.show()


def plot_pca_weekdays_3d(
    pca_df, pca_path, fs_splitting_cluster, plot_inner=False, save_figs=False
):
    # Define colors for each day and gray for others
    if not plot_inner:
        color_map = {
            "Monday": "#FF9999",
            "Tuesday": "#66B2FF",
            "Wednesday": "#99FF99",
            "Thursday": "#FFCC99",
            "Friday": "#FF99FF",
            "Saturday": "#99FFFF",
            "Sunday": "#FFFF99",
            "Other": "#CCCCCC",
        }
    else:
        color_map = {
            "Mon": "#FF9999",
            "Tues": "#66B2FF",
            "Wed": "#99FF99",
            "Thurs": "#FFCC99",
            "Fri": "#FF99FF",
            "Sat": "#99FFFF",
            "Sun": "#FFFF99",
            "Other": "#CCCCCC",
        }

    # Function to determine color
    def get_color(token):
        token_lower = token.lower()
        for day in color_map.keys():
            if day.lower() in token_lower:
                return color_map[day]
        return color_map["Other"]

    # Apply the function to get colors
    pca_df["color"] = pca_df["tokens"].apply(get_color)

    # Create a 3D figure
    fig = go.Figure()

    # Add traces for colors (days)
    for day in list(color_map.keys()):
        df_day = pca_df[pca_df["color"] == color_map[day]]
        fig.add_trace(
            go.Scatter3d(
                x=df_day["PC1"],
                y=df_day["PC2"],
                z=df_day["PC3"],
                mode="markers",
                marker=dict(color=color_map[day], size=3, line=dict(width=0)),
                name=day,
                text=[
                    f"Token: {t}<br>Context: {c}"
                    for t, c in zip(df_day["tokens"], df_day["context"])
                ],
                hoverinfo="text",
            )
        )

    # Update layout
    fig.update_layout(
        height=800,
        width=800,
        title_text=f"3D PCA Analysis - Cluster {fs_splitting_cluster}",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        legend=dict(groupclick="toggleitem", tracegroupgap=20),
    )

    outer_suffix = "" if not plot_inner else "_inner"

    if save_figs:
        # Save as PNG
        png_path = os.path.join(
            pca_path, f"pca_plot_weekdays_3d_{fs_splitting_cluster}{outer_suffix}.png"
        )
        fig.write_image(png_path, scale=3.0)

        # Save as HTML
        html_path = os.path.join(
            pca_path, f"pca_plot_weekdays_3d_{fs_splitting_cluster}{outer_suffix}.html"
        )
        fig.write_html(html_path)
    else:
        fig.show()


def plot_pca_filtered_context(pca_df, pca_path, fs_splitting_cluster, save_figs=False):
    def process_and_count_chars(context):
        # Remove '<|endoftext|>' from the context
        cleaned_context = context.replace("<|endoftext|>", "")

        # Split the cleaned context by '|'
        parts = cleaned_context.split("|")

        # Check if there's exactly one character between '|' symbols
        if len(parts) == 3 and len(parts[1]) == 1:
            # single_char = parts[1]
            before_part = parts[0]

            # Check for '/watch?' string
            watch_index = before_part.rfind("/watch?")
            if watch_index != -1:
                # Count characters from end of '/watch?' to the single character
                return len(before_part) - (
                    watch_index + 7
                )  # 7 is the length of '/watch?'
            else:
                # Check if there's a '/' before the single character without spaces
                match = re.search(r"/([^/\s]+)$", before_part)
                if match:
                    # Count characters between the last '/' and the single character
                    return len(match.group(1))

        # Return None for cases that don't meet the criteria
        return None

    # Apply the processing and counting function
    pca_df["char_count"] = pca_df["context"].apply(process_and_count_chars)

    # Filter out None values
    pca_df_filtered = pca_df.dropna(subset=["char_count"])

    # Create the plot
    fig = go.Figure()

    # Add trace for all points
    fig.add_trace(
        go.Scatter(
            x=pca_df_filtered["PC2"],
            y=pca_df_filtered["PC3"],
            mode="markers",
            marker=dict(
                color=pca_df_filtered["char_count"],
                colorscale="turbo",
                size=12,
                colorbar=dict(title="Character Count"),
                line=dict(width=1, color="DarkSlateGrey"),
            ),
            text=[
                f"Token: {t}<br>Context: {c}<br>Char Count: {count}"
                for t, c, count in zip(
                    pca_df_filtered["tokens"],
                    pca_df_filtered["context"],
                    pca_df_filtered["char_count"],
                )
            ],
            hoverinfo="text",
        )
    )

    # Update layout
    fig.update_layout(
        height=800,
        width=800,
        title_text=f"PCA Analysis - Cluster {fs_splitting_cluster} (Filtered Context Character Count)",
        xaxis_title="PC2",
        yaxis_title="PC3",
    )

    if save_figs:
        # Save as PNG
        png_path = os.path.join(
            pca_path, f"pca_plot_filtered_context_char_count_{fs_splitting_cluster}.png"
        )
        fig.write_image(png_path, scale=3.0)

        # Save as HTML
        html_path = os.path.join(
            pca_path,
            f"pca_plot_filtered_context_char_count_{fs_splitting_cluster}.html",
        )
        fig.write_html(html_path)
    else:
        fig.show()

    return fig


def plot_feature_activation_normalized_area_chart(
    results,
    fs_splitting_nodes,
    pca_df,
    pca_path,
    fs_splitting_cluster,
    max_examples=1000,
    save=False,
):
    def process_context(context):
        parts = context.split("|")
        if len(parts) == 3 and len(parts[1]) == 1:
            before_part = parts[0]
            watch_index = before_part.rfind("/watch?")
            if watch_index != -1:
                return len(before_part) - (watch_index + 7)
            else:
                match = re.search(r"/([^/\s]+)$", before_part)
                if match:
                    return len(match.group(1))
        return None

    # Extract feature activations
    feature_activations = results.all_graph_feature_acts.cpu().numpy()

    # Limit the number of examples if there are too many
    n_examples = min(feature_activations.shape[0], max_examples)
    feature_activations = feature_activations[:n_examples]

    # Calculate char_count for each example
    char_counts = pca_df["context"].iloc[:n_examples].apply(process_context)

    # Remove examples with None char_count
    valid_indices = char_counts.notna()
    feature_activations = feature_activations[valid_indices]
    char_counts = char_counts[valid_indices]

    # Create a DataFrame with char_counts and feature activations
    df = pd.DataFrame(feature_activations, columns=fs_splitting_nodes)
    df["char_count"] = char_counts.values

    # Group by char_count and calculate mean activations
    grouped = df.groupby("char_count").mean().reset_index()
    grouped = grouped.sort_values("char_count")

    # Normalize activations to sum to 1 for each char_count
    activation_columns = grouped.columns.drop("char_count")
    grouped[activation_columns] = grouped[activation_columns].div(
        grouped[activation_columns].sum(axis=1), axis=0
    )

    # Create area chart
    fig = go.Figure()

    for feature in fs_splitting_nodes:
        fig.add_trace(
            go.Scatter(
                x=grouped["char_count"],
                y=grouped[feature],
                mode="lines",
                line=dict(width=0.5),
                stackgroup="one",
                groupnorm="fraction",
                name=f"Feature {feature}",
                hoverinfo="text",
                text=[
                    f"Feature: {feature}<br>Char Count: {count}<br>Normalized Activation: {act:.4f}"
                    for count, act in zip(grouped["char_count"], grouped[feature])
                ],
            )
        )

    # Update layout
    fig.update_layout(
        title=f"Normalized Feature Activation by Character Count - Cluster {fs_splitting_cluster}",
        xaxis_title="Character Count",
        yaxis_title="Proportion of Feature Activation",
        width=1200,
        height=800,
        legend_title="Features",
        hovermode="closest",
        showlegend=True,
        yaxis=dict(tickformat=".0%"),  # Format y-axis as percentages
    )

    # Show the plot
    if save:
        # Save as PNG
        png_path = os.path.join(
            pca_path,
            f"feature_activation_normalized_area_chart_{fs_splitting_cluster}.png",
        )
        fig.write_image(png_path, scale=4.0)

        svg_path = os.path.join(
            pca_path,
            f"feature_activation_normalized_area_chart_{fs_splitting_cluster}.svg",
        )
        fig.write_image(svg_path)

        # Save as HTML
        html_path = os.path.join(
            pca_path,
            f"feature_activation_normalized_area_chart_{fs_splitting_cluster}.html",
        )
        fig.write_html(html_path)
    else:
        fig.show()
    return fig


def plot_pca_domain(pca_df, pca_path, fs_splitting_cluster, save_figs=False):
    # Define colors for each category
    color_map = {
        "twitter": "#1DA1F2",  # Twitter blue
        "usat": "#FF0000",  # Red for USA Today
        "youtube": "#00FF00",  # YouTube red
        "other": "#CCCCCC",  # Gray for others
    }

    # Function to determine color
    def get_color(row):
        context = row["context"].lower()
        if "twitter" in context or "t.co" in context:
            return color_map["twitter"]
        elif "usat" in context:
            return color_map["usat"]
        elif "watch?v=" in context:
            return color_map["youtube"]
        else:
            return color_map["other"]

    # Apply the function to get colors
    pca_df["color"] = pca_df.apply(get_color, axis=1)

    # Create the plot
    fig = go.Figure()

    # Add traces for colors (categories)
    for category, color in color_map.items():
        df_category = pca_df[pca_df["color"] == color]
        fig.add_trace(
            go.Scatter(
                x=df_category["PC2"],
                y=df_category["PC3"],
                mode="markers",
                marker=dict(color=color, size=8),
                name=category.capitalize(),
                text=[
                    f"Token: {t}<br>Context: {c}"
                    for t, c in zip(df_category["tokens"], df_category["context"])
                ],
                hoverinfo="text",
            )
        )

    # Update layout
    fig.update_layout(
        height=800,
        width=800,
        title_text=f"PCA Analysis - Cluster {fs_splitting_cluster} (Context Categories)",
        xaxis_title="PC2",
        yaxis_title="PC3",
        legend_title_text="Context Category",
    )

    fig.update_traces(
        marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )

    if save_figs:
        # Save as PNG
        png_path = os.path.join(
            pca_path, f"pca_plot_context_{fs_splitting_cluster}.png"
        )
        fig.write_image(png_path, scale=3.0)

        # Save as HTML
        html_path = os.path.join(
            pca_path, f"pca_plot_context_{fs_splitting_cluster}.html"
        )
        fig.write_html(html_path)
    else:
        fig.show()

    return fig
