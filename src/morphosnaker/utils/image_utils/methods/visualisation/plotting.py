import os
import textwrap
import webbrowser
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from IPython.display import display
from plotly.subplots import make_subplots
from termcolor import colored


class PlotMethod:
    """Plot TCZYX images"""

    def _wrap_text(self, text, width=80):
        """Wrap text to fit within a specified width."""
        return "<br>".join(textwrap.wrap(text, width=width))

    def plot_2_images(
        self,
        image_1: np.ndarray,
        image_2: np.ndarray,
        title_image_1: str,
        title_image_2: str,
        main_title: str,
        x_unit: str = "",
        y_unit: str = "",
        metadata_text: str = "metadata: no metadata",
    ) -> go.Figure:

        # Select middle T, C, Z for visualization if they exist
        t_slice = image_1.shape[0] // 2 if image_1.shape[0] > 1 else 0
        c_slice = image_1.shape[1] // 2 if image_1.shape[1] > 1 else 0
        z_slice = image_1.shape[2] // 2 if image_1.shape[2] > 1 else 0

        image_1_slice = image_1[t_slice, c_slice, z_slice, :, :]
        image_2_slice = image_2[t_slice, c_slice, z_slice, :, :]
        # Create subplots
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(title_image_1, title_image_2),
            shared_yaxes=True,
            horizontal_spacing=0.05,
        )
        global_min = min(image_1_slice.min(), image_2_slice.min())
        global_max = max(image_1_slice.max(), image_2_slice.max())
        # Add heatmaps
        for i, (z, title) in enumerate(
            [(image_1_slice, title_image_1), (image_2_slice, title_image_2)], 1
        ):
            fig.add_trace(
                go.Heatmap(
                    z=z,
                    colorscale="Magma",
                    showscale=i == 2,  # Only show colorbar for the second image
                    colorbar=dict(title="Intensity", x=1.05, y=0.5),
                    hoverinfo="none",
                    zmin=global_min,
                    zmax=global_max,
                ),
                row=1,
                col=i,
            )

            # Update axes
        for i in [1, 2]:
            fig.update_xaxes(
                title_text=f"X ({x_unit})" if x_unit else "X",
                showgrid=False,
                zeroline=False,
                range=[0, z.shape[1]],
                row=1,
                col=i,
            )
            fig.update_yaxes(
                title_text=f"Y ({y_unit})" if y_unit and i == 1 else "",
                showgrid=False,
                zeroline=False,
                range=[z.shape[0], 0],  # Reverse y-axis
                row=1,
                col=i,
            )

        # Explicitly link the second subplot to the first
        fig.update_xaxes(scaleanchor="x", scaleratio=1, row=1, col=2)
        fig.update_yaxes(scaleanchor="y", scaleratio=1, row=1, col=2)
        # Wrap metadata text
        wrapped_metadata = self._wrap_text(metadata_text, width=100)

        # Update layout
        fig.update_layout(
            title=dict(
                text=main_title,
                x=0.5,
                y=0.95,
                xanchor="center",
                yanchor="top",
                font=dict(size=24, color="black", family="Arial, sans-serif"),
            ),
            height=700,  # Increased height to accommodate wrapped text
            width=1200,
            autosize=False,
            margin=dict(t=100, b=200, l=100, r=100),  # Increased bottom margin
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="white",
        )

        # Add metadata annotation with wrapped text
        fig.add_annotation(
            x=0.5,
            y=-0.3,  # Adjusted y position
            xref="paper",
            yref="paper",
            text=wrapped_metadata,
            showarrow=False,
            bordercolor="black",
            borderwidth=1,
            borderpad=8,
            bgcolor="white",
            opacity=0.8,
            align="left",
            font=dict(family="monospace", size=10),
        )
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
        )
        return fig  # Return the figure object for further manipulation if needed

    def show_plot(self, fig: go.Figure, filename: Optional[str] = None) -> None:
        """
        Display the plot in a Jupyter notebook or open it in a web browser.

        Args:
            fig (go.Figure): The plotly figure to display.
            filename (Optional[str]): If provided, save the plot as an HTML file with
            this name.
        """
        try:
            # Try to display in Jupyter notebook
            display(fig)
        except NameError:
            # If not in a Jupyter environment, save as HTML and open in a web browser
            if filename is None:
                filename = "temp_plot.html"

            # Save the figure as an HTML file
            fig.write_html(filename)

            # Open the HTML file in the default web browser
            webbrowser.open(f"file://{os.path.abspath(filename)}", new=2)

            print(f"Plot opened in web browser. Saved as {filename}")
        except Exception as e:
            # Handle any other unexpected exceptions
            print(f"An error occurred while trying to display the plot: {str(e)}")

    def save_plot(
        self,
        fig: go.Figure,
        file_name_base: str,
        result_dir="./results",
        fig_dir="figures",
    ) -> None:
        full_fig_dir = os.path.join(result_dir, fig_dir)
        # Ensure output directory exists
        os.makedirs(full_fig_dir, exist_ok=True)

        for ext in ["png", "svg"]:
            if fig_dir is None:
                raise ValueError("Please set the figure directory before saving plots")
            filename = os.path.join(fig_dir, f"{file_name_base}.{ext}")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            fig.write_image(filename, scale=2)  # Increased scale for better resolution
            print(colored(f"Plot saved as {ext.upper()} to {filename}", "green"))
