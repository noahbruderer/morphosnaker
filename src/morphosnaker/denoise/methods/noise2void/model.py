import json
import os
import textwrap

import numpy as np
import plotly.graph_objects as go
import tifffile
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from n2v.models import N2V, N2VConfig
from plotly.subplots import make_subplots
from termcolor import colored

from .config import Noise2VoidConfig


class Noise2VoidModel:
    def __init__(self, config: Noise2VoidConfig):
        self.config = config
        self.datagen = N2V_DataGenerator()

    def _initialize_model(self, training_patches):
        n2v_config = N2VConfig(
            training_patches,
            unet_kern_size=self.config.unet_kern_size,
            train_steps_per_epoch=self.config.train_steps_per_epoch,
            train_epochs=self.config.train_epochs,
            train_loss=self.config.train_loss,
            batch_norm=self.config.batch_norm,
            train_batch_size=self.config.train_batch_size,
            n2v_perc_pix=self.config.n2v_perc_pix,
            n2v_patch_shape=self.config.n2v_patch_shape,
            n2v_manipulator=self.config.n2v_manipulator,
            n2v_neighborhood_radius=self.config.n2v_neighborhood_radius,
            structN2Vmask=self.config.structN2Vmask,
        )
        self._noise2void = N2V(
            n2v_config,
            self.config.trained_model_name,
            basedir=self.config.result_dir,
        )

    def _generate_patches(self, images):
        images = self._validate_input(images)
        patch_shape = self.config.n2v_patch_shape
        if self.config.denoising_mode == "2D" and len(patch_shape) != 2:
            raise ValueError(
                "For 2D denoising, n2v_patch_shape must be 2D."
                f"Got: {patch_shape}"
            )
        elif self.config.denoising_mode == "3D" and len(patch_shape) != 3:
            raise ValueError(
                "For 3D denoising, n2v_patch_shape must be 3D."
                f"Got: {patch_shape}"
            )

        patches = self.datagen.generate_patches_from_list(
            images, shape=patch_shape, shuffle=True
        )
        split_idx = int(len(patches) * self.config.training_patch_fraction)
        training_patches = patches[:split_idx]
        validation_patches = patches[split_idx:]

        print(
            colored(
                f"Number of training patches: {len(training_patches)}", "green"
            )
        )
        print(
            colored(
                f"Number of validation patches: {len(validation_patches)}",
                "green",
            )
        )

        return training_patches, validation_patches

    def train(self, images):
        training_patches, validation_patches = self._generate_patches(images)
        self._initialize_model(training_patches)

        history = self._noise2void.train(training_patches, validation_patches)
        non_zero_patch_training = self._select_non_zero_patch(training_patches)
        non_zero_patch_validation = self._select_non_zero_patch(
            validation_patches
        )
        self._save_training_processes(
            history,
            non_zero_patch_training,
            non_zero_patch_validation,
        )
        return history

    def train_2D(self, images):
        self.config.denoising_mode = "2D"
        return self.train(images)

    def train_3D(self, images):
        self.config.denoising_mode = "3D"
        return self.train(images)

    def predict(self, images):
        if self._noise2void is None:
            raise ValueError(
                "Model not trained or loaded. Please train or load"
                "a model first."
            )
        axes = "TYXC" if self.config.denoising_mode == "2D" else "TZYXC"
        prediction = self._noise2void.predict(images, axes=axes)
        self._save_prediction(images, prediction)
        return prediction

    def load(self, path):
        self._noise2void = N2V(
            config=None,
            name=self.config.trained_model_name,
            basedir=path,
        )

    def load_metadata(self, path):
        metadata_name = f"{self.config.trained_model_name}_metadata.json"
        metadata_path = os.path.join(os.path.dirname(path), metadata_name)
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            print(f"Loaded metadata for model: {metadata['name']}")
            return metadata
        else:
            print(f"No metadata found at {metadata_path}")
            return None

    def _validate_input(self, images):
        if not isinstance(images, list):
            images = [images]
        return images

    def _patch_generator(self, images):
        patch_shape = self.config.n2v_patch_shape
        if self.config.denoising_mode == "2D" and len(patch_shape) != 2:
            raise ValueError(
                f"For 2D denoising, n2v_patch_shape must be 2D."
                f"Got: {patch_shape}"
            )
        elif self.config.denoising_mode == "3D" and len(patch_shape) != 3:
            raise ValueError(
                f"For 3D denoising, n2v_patch_shape must be 3D."
                f"Got: {patch_shape}"
            )

        patches = self.datagen.generate_patches_from_list(
            images, shape=patch_shape, shuffle=True
        )
        split_idx = int(len(patches) * self.config.training_patch_fraction)
        training_patches = patches[:split_idx]
        validation_patches = patches[split_idx:]
        print(
            colored(
                f"Number of training patches: {len(training_patches)}", "green"
            )
        )
        print(
            colored(
                f"Number of validation patches: {len(validation_patches)}",
                "green",
            )
        )
        return training_patches, validation_patches

    def _select_non_zero_patch(self, patches):
        """Select a non-zero patch from a batch of patches."""
        for patch in patches:
            if np.any(patch):
                return patch
        raise ValueError(
            "All patches are zero. Cannot select a non-zero patch."
        )

    # saving

    def wrap_text(self, text, width=80):
        """Wrap text to fit within a specified width."""
        return "<br>".join(textwrap.wrap(text, width=width))

    def _plot_and_save_comparison(
        self,
        image1,
        image2,
        title1,
        title2,
        main_title,
        metadata_text,
        filename_base,
        x_unit="",
        y_unit="",
    ):
        print(f"patch size is {image1.shape}")
        os.makedirs(self.config.fig_dir, exist_ok=True)
        # Determine if we're dealing with 2D or 3D images
        if image1.ndim == 3:  # 2D image
            z1, z2 = image1[..., 0], image2[..., 0]
        elif image1.ndim == 4:  # 3D image
            mid_z = image1.shape[1] // 2
            z1, z2 = image1[0, mid_z, ..., 0], image2[0, mid_z, ..., 0]
        else:
            raise ValueError("Unsupported image dimensions")
        # Create subplots with correct aspect ratio
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(title1, title2),
            column_widths=[0.5, 0.5],
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}]],
        )
        fig.add_trace(
            go.Heatmap(z=z1, colorscale="Magma", showscale=False), row=1, col=1
        )
        fig.add_trace(
            go.Heatmap(z=z2, colorscale="Magma", showscale=False), row=1, col=2
        )
        # Wrap metadata text
        wrapped_metadata = self.wrap_text(metadata_text, width=100)
        # Calculate the aspect ratio of the images
        aspect_ratio = z1.shape[0] / z1.shape[1]
        # Set the figure size maintaining the aspect ratio
        width = 1000
        height = (
            int(width * aspect_ratio * 0.5) + 300
        )  # Add extra space for metadata
        # Update layout with white background for PNGs
        fig.update_layout(
            title_text=main_title,
            width=width,
            height=height,
            showlegend=False,
            margin=dict(t=50, b=200, l=50, r=50),
            plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot area background
            paper_bgcolor="white",  # White paper background
        )
        # Update axes to maintain aspect ratio and remove background
        fig.update_xaxes(
            scaleanchor="y",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
            showline=False,  # Hide x-axis lines
        )
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
            showline=False,  # Hide y-axis lines
            range=[
                0,
                z1.shape[0],
            ],  # Explicitly set y-axis range to match data
        )
        # Add metadata annotation
        fig.add_annotation(
            x=0.5,
            y=-0.2,  # Adjusted y position
            xref="paper",
            yref="paper",
            text=wrapped_metadata,
            showarrow=False,
            bordercolor="black",
            borderwidth=1,
            borderpad=10,
            bgcolor="white",
            opacity=0.8,
            align="left",
            font=dict(family="monospace", size=10),
        )
        # Update axis labels with units
        fig.update_xaxes(title_text=f"X ({x_unit})" if x_unit else "X")
        fig.update_yaxes(title_text=f"Y ({y_unit})" if y_unit else "Y")
        # Save the figure as PNG and SVG
        for ext in ["png", "svg"]:
            filename = os.path.join(
                self.config.fig_dir, f"{filename_base}.{ext}"
            )
            fig.write_image(filename)
            print(
                colored(
                    f"Comparison plot saved as {ext.upper()} to {filename}",
                    "green",
                )
            )

    def _plot_and_save_history(self, history, common_metadata):

        # Plot and save training history using Plotly
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(y=history.history["loss"], name="Training Loss")
        )
        fig.add_trace(
            go.Scatter(y=history.history["val_loss"], name="Validation Loss")
        )
        wrapped_metadata = self.wrap_text(common_metadata, width=100)
        fig.update_layout(
            title="Model Loss",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=600,  # Adjusted height
            width=1000,
            margin=dict(t=100, b=200),  # Adjusted margins
        )

        # Add metadata annotation
        fig.add_annotation(
            x=0.5,
            y=-0.3,  # Adjusted y position
            xref="paper",
            yref="paper",
            text=wrapped_metadata,
            showarrow=False,
            bordercolor="black",
            borderwidth=1,
            borderpad=10,
            bgcolor="white",
            opacity=0.8,
            align="left",
            font=dict(family="monospace", size=10),
        )

        # Save training history as PNG and SVG
        history_base = os.path.join(
            self.config.fig_dir,
            f"{self.config.trained_model_name}_training_history",
        )
        for ext in ["png", "svg"]:
            filename = f"{history_base}.{ext}"
            fig.write_image(filename)
            print(
                colored(
                    f"Training history saved as {ext.upper()} to {filename}",
                    "green",
                )
            )

    def _save_training_processes(self, history, train_patch, val_patch):
        os.makedirs(self.config.fig_dir, exist_ok=True)

        # Create detailed metadata
        common_metadata = (
            f"Model: {self.config.trained_model_name} | "
            f"Denoising dimension: {self.config.denoising_mode} | "
            f"Training loss: {self.config.train_loss} | "
            f"Training epochs: {self.config.train_epochs} | "
            f"Training steps/epoch: {self.config.train_steps_per_epoch} | "
            f"Patch sizes: {self.config.n2v_patch_shape} | "
            f"Train batch Size: {self.config.train_batch_size}"
        )
        # Use the generic plotting function for sample patches
        patches_metadata = (
            f"{common_metadata} | "
            f"Training patches shape: {train_patch.shape} | "
            f"Validation patches shape: {val_patch.shape}"
        )

        self._plot_and_save_history(history, common_metadata)

        self._plot_and_save_comparison(
            train_patch,
            val_patch,
            "Training Patch",
            "Validation Patch",
            "Sample Patches",
            patches_metadata,
            f"{self.config.trained_model_name}_sample_patches",
            x_unit="pixels",
            y_unit="pixels",
        )

        print(
            colored(
                f"Training processes saved in {self.config.fig_dir} "
                f"and {self.config.result_dir}",
                "green",
            )
        )

    def _plot_and_save_prediction_process(self, image, prediction):
        os.makedirs(self.config.fig_dir, exist_ok=True)
        print(f"shape is {image.shape}")
        print(prediction.shape)
        if image.ndim == 4:  # 2D image TYXC
            print("in 2D")
            input_slice = image[0, ...]
            prediction_slice = prediction[0, ...]
        elif image.ndim == 5:  # 3D image TZYXC
            input_slice = image[0, ..., 0]
            prediction_slice = prediction[0, ..., 0]
        else:
            raise ValueError("Unsupported image dimensions")
        # Determine the middle slice for 3D images
        print(
            f" the shape of input slice is: {input_slice.shape} and the shape of prediction slice is: {prediction_slice.shape}"
        )
        # Create metadata
        metadata_text = (
            f"Model: {self.config.trained_model_name} | "
            f"Denoising dimension: {self.config.denoising_mode} | "
            f"Input shape: {image.shape} | "
            f"Prediction shape: {prediction.shape}"
        )

        # Use the existing comparison function to plot and save
        self._plot_and_save_comparison(
            input_slice,
            prediction_slice,
            "Input Image",
            "Predicted (Denoised) Image",
            "Denoising Result Comparison",
            metadata_text,
            f"{self.config.trained_model_name}_prediction_comparison",
            x_unit="pixels",
            y_unit="pixels",
        )

        # # Optionally, calculate and plot the difference
        # difference = prediction_slice - input_slice

        # fig = make_subplots(
        #     rows=1,
        #     cols=3,
        #     subplot_titles=("Input", "Prediction", "Difference"),
        #     specs=[
        #         [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}]
        #     ],
        # )

        # fig.add_trace(
        #     go.Heatmap(z=input_slice, colorscale="Viridis", showscale=False),
        #     row=1,
        #     col=1,
        # )
        # fig.add_trace(
        #     go.Heatmap(
        #         z=prediction_slice, colorscale="Viridis", showscale=False
        #     ),
        #     row=1,
        #     col=2,
        # )
        # fig.add_trace(
        #     go.Heatmap(z=difference, colorscale="RdBu", showscale=True),
        #     row=1,
        #     col=3,
        # )

        # fig.update_layout(
        #     title="Denoising Process Visualization",
        #     height=400,
        #     width=1200,
        #     margin=dict(t=50, b=50, l=50, r=50),
        # )

        # # Save the figure
        # for ext in ["png", "svg"]:
        #     filename = os.path.join(
        #         self.config.fig_dir,
        #         f"{self.config.trained_model_name}_denoising_process.{ext}",
        #     )
        #     fig.write_image(filename)
        #     print(
        #         colored(
        #             f"Denoising process visualization saved as {ext.upper()} to {filename}",
        #             "green",
        #         )
        #     )

    # logic to save the middle stack of the images and prediction to see the difference in 2D

    def _save_prediction(self, image, prediction):
        os.makedirs(self.config.result_dir, exist_ok=True)

        # self._plot_and_save_prediction_process(image, prediction)
        print("shape of image is ", image.shape)
        print("shape of prediction is ", prediction.shape)
        self._plot_and_save_prediction_process(
            image,
            prediction,
        )

        filename = os.path.join(
            self.config.result_dir,
            f"{self.config.trained_model_name}_prediction.tif",
        )
        tifffile.imsave(filename, prediction)
        print(colored(f"Prediction saved as TIFF to {filename}", "green"))
