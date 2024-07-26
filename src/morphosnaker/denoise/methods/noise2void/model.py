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
                "For 2D denoising, n2v_patch_shape must be 2D. "
                f"Got: {patch_shape}"
            )
        elif self.config.denoising_mode == "3D" and len(patch_shape) != 3:
            raise ValueError(
                "For 3D denoising, n2v_patch_shape must be 3D. "
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
        return history, non_zero_patch_training, non_zero_patch_validation

    def train_2D(self, images):
        self.config.denoising_mode = "2D"
        history, non_zero_patch_training, non_zero_patch_validation = (
            self.train(images)
        )
        self._save_training_processes(
            history,
            non_zero_patch_training,
            non_zero_patch_validation,
        )
        return history

    def train_3D(self, images):
        self.config.denoising_mode = "3D"
        # patch dimensions are handled here, so we can plot them in 2D
        history, non_zero_patch_training, non_zero_patch_validation = (
            self.train(images)
        )
        z_mid = non_zero_patch_training.shape[0] // 2
        non_zero_patch_training = non_zero_patch_training[z_mid, ...]
        non_zero_patch_validation = non_zero_patch_validation[z_mid, ...]

        self._save_training_processes(
            history,
            non_zero_patch_training,
            non_zero_patch_validation,
        )
        return history

    def predict(self, images):
        if self._noise2void is None:
            raise ValueError(
                "Model not trained or loaded. "
                "Please train or loada model first."
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
        print(f"Image 1 shape: {image1.shape}, Image 2 shape: {image2.shape}")
        os.makedirs(self.config.fig_dir, exist_ok=True)

        # Determine if we're dealing with 2D or 3D images
        if image1.ndim == 3:  # 2D image
            z1, z2 = image1[..., 0], image2[..., 0]
        elif image1.ndim == 4:  # 3D image
            mid_z = image1.shape[1] // 2
            z1, z2 = image1[0, mid_z, ..., 0], image2[0, mid_z, ..., 0]
        else:
            raise ValueError("Unsupported image dimensions")

        # Create subplots
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(title1, title2),
            shared_yaxes=True,
            horizontal_spacing=0.05,
        )

        # Add heatmaps
        for i, (z, title) in enumerate([(z1, title1), (z2, title2)], 1):
            fig.add_trace(
                go.Heatmap(
                    z=z,
                    colorscale="Magma",
                    showscale=i
                    == 2,  # Only show colorbar for the second image
                    colorbar=dict(title="Intensity", x=1.05, y=0.5),
                    hoverinfo="none",
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
                range=[0, max(z1.shape[1], z2.shape[1])],
                row=1,
                col=i,
            )
            fig.update_yaxes(
                title_text=f"Y ({y_unit})" if y_unit and i == 1 else "",
                showgrid=False,
                zeroline=False,
                range=[max(z1.shape[0], z2.shape[0]), 0],  # Reverse y-axis
                row=1,
                col=i,
                scaleanchor="x" if i == 1 else "x",
                scaleratio=1,
            )

        # Explicitly link the second subplot to the first
        fig.update_xaxes(scaleanchor="x", scaleratio=1, row=1, col=2)
        fig.update_yaxes(scaleanchor="y", scaleratio=1, row=1, col=2)
        # Wrap metadata text
        wrapped_metadata = self.wrap_text(metadata_text, width=100)

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

        # Save the figure as PNG and SVG
        for ext in ["png", "svg"]:
            filename = os.path.join(
                self.config.fig_dir, f"{filename_base}.{ext}"
            )
            fig.write_image(
                filename, scale=2
            )  # Increased scale for better resolution
            print(
                colored(
                    f"Comparison plot saved as {ext.upper()} to {filename}",
                    "green",
                )
            )

        return (
            fig  # Return the figure object for further manipulation if needed
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
            f" the shape of input slice is: {input_slice.shape} and"
            f"the shape of prediction slice is: {prediction_slice.shape}"
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
