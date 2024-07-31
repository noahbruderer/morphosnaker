import json
import os
import textwrap

import numpy as np
import plotly.graph_objects as go
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from n2v.models import N2V, N2VConfig
from termcolor import colored

from morphosnaker.utils import ImageProcessor

from .config import Noise2VoidConfig


class Noise2VoidModel:
    def __init__(self, config: Noise2VoidConfig):
        self.config = config
        self.datagen = N2V_DataGenerator()
        self.image_processor = ImageProcessor()

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
                f"For 2D denoising, n2v_patch_shape must be 2D. Got: {patch_shape}"
            )
        elif self.config.denoising_mode == "3D" and len(patch_shape) != 3:
            raise ValueError(
                f"For 3D denoising, n2v_patch_shape must be 3D. Got: {patch_shape}"
            )

        patches = self.datagen.generate_patches_from_list(
            images, shape=patch_shape, shuffle=True
        )
        split_idx = int(len(patches) * self.config.training_patch_fraction)
        training_patches = patches[:split_idx]
        validation_patches = patches[split_idx:]

        print(colored(f"Number of training patches: {len(training_patches)}", "green"))
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
        non_zero_patch_validation = self._select_non_zero_patch(validation_patches)
        return history, non_zero_patch_training, non_zero_patch_validation

    def train_2D(self, images):
        self.config.denoising_mode = "2D"
        history, non_zero_patch_training, non_zero_patch_validation = self.train(images)
        self._save_training_processes(
            history,
            non_zero_patch_training,
            non_zero_patch_validation,
        )
        return history

    def train_3D(self, images):
        self.config.denoising_mode = "3D"
        # patch dimensions are handled here, so we can plot them in 2D
        history, non_zero_patch_training, non_zero_patch_validation = self.train(images)
        z_mid = non_zero_patch_training.shape[0] // 2
        non_zero_patch_training = non_zero_patch_training[z_mid, ...]
        non_zero_patch_validation = non_zero_patch_validation[z_mid, ...]

        self._save_training_processes(
            history,
            non_zero_patch_training,
            non_zero_patch_validation,
        )
        return history

    def predict(self, image):
        if self._noise2void is None:
            raise ValueError(
                "Model not trained or loaded. Please train or loada model first."
            )
        axes = "TYXC" if self.config.denoising_mode == "2D" else "TZYXC"
        print(axes)
        prediction = self._noise2void.predict(image, axes=axes)
        print(f"shape of prediction is {prediction.shape}")
        # format prediction here depending on the axes
        prediction = self.image_processor.standardise_image(prediction, input_dims=axes)
        image = self.image_processor.standardise_image(image, input_dims=axes)
        print(f"shape of rearanged prediction is {prediction.shape}")

        self._save_prediction_process_plots(image, prediction)
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
                f"For 2D denoising, n2v_patch_shape must be 2D.Got:  {patch_shape}"
            )
        elif self.config.denoising_mode == "3D" and len(patch_shape) != 3:
            raise ValueError(
                f"For 3D denoising, n2v_patch_shape must be 3D.Got:  {patch_shape}"
            )

        patches = self.datagen.generate_patches_from_list(
            images, shape=patch_shape, shuffle=True
        )
        split_idx = int(len(patches) * self.config.training_patch_fraction)
        training_patches = patches[:split_idx]
        validation_patches = patches[split_idx:]
        print(colored(f"Number of training patches: {len(training_patches)}", "green"))
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
        raise ValueError("All patches are zero. Cannot select a non-zero patch.")

    # saving

    def _plot_and_save_history(self, history, common_metadata):

        # Plot and save training history using Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=history.history["loss"], name="Training Loss"))
        fig.add_trace(go.Scatter(y=history.history["val_loss"], name="Validation Loss"))
        wrapped_metadata = "<br>".join(textwrap.wrap(common_metadata, width=100))
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
        train_patch = self.image_processor.standardise_image(
            train_patch, input_dims="XYC"
        )
        val_patch = self.image_processor.standardise_image(val_patch, input_dims="XYC")
        training_patch_plot = self.image_processor.plot.plot_2_images(
            image_1=train_patch,
            image_2=val_patch,
            title_image_1="Training Patch",
            title_image_2="Validation Patch",
            main_title="Sample Patches",
            metadata_text=patches_metadata,
            x_unit="pixels",
            y_unit="pixels",
        )
        print(
            colored(
                "Saving training patches...",
                "green",
            )
        )

        self.image_processor.plot.save_plot(
            fig=training_patch_plot,
            file_name_base=f"{self.config.trained_model_name}_sample_patches",
            result_dir=self.config.result_dir,
            fig_dir=self.config.fig_dir,
        )

    def _save_prediction_process_plots(self, image, prediction):
        os.makedirs(self.config.result_dir, exist_ok=True)

        # self._plot_and_save_prediction_process(image, prediction)
        print(f"shape of image is {image.shape}")
        print(f"shape of prediction is {prediction.shape}")
        process_plot = self.image_processor.plot.plot_2_images(
            image_1=image,
            image_2=prediction,
            title_image_1="Raw Image",
            title_image_2="Denoised Image",
            main_title="Raw vs Denoised Image",
            metadata_text=f"Model used: {self.config.trained_model_name}",
            x_unit="pixels",
            y_unit="pixels",
        )
        print(f"fig_dir is {self.config.fig_dir}")
        self.image_processor.plot.save_plot(
            fig=process_plot,
            file_name_base=f"{self.config.trained_model_name}_prediction_process",
            result_dir=self.config.result_dir,
            fig_dir=self.config.fig_dir,
        )
