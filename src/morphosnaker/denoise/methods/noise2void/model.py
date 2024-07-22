# src/morphosnaker/denoising/methods/noise2void.py

from morphosnaker.denoise.methods.base import DenoiseTrainBase, DenoisePredictBase
from .config import Noise2VoidPredictionConfig, Noise2VoidTrainingConfig
from n2v.models import N2V, N2VConfig
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
import numpy as np
import os
from typing import Union
from termcolor import colored
import tifffile
import matplotlib.pyplot as plt
import os 
import json

class Noise2VoidTrain(DenoiseTrainBase):
    def __init__(self):
        self.datagen = N2V_DataGenerator()
        self.model = None
        self.config = None

    def configure(self, config: Noise2VoidTrainingConfig):
        self.config = config
        
    def validate_input(self, images):
        """TO DO: add docstring"""
        if not self.config:
            raise ValueError("Training or Denoise configuration is required for training")
        #images must be a list for n2v
        if not isinstance(images, list):
            images = [images]
        return images

    def train_2D(self, images, **kwargs):
        """Input: image (np.ndarray) shape (x,y,channel) ONE channel at the time, you have to split the channels before
        For now this is the easiest way of splitting up the modules, later another method should be added to the class that will use train_2D to 
        process multiple channels at once
        TODO: if someone wants to input a stack of 2D images and train on the stack we have to implement this!
        -> make it so that the images_to_process do NOT add 
        
        """
        images = self.validate_input(images)
        #check if images have right dimesnions for 2D here
        patches = self.datagen.generate_patches_from_list(images,
                                                          shape=self.config.n2v_patch_shape, 
                                                          shuffle = True)
        split_idx = int(len(patches) * self.config.training_patch_fraction)
        training_patches = patches[:split_idx]
        validation_patches = patches[split_idx:]
        print(colored(f"Number of training patches: {len(training_patches)}", "green"))
        print(colored(f"Number of validation patches: {len(validation_patches)}", "green"))

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
            structN2Vmask=self.config.structN2Vmask
        )

        model_name = f"{self.config.trained_model_name}"
        self.model = N2V(n2v_config, model_name, basedir=self.config.result_dir)
        history = self.model.train(training_patches, validation_patches)
        
    # Save training processes
        train_patch = self._select_non_zero_patch(training_patches)
        val_patch = self._select_non_zero_patch(validation_patches)        
        self._save_training_processes(history, train_patch, val_patch, image=images[0])       
        return


    def train_3D(self, images, **kwargs):
        """
        Train the Noise2Void model on 3D images.
        
        Args:
            images (list): List of 3D images to train on.
            **kwargs: Additional arguments for training.
        """
        images = self.validate_input(images)
        
        # Generate patches
        patches = self.datagen.generate_patches_from_list(images, shape=self.config.n2v_patch_shape)
        
        # Split patches into training and validation sets
        split_idx = int(len(patches) * self.config.training_patch_fraction)
        X = patches[:split_idx]
        X_val = patches[split_idx:]
        
        print(f"Generated patches: {patches.shape}")
        print(f"Training patches: {X.shape}")
        print(f"Validation patches: {X_val.shape}")

        # Create N2V config
        n2v_config = N2VConfig(
            X,
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
            structN2Vmask=self.config.structN2Vmask
        )

        # Create and train the model
        model_name = self.config.trained_model_name
        self.model = N2V(config=n2v_config, 
                         name=model_name, 
                         basedir=self.config.result_dir)
        history = self.model.train(X, X_val)

        # Save training processes
        train_patch = self._select_non_zero_patch(X)
        val_patch = self._select_non_zero_patch(X_val)
        self._save_training_processes(history, train_patch, val_patch, image=images[0])

        return history

    def _save_training_processes(self, history, train_patch, val_patch, image):
        # Ensure directories exist
        os.makedirs(self.config.fig_dir, exist_ok=True)

        # Plot and save training history
        plt.figure(figsize=(16, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(os.path.join(self.config.fig_dir, f"{self.config.trained_model_name}_training_history.png"))
        plt.close()

        # Plot and save sample patches
        plt.figure(figsize=(14, 7))
        if self.config.denoising_mode == '2D':
            plt.subplot(1, 2, 1)
            plt.imshow(train_patch[..., 0], cmap='magma')
            plt.title('Training Patch')
            plt.subplot(1, 2, 2)
            plt.imshow(val_patch[..., 0], cmap='magma')
            plt.title('Validation Patch')
        elif self.config.denoising_mode == '3D':
            print(train_patch.shape)
            mid_z = train_patch.shape[1] // 2
            plt.subplot(1, 2, 1)
            plt.imshow(train_patch[0, :, :, 0], cmap='magma')
            plt.title('Training Patch (Middle Slice)')
            plt.subplot(1, 2, 2)
            plt.imshow(val_patch[0, :, :, 0], cmap='magma')
            plt.title('Validation Patch (Middle Slice)')
        else:
            raise ValueError(f"Unsupported denoising mode: {self.config.denoising_mode}")

        plt.savefig(os.path.join(self.config.fig_dir, f"{self.config.trained_model_name}_sample_patches.png"))
        plt.close()

        # Prepare export image
        if self.config.denoising_mode == '2D':
            export_image = image[0, ..., 0]
        elif self.config.denoising_mode == '3D':
            export_image = image[0, :, :, :, 0]  # Assuming image is already in the correct format
        else:
            raise ValueError(f"Unsupported denoising mode: {self.config.denoising_mode}")

        self._save_model(export_image)
        print(colored(f"Training processes saved in {self.config.fig_dir} and {self.config.result_dir}", "green"))
        
    def _save_model(self, export_image):
        # Define the export image        
        # Save the model
        model_path = f"{self.config.trained_model_name}.h5"
        # self.model.save(model_path)
        
        # Additional information can be saved separately if needed
        metadata = {
            "name": f"{self.config.trained_model_name}",
            "description": f"Noise2Void model trained with {self.config.denoising_mode} mode",
            "authors": [self.config.author],
            "test_img_shape": export_image.shape,
            "axes": 'YXC' if self.config.denoising_mode == '2D' else 'ZYXC',
            "patch_shape": self.config.n2v_patch_shape
        }
        
        # Save metadata to a JSON file
        metadata_name = f"{self.config.trained_model_name}_metadata.json"
        metadata_path = os.path.join(self.config.result_dir, metadata_name)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        print(f"Model and metadata saved to {model_path} and {metadata_name} respectively.")
        
        
    def _select_non_zero_patch(self, patches):
        """Select a non-zero patch from a batch of patches."""
        for patch in patches:
            if np.any(patch):
                return patch
        raise ValueError("All patches are zero. Cannot select a non-zero patch.")
    

class Noise2VoidPredict(DenoisePredictBase):
    def __init__(self):
        self.config = None
                
    def configure(self, config: Noise2VoidPredictionConfig):
        self.config = config
        
    def load_model(self, path):
        self.model = N2V(config=None, 
                         name=self.config.trained_model_name, 
                         basedir=path)
        
    def denoise(self, image):
        if self.model is None:
            raise ValueError("Model is None. Please provide a model. You can train a model before denoising.")
        
        if self.config.denoising_mode == '2D':
            axes = 'TYXC'
        elif self.config.denoising_mode == '3D':
            axes = 'TZYXC'
        
        prediction = self.model.predict(image, axes=axes)
        
        # Save the prediction results
        self._save_prediction_results(image, prediction)
        
        return prediction
    
    def _save_prediction_results(self, input_image, prediction):
        # Ensure directories exist
        os.makedirs(self.config.fig_dir, exist_ok=True)
        os.makedirs(self.config.result_dir, exist_ok=True)
        tifffile.imsave(os.path.join(self.config.result_dir, f"{self.config.trained_model_name}_prediction.tif"), prediction)

        # Create annotations with basic configs
        annotations = (
            f"{'Model:':<25} {self.config.trained_model_name}\n"
            f"{'Denoise dimension:':<25} {self.config.denoising_mode}\n"
            f"{'Denoising Model:':<25} {self.config.denoising_mode}\n"
            f"{'Training loss:':<25} {self.model.config.train_loss}\n"
            f"{'Training epochs:':<25} {self.model.config.train_epochs}\n"
            f"{'Training steps/epoch:':<25} {self.model.config.train_steps_per_epoch}\n"
            f"{'Patch sizes:':<25} {self.model.config.n2v_patch_shape}\n"
            f"{'Train batch Size:':<25} {self.model.config.train_batch_size}\n"
            f"{'N2V Tile Shape:':<25} {self.config.tile_shape}"
        )

        # Create main title
        main_title = f"Noise2Void Denoising Results: {self.config.trained_model_name}"

        # Plot and save input vs prediction
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))

        # Add annotations and main title as text
        fig.text(0.1, 0, annotations, fontsize=10, verticalalignment='top', fontfamily='monospace')
        fig.text(0.5, 0.88, main_title, fontsize=16, horizontalalignment='center', fontweight='bold')

        if self.config.denoising_mode == '2D':
            ax1.imshow(input_image[0, ..., 0], cmap="magma")
            ax1.set_title('Input')
            ax1.axis('off')

            ax2.imshow(prediction[0, ..., 0], cmap="magma")
            ax2.set_title('Prediction')
            ax2.axis('off')
        elif self.config.denoising_mode == '3D':

            mid_z = input_image.shape[1] // 2
            ax1.imshow(input_image[0, mid_z, ..., 0], cmap="magma")
            ax1.set_title('Input (Middle Slice)')
            ax1.axis('off')

            ax2.imshow(prediction[0, mid_z, ..., 0], cmap="magma")
            ax2.set_title('Prediction (Middle Slice)')
            ax2.axis('off')
        else:
            raise ValueError(f"Unsupported denoising mode: {self.config.denoising_mode}")

        plt.tight_layout(rect=[0, 0, 1, 0.85])  # Adjust layout to make room for annotations and title
        plt.savefig(os.path.join(self.config.fig_dir, f"{self.config.trained_model_name}_input_vs_prediction.png"), dpi=300, bbox_inches='tight')
        plt.close()

        print(colored(f"Prediction result snapshot saved in {self.config.fig_dir}", "green"))