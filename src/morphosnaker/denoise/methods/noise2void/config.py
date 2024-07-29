# src/morphosnaker/denoise/methods/noise2void/config.py

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from morphosnaker.denoise.methods.base import DenoiseConfigBase


@dataclass
class Noise2VoidConfig(DenoiseConfigBase):
    """
    Configuration class for the Noise2Void denoising method.

    This class extends DenoiseConfigBase with Noise2Void-specific parameters.

    Attributes:
        method (str): The denoising method, set to "n2v" for Noise2Void.
        denoising_mode (str): The dimensionality of denoising, either "2D" or
        "3D".
        trained_model_name (str): Name of the trained model.
        train_steps_per_epoch (int): Number of training steps per epoch.
        train_epochs (int): Total number of training epochs.
        train_loss (str): Loss function for training.
        batch_norm (bool): Whether to use batch normalization.
        train_batch_size (int): Batch size for training.
        result_dir (str): Directory to save results.
        fig_dir (str): Directory to save figures.
        n2v_patch_shape (Tuple[int, ...]): Shape of patches for Noise2Void.
        training_patch_fraction (float): Fraction of patches to use for
        training.
        unet_kern_size (int): Kernel size for the U-Net architecture.
        n2v_perc_pix (float): Percentage of pixels to manipulate.
        n2v_manipulator (str): Pixel manipulation strategy.
        n2v_neighborhood_radius (int): Radius of the neighborhood for pixel
        manipulation.
        structN2Vmask (Optional[List[List[int]]]): Structured mask for N2V.
        save_weights_only (bool): Whether to save only the model weights.
        channel (Optional[int]): Channel to use for denoising.
        author (str): Author of the model.
        tile_overlap (int): Overlap between tiles for prediction.
        tile_shape (Tuple[int, int]): Shape of tiles for prediction.
        input_dims (str): Input dimensions of the data.
    """

    method: str = "n2v"
    denoising_mode: str = "2D"
    trained_model_name: str = "my_model"
    train_steps_per_epoch: int = 100
    train_epochs: int = 100
    train_loss: str = "mse"
    batch_norm: bool = True
    train_batch_size: int = 128
    result_dir: str = "./results"
    fig_dir: str = "figures"
    n2v_patch_shape: Tuple[int, ...] = (64, 64)
    training_patch_fraction: float = 0.8
    unet_kern_size: int = 3
    n2v_perc_pix: float = 0.198
    n2v_manipulator: str = "uniform_withCP"
    n2v_neighborhood_radius: int = 5
    structN2Vmask: Optional[List[List[int]]] = None
    save_weights_only: bool = False
    channel: Optional[int] = None
    author: str = "morphosnaker"
    tile_overlap: int = 16
    tile_shape: Tuple[int, int] = (64, 64)
    input_dims: str = "YX"

    def __post_init__(self):
        self.fig_dir = os.path.join(self.result_dir, self.fig_dir)
        self.validate()

    def validate(self) -> None:
        """
        Validate the configuration parameters.

        This method checks the validity of various configuration parameters and
        raises
        ValueError if any parameter is invalid.
        """
        super().validate()
        self._validate_n2v_perc_pix()
        self._validate_n2v_patch_shape()
        # self._validate_tile_shape()
        self._validate_tile_overlap()

    def _validate_n2v_perc_pix(self) -> None:
        """Validate the n2v_perc_pix parameter."""
        if not (0 <= self.n2v_perc_pix <= 1):
            raise ValueError(
                f"n2v_perc_pix must be between 0 and 1,got:{self.n2v_perc_pix}"
            )

    def _validate_n2v_patch_shape(self) -> None:
        """Validate the n2v_patch_shape parameter.
        TODO: Fix the issue that default is 2D, when we load a model that is
        3D we must update the configs otherwise we can not validate the config
        (it throws an error because the n2v_patch_shape is a 2-tuple and not
        a 3-tuple) see commented code below"""
        if self.denoising_mode == "2D" and len(self.n2v_patch_shape) != 2:
            raise ValueError(
                f"For 2D, n2v_patch_shape must be a 2-tuple,got: {self.n2v_patch_shape}"
            )
        # if self.denoising_mode == "2D" and len(self.n2v_patch_shape) != 2:
        #     raise ValueError(
        #         "For 2D, n2v_patch_shape must be a 2-tuple,got: "
        #         f"{self.n2v_patch_shape}"
        #     )

    # def _validate_tile_shape(self):
    #     if self.denoising_mode == "2D" and len(self.tile_shape) != 2:
    #         raise ValueError(
    #             f"tile_shape must be a 2-tuple, got: {self.tile_shape}"
    #         )
    #     if self.denoising_mode == "3D" and len(self.tile_shape) != 3:
    #         raise ValueError(
    #             f"tile_shape must be a 3-tuple, got: {self.tile_shape}"
    #         )

    def _validate_tile_overlap(self) -> None:
        """Validate the tile_overlap parameter."""
        if self.tile_overlap < 0:
            raise ValueError(
                f"tile_overlap must be non-negative, got: {self.tile_overlap}"
            )
