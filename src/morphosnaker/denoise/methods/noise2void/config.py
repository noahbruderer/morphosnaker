# src/morphosnaker/denoise/methods/noise2void/config.py

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from morphosnaker.denoise.methods.base import DenoiseConfigBase


@dataclass
class Noise2VoidConfig(DenoiseConfigBase):
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

    def validate(self):
        super().validate()
        self._validate_n2v_perc_pix()
        self._validate_n2v_patch_shape()
        self._validate_tile_shape()
        self._validate_tile_overlap()

    def _validate_n2v_perc_pix(self):
        if not (0 <= self.n2v_perc_pix <= 1):
            raise ValueError(
                "n2v_perc_pix must be between 0 and 1,"
                f"got: {self.n2v_perc_pix}"
            )

    def _validate_n2v_patch_shape(self):
        if self.denoising_mode == "2D" and len(self.n2v_patch_shape) != 2:
            raise ValueError(
                "For 2D, n2v_patch_shape must be a 2-tuple,"
                f"got: {self.n2v_patch_shape}"
            )
        if self.denoising_mode == "3D" and len(self.n2v_patch_shape) != 3:
            raise ValueError(
                "For 3D, n2v_patch_shape must be a 3-tuple,"
                f"got: {self.n2v_patch_shape}"
            )

    def _validate_tile_shape(self):
        if len(self.tile_shape) != 2:
            raise ValueError(
                f"tile_shape must be a 2-tuple, got: {self.tile_shape}"
            )

    def _validate_tile_overlap(self):
        if self.tile_overlap < 0:
            raise ValueError(
                f"tile_overlap must be non-negative, got: {self.tile_overlap}"
            )
