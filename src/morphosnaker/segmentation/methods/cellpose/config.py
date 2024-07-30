# src/morphosnaker/segmentation/methods/cellpose/config.py

from dataclasses import dataclass
from typing import Tuple, Union

from ..base import SegmentationConfigBase


@dataclass
class CellposeConfig(SegmentationConfigBase):
    method: str = "cellpose"
    segmentation_mode: str = "2D"
    channels: Union[int, Tuple[int, int]] = (0, 0)
    model_type: str = "cyto3"
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    min_size: int = 15
    stitch_threshold: float = 0.0
    do_3D: bool = False
    diameter: float = 30.0

    def validate(self):
        super().validate()
        self._validate_model_type()
        self._validate_channels()

    def __post_init__(self):
        # Convert single channel to tuple
        if isinstance(self.channels, int):
            self.channels = (self.channels, self.channels)

        # Now call the validation
        self.validate()

    def _validate_model_type(self):
        valid_models = ["cyto", "nuclei", "cyto2", "cyto3"]
        if self.model_type not in valid_models:
            raise ValueError(
                f"model_type must be one of {valid_models}, got: {self.model_type}"
            )

    def _validate_channels(self):
        if not isinstance(self.channels, tuple) or len(self.channels) != 2:
            raise ValueError(
                f"channels must be a tuple of two integers, got: {self.channels}"
            )
