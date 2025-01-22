# src/morphosnaker/segmentation/methods/cellpose/config.py

import os
from dataclasses import dataclass, field
from typing import Any, Tuple, Union

from ..base import SegmentationConfigBase


@dataclass
class CellposeConfig(SegmentationConfigBase):
    channels: Union[int, Tuple[int, int]] = (0, 0)
    model_type: str = "cyto3"
    diameter: float = 30.0
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    do_3D: bool = False
    pretrained_model: str = field(default="cyto3", init=False)
    segmentation_mode: str = "2D"
    GPU: bool = False

    def __post_init__(self):
        super().validate()
        self.validate()

    def validate(self):
        super().validate()
        if self.model_type not in ["cyto", "nuclei", "cyto2", "cyto3"]:
            raise ValueError(f"Invalid model_type: {self.model_type}")

    def update_from_model(self, model: Any) -> None:
        """
        Update configuration based on a loaded Cellpose model.

        :param model: A loaded Cellpose model
        """
        if hasattr(model, "pretrained_model"):
            self.pretrained_model = model.pretrained_model
        if hasattr(model, "diam_mean"):
            self.diameter = model.diam_mean
        if hasattr(model, "nclasses"):
            self.model_type = "cyto" if model.nclasses == 4 else "nuclei"
        if hasattr(model, "diam_labels"):
            self.diam_labels = model.diam_labels

    def update_model_name(self, path: str) -> None:
        """
        Update the trained model name based on the loaded model path.

        :param path: Path to the loaded model
        """
        self.trained_model_name = os.path.basename(path)
