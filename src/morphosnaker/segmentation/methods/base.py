# src/morphosnaker/segmentation/methods/base.py

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union


@dataclass
class SegmentationConfigBase:
    method: str
    segmentation_mode: str
    trained_model_name: str = "my_segmentation_model"
    result_dir: str = "./segmentation_results"
    fig_dir: str = "figures"
    channel: Optional[Union[int, Tuple[int]]] = 0

    def __post_init__(self):
        self.fig_dir = os.path.join(self.result_dir, self.fig_dir)
        self.validate()

    def validate(self):
        self._validate_segmentation_mode()

    def _validate_segmentation_mode(self):
        if self.segmentation_mode not in ["2D", "3D"]:
            raise ValueError(
                "segmentation_mode must be either '2D' or '3D', got:"
                f"{self.segmentation_mode}"
            )


class SegmentationMethodBase(ABC):
    @abstractmethod
    def train(self, images: Any, masks: Optional[Any], **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def predict(self, image: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        pass
