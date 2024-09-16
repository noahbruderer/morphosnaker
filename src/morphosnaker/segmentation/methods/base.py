# src/morphosnaker/segmentation/methods/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SegmentationConfigBase:
    method: str
    segmentation_mode: str
    trained_model_name: str = "my_segmentation_model"
    result_dir: str = "./segmentation_results"

    def validate(self):
        if self.segmentation_mode not in ["2D", "3D"]:
            raise ValueError(f"Invalid segmentation_mode: {self.segmentation_mode}")


class SegmentationModuleBase(ABC):
    @abstractmethod
    def train(self, images: Any, masks: Optional[Any], **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def predict(self, image: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        pass

    @abstractmethod
    def get_config(self) -> Any:
        pass
