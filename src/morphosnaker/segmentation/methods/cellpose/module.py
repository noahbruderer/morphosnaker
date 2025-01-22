# src/morphosnaker/segmentation/methods/cellpose/module.py
from typing import Any, Optional

import numpy as np

from ..base import SegmentationModuleBase
from .config import CellposeConfig
from .model import CellposeModel


class CellposeModule(SegmentationModuleBase):
    def __init__(self, config: CellposeConfig):
        self.config = config
        self.model = CellposeModel(config)

    def train(
        self, images: np.ndarray, masks: Optional[np.ndarray], **kwargs: Any
    ) -> Any:
        return self.model.train(images, masks, **kwargs)

    def predict(self, image: np.ndarray, **kwargs: Any) -> Any:
        return self.model.predict(image, **kwargs)

    def predict_for_training(self, image: np.ndarray, **kwargs: Any) -> Any:
        return self.model.predict_for_training(image, **kwargs)

    def load_model(self, path: str) -> None:
        self.model.load_model(path)

    def get_config(self) -> CellposeConfig:
        return self.config
