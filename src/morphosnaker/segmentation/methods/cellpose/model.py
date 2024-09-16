# src/morphosnaker/segmentation/methods/cellpose/model.py
import os
from typing import Any, Optional

from cellpose import models

from ..base import SegmentationModuleBase
from .config import CellposeConfig


class CellposeModel(SegmentationModuleBase):
    def __init__(self, config: CellposeConfig):
        self.config = config
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model based on the configuration."""
        if self.config.model_type in ["cyto", "nuclei", "cyto2", "cyto3"]:
            self.model = models.CellposeModel(model_type=self.config.model_type)
        else:
            # For custom model types, we'll expect load_model to be called
            self.model = None

    def load_model(self, model_path: str) -> None:
        try:
            self.model = models.CellposeModel(pretrained_model=model_path)
            self.config.update_from_model(self.model)
            model_name = os.path.basename(model_path)
            model_name = os.path.splitext(model_name)[0]
            self.config.update_model_name(model_name)
            print(f"Custom model loaded successfully from {model_path}")
        except Exception as e:
            raise ValueError(f"Error loading Cellpose model: {str(e)}")

    def predict(self, image: Any, **kwargs: Any) -> Any:
        self._ensure_model()
        assert self.model is not None
        masks, flows, styles = self.model.eval(
            image,
            channels=self.config.channels,
            diameter=self.config.diameter,
            flow_threshold=self.config.flow_threshold,
            cellprob_threshold=self.config.cellprob_threshold,
            do_3D=self.config.do_3D,
        )
        return masks

    def _ensure_model(self):
        if self.model is None:
            self.model = models.CellposeModel(model_type=self.config.model_type)
            self.config.update_from_model(self.model)

    def train(self, images: Any, masks: Optional[Any], **kwargs: Any) -> Any:
        # Implement Cellpose training logic here
        raise NotImplementedError("Cellpose training not implemented yet")

    def get_config(self) -> CellposeConfig:
        return self.config
