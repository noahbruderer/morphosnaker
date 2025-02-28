# src/morphosnaker/segmentation/methods/cellpose/model.py
import os
from typing import Any, Dict, Optional

import numpy as np
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
            self.model = models.CellposeModel(
                model_type=self.config.model_type, gpu=self.config.GPU
            )
        else:
            # For custom model types, we'll expect load_model to be called
            self.model = None

    def load_model(self, model_path: str) -> None:
        try:
            self.model = models.CellposeModel(
                pretrained_model=model_path,
                gpu=self.config.GPU,  # Use the GPU config parameter
            )
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

    def predict_for_training(self, image: Any, **kwargs: Any) -> Dict[str, np.ndarray]:
        """
        Predict segmentation with additional outputs for training visualization.
        Similar to Cellpose GUI output format.

        Args:
            image: Input image to segment
            **kwargs: Additional keyword arguments passed to model.eval

        Returns:
            Dict containing:
                - 'masks': Binary segmentation masks
                - 'flows': Flow fields/probability maps
                - 'styles': Style vectors
                - 'seg': Combined array with:
                    - chan0: Segmentation mask
                    - chan1: Cell probability
                    - chan2: Flow field X
                    - chan3: Flow field Y
        """
        self._ensure_model()
        assert self.model is not None

        # Get predictions from model
        masks, flows, styles = self.model.eval(
            image,
            channels=self.config.channels,
            diameter=self.config.diameter,
            flow_threshold=self.config.flow_threshold,
            cellprob_threshold=self.config.cellprob_threshold,
            do_3D=self.config.do_3D,
        )

        cellprob = flows[2]  # Cellpose stores probability in flows[2]
        dP = flows[0]  # X flow field
        dY = flows[1]  # Y flow field

        # Create combined seg array (similar to GUI seg.npy)
        # Shape will be [4, H, W] for 2D or [4, Z, H, W] for 3D
        if self.config.do_3D:
            seg = np.stack([masks, cellprob, dP, dY], axis=0)
        else:
            seg = np.stack([masks, cellprob, dP, dY], axis=0)

        return {
            "masks": masks,  # Binary segmentation masks
            "flows": flows,  # Raw flow outputs
            "styles": styles,  # Style vectors
            "seg": seg,  # Combined array similar to GUI output
        }

    def _ensure_model(self):
        if self.model is None:
            self.model = models.CellposeModel(model_type=self.config.model_type)
            self.config.update_from_model(self.model)

    def train(self, images: Any, masks: Optional[Any], **kwargs: Any) -> Any:
        # Implement Cellpose training logic here
        raise NotImplementedError("Cellpose training not implemented yet")

    def get_config(self) -> CellposeConfig:
        return self.config
