# src/morphosnaker/segmentation/methods/cellpose/model.py


import os
from typing import Optional

import numpy as np
from cellpose import models as cellpose_models

from ..base import SegmentationMethodBase
from .config import CellposeConfig


class CellposeModel(SegmentationMethodBase):
    def __init__(self, config: CellposeConfig):
        self.config = config
        self.model: Optional[cellpose_models.Cellpose] = None

    def load_model(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        # Cellpose model loading
        self.model = cellpose_models.CellposeModel(
            pretrained_model=path,
            gpu=self.config.use_gpu,
        )

        # Update config based on loaded model
        # Only update attributes that we're sure exist
        self.config.diameter = getattr(self.model, "diam_mean", self.config.diameter)
        self.config.pretrained_model = path

        print(f"Cellpose model loaded successfully from: {path}")
        print(f"Updated config: {self.config}")

    def get_model_info(self):
        if self.model is None:
            return {"error": "No Cellpose model loaded"}

        # Use getattr with default values to avoid AttributeError
        return {
            "model_name": os.path.basename(self.config.pretrained_model),
            "model_type": self.config.model_type,
            "diameter": self.config.diameter,
            "pretrained_model": self.config.pretrained_model,
            "device": str(getattr(self.model, "device", "unknown")),
            "nclasses": getattr(self.model, "nclasses", "unknown"),
            "nchan": getattr(self.model, "nchan", "unknown"),
        }

    def _ensure_model(self):
        if self.model is None:
            self.model = cellpose_models.Cellpose(model_type=self.config.model_type)

    def train(self, images: np.ndarray, masks, **kwargs):
        # Implement Cellpose training logic here
        print("Training Cellpose must still be implemented")
        pass

    def predict(self, image: np.ndarray, **kwargs):
        self._ensure_model()

        assert self.model is not None, "Model should be initialized"

        result = self.model.eval(
            image,
            channels=self.config.channels,
            diameter=self.config.diameter,
            flow_threshold=self.config.flow_threshold,
            cellprob_threshold=self.config.cellprob_threshold,
            min_size=self.config.min_size,
            stitch_threshold=self.config.stitch_threshold,
            do_3D=self.config.do_3D,
        )

        if len(result) == 4:
            masks, flows, styles, diams = result
        elif len(result) == 3:
            masks, flows, styles = result
            diams = None
        else:
            raise ValueError(
                f"Unexpected number of return values from Cellpose model: {len(result)}"
            )

        return masks


# comments:
# the type error in load_model method is due to the fact that the CellposeModel class is
# not defined in the cellpose.models module.
