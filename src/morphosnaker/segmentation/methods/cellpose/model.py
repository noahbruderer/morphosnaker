# src/morphosnaker/segmentation/methods/cellpose/model.py


from typing import Optional

import numpy as np
from cellpose import models as cellpose_models

from ..base import SegmentationMethodBase
from .config import CellposeConfig


class CellposeModel(SegmentationMethodBase):
    def __init__(self, config: CellposeConfig):
        self.config = config
        self.model: Optional[cellpose_models.Cellpose] = None

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

    def load_model(self, path: str):
        self.model = cellpose_models.CellposeModel(
            pretrained_model=path, model_type=self.config.model_type  # type: ignore
        )
        self.config.update_from_model(self.model)


# comments:
# the type error in load_model method is due to the fact that the CellposeModel class is
# not defined in the cellpose.models module.
