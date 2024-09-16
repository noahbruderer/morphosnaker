from typing import Any, Dict, Union, cast

import numpy as np

from .factory import create_config, create_model
from .methods.base import SegmentationConfigBase
from .methods.cellpose.config import CellposeConfig


class SegmentationModule:
    def __init__(
        self,
        method: str = "cellpose",
        segmentation_mode: str = "2D",
        **config_kwargs: Any,
    ):
        self.method = method
        config_kwargs["segmentation_mode"] = segmentation_mode
        self.config = create_config(method, **config_kwargs)
        if method == "cellpose":
            self.config = cast(CellposeConfig, self.config)
        self.model = create_model(method, self.config)

    def configurate(
        self, **config_kwargs: Any
    ) -> Union[CellposeConfig, SegmentationConfigBase]:
        new_method = config_kwargs.pop("method", self.method)

        if "segmentation_mode" not in config_kwargs:
            config_kwargs["segmentation_mode"] = self.config.segmentation_mode

        if new_method != self.method:
            new_config = create_config(new_method, **config_kwargs)
        else:
            new_config = self._update_config(config_kwargs)

        if new_config != self.config:
            self.config = new_config
            self.method = new_method
            if new_method == "cellpose":
                self.config = cast(CellposeConfig, self.config)
            self.model = create_model(self.method, self.config)

        return self.config

    def _update_config(
        self, new_kwargs: Dict[str, Any]
    ) -> Union[CellposeConfig, SegmentationConfigBase]:
        updated_dict = self.config.__dict__.copy()
        updated_dict.update(new_kwargs)
        updated_dict.pop("method", None)
        return create_config(self.method, **updated_dict)

    def train(self, images: np.ndarray, masks: np.ndarray, **kwargs: Any) -> Any:
        return self.model.train(images, masks, **kwargs)

    def predict(self, image: np.ndarray, **kwargs: Any) -> Any:
        return self.model.predict(image, **kwargs)

    def load_model(self, path: str) -> None:
        self.model.load_model(path)

    def get_config(self) -> Union[CellposeConfig, SegmentationConfigBase]:
        return self.config
