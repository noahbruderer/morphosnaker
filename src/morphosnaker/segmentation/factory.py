# src/morphosnaker/segmentation/factory.py

from typing import Any

from .methods.base import SegmentationConfigBase, SegmentationModuleBase
from .methods.cellpose.config import CellposeConfig
from .methods.cellpose.module import CellposeModule


def create_config(method: str, **kwargs: Any) -> SegmentationConfigBase:
    if method == "cellpose":
        # Set default segmentation_mode to "2D" if not provided
        kwargs.setdefault("segmentation_mode", "2D")
        return CellposeConfig(method=method, **kwargs)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")


def create_model(method: str, config: SegmentationConfigBase) -> SegmentationModuleBase:
    if method == "cellpose":
        if isinstance(config, CellposeConfig):
            return CellposeModule(config)
        else:
            raise ValueError(
                f"Expected CellposeConfig for method 'cellpose', got {type(config)}"
            )
    else:
        raise ValueError(f"Unknown segmentation method: {method}")
