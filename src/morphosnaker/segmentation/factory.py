# src/morphosnaker/segmentation/factory.py

from .methods.cellpose.config import CellposeConfig
from .methods.cellpose.module import CellposeModule


def create_config(method: str, **kwargs):
    if method == "cellpose":
        return CellposeConfig(**kwargs)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")


def create_model(method: str, config):
    if method == "cellpose":
        return CellposeModule(config)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")
