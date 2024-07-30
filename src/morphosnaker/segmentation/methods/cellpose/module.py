# src/morphosnaker/segmentation/methods/cellpose/module.py

from .config import CellposeConfig
from .model import CellposeModel


class CellposeModule:
    def __init__(self, config: CellposeConfig):
        self.config = config
        self.model = CellposeModel(config)

    def train(self, images, masks, **kwargs):
        return self.model.train(images, masks, **kwargs)

    def predict(self, image, **kwargs):
        return self.model.predict(image, **kwargs)

    def load_model(self, path):
        self.model.load_model(path)
