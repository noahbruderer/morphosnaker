from .config import Noise2VoidConfig
from .model import Noise2VoidModel


class Noise2VoidModule:
    def __init__(self, config=None):
        self.config = config if config else Noise2VoidConfig()
        self.model = Noise2VoidModel(self.config)

    def train_2D(self, images):
        return self.model.train_2D(images)

    def train_3D(self, images):
        return self.model.train_3D(images)

    def predict(self, image):
        return self.model.predict(image)

    def load(self, path):
        self.model.load(path)
