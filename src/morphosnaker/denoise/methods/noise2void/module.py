from morphosnaker.denoise.methods.base import DenoiseMethodModuleBase
from morphosnaker.denoise.methods.noise2void.config import Noise2VoidTrainingConfig, Noise2VoidPredictionConfig
from morphosnaker.denoise.factory import create_training_config, create_predict_config
from typing import Union, Literal, Optional
from morphosnaker.denoise.methods.noise2void.model import Noise2VoidPredict, Noise2VoidTrain

class Noise2VoidDenoiseModule(DenoiseMethodModuleBase):
    def __init__(self, mode: Literal['train', 'predict']):
        self.mode = mode
        if mode == 'train':
            self.model = Noise2VoidTrain()
        elif mode == 'predict':
            self.model = Noise2VoidPredict()
        else:
            raise ValueError("Mode must be either 'train' or 'predict'")
        self.config = None

    @classmethod
    def config(cls, mode: Literal['train', 'predict'], **kwargs):
        instance = cls(mode)
        if mode == 'train':
            instance.config = create_training_config('n2v', **kwargs)
        elif mode == 'predict':
            instance.config = create_predict_config('n2v', **kwargs)
        instance.configure(instance.config)
        return instance

    def configure(self, config: Union[Noise2VoidTrainingConfig, Noise2VoidPredictionConfig]):
        self.config = config
        self.model.configure(config)

    def train_2D(self, images, **kwargs):
        if self.mode != 'train':
            raise AttributeError("train_2D is only available in train mode")
        return self.model.train_2D(images, **kwargs)

    def train_3D(self, images, **kwargs):
        if self.mode != 'train':
            raise AttributeError("train_3D is only available in train mode")
        return self.model.train_3D(images, **kwargs)
    
    def denoise(self, image, **kwargs):
        if self.mode != 'predict':
            raise AttributeError("denoise is only available in predict mode")
        return self.model.denoise(image, **kwargs)

    def load_model(self, path):
        if self.mode != 'predict':
            raise AttributeError("load_model is only available in predict mode")
        self.model.load_model(path)

    def save_model(self, path):
        if self.mode != 'train':
            raise AttributeError("save_model is only available in train mode")
        self.model.save_model(path)

    def __getattr__(self, name):
        if self.mode == 'train' and name in ['denoise', 'load_model']:
            raise AttributeError(f"{name} is not available in train mode")
        elif self.mode == 'predict' and name in ['train_2D', 'save_model']:
            raise AttributeError(f"{name} is not available in predict mode")
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")