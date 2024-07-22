import numpy as np
import os
from typing import List, Union, Optional
from morphosnaker.denoise.methods.noise2void.module import Noise2VoidDenoiseModule

class DenoiseModule:
    def __init__(self):
        self.train_module = None
        self.predict_module = None

    @classmethod
    def config(cls, method: str, mode: str, **kwargs) -> 'DenoiseModule':
        instance = cls()
        if method == 'n2v':
            if mode == 'train':
                instance.train_module = Noise2VoidDenoiseModule.config('train', **kwargs)
            elif mode == 'predict':
                instance.predict_module = Noise2VoidDenoiseModule.config('predict', **kwargs)
            else:
                raise ValueError(f"Unknown mode: {mode}. Must be 'train' or 'predict'")
        else:
            raise ValueError(f"Unknown denoising method: {method}")
        return instance

    def train_2D(self, images, **kwargs):
        if self.train_module is None:
            raise ValueError("Training module not configured. Use .config('n2v', 'train', ...) first.")
        return self.train_module.train_2D(images, **kwargs)

    def train_3D(self, images, **kwargs):
        if self.train_module is None:
            raise ValueError("Training module not configured. Use .config('n2v', 'train', ...) first.")
        return self.train_module.train_3D(images, **kwargs)


    def denoise(self, image, **kwargs):
        if self.predict_module is None:
            raise ValueError("Prediction module not configured. Use .config('n2v', 'predict', ...) first.")
        return self.predict_module.denoise(image, **kwargs)

    def save_model(self, path):
        if self.train_module is None:
            raise ValueError("Training module not configured. Cannot save model.")
        self.train_module.save_model(path)

    def load_model(self, path):
        if self.predict_module is None:
            raise ValueError("Prediction module not configured. Use .config('n2v', 'predict', ...) first.")
        self.predict_module.load_model(path)

    def get_config(self, mode: str):
        if mode == 'train':
            return self.train_module.config if self.train_module else None
        elif mode == 'predict':
            return self.predict_module.config if self.predict_module else None
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'train' or 'predict'")

    def set_config(self, config, mode: str):
        if mode == 'train':
            if self.train_module is None:
                raise ValueError("Training module not configured. Use .config('n2v', 'train', ...) first.")
            self.train_module.configure(config)
        elif mode == 'predict':
            if self.predict_module is None:
                raise ValueError("Prediction module not configured. Use .config('n2v', 'predict', ...) first.")
            self.predict_module.configure(config)
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'train' or 'predict'")