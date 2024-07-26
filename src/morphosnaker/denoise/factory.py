# factory.py
from .methods.noise2void.config import Noise2VoidConfig
from .methods.noise2void.module import Noise2VoidModule


def create_model(method, config):
    if method == "n2v":
        return Noise2VoidModule(config)
    # Add more elif statements here for other methods
    else:
        raise ValueError(f"Unknown denoising method: {method}")


def create_config(method, **kwargs):
    if method == "n2v":
        return Noise2VoidConfig(**kwargs)
    # Add more elif statements here for other methods
    else:
        raise ValueError(f"Unknown denoising method: {method}")
