# factory.py
from typing import Any

from .methods.noise2void.config import Noise2VoidConfig
from .methods.noise2void.module import Noise2VoidModule


def create_config(method: str, **kwargs: Any) -> Noise2VoidConfig:
    """
    Create a configuration object for the specified denoising method.

    Args:
        method: The denoising method to use (e.g., "n2v" for Noise2Void).
        **kwargs: Additional keyword arguments for configuring the denoising
        method.

    Returns:
        A configuration object for the specified method.

    Raises:
        ValueError: If an unknown denoising method is specified.
    """
    if method == "n2v":
        return Noise2VoidConfig(method=method, **kwargs)
    # Add more elif statements here for other methods
    else:
        raise ValueError(f"Unknown denoising method: {method}")


def create_model(method: str, config: Noise2VoidConfig) -> Noise2VoidModule:
    """
    Create a denoising model based on the specified method and configuration.

    Args:
        method (str): The denoising method to use (e.g., "n2v" for Noise2Void).
        config (Noise2VoidConfig): The configuration object for the denoising,
        for now only Noise2VoidConfig, must be extended as new methods are
        added.
        method.

    Returns:
        Noise2VoidModule: An instance of the appropriate denoising
        model.

    Raises:
        ValueError: If an unknown denoising method is specified.
    """
    if method == "n2v":
        return Noise2VoidModule(config)
    # Add more elif statements here for other methods
    else:
        raise ValueError(f"Unknown denoising method: {method}")
