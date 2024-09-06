from typing import Any, Dict, List, Union

import numpy as np

from .factory import create_config, create_model
from .methods.noise2void.config import DenoiseConfigBase, Noise2VoidConfig
from .methods.noise2void.module import Noise2VoidModule


class DenoiseModule:
    """
    A module for denoising images using various methods.

    This class provides a high-level interface for configuring, training,
    and applying denoising models to images.

    Attributes:
        method (str): The denoising method to use (e.g., "n2v" for Noise2Void).
        config (Noise2VoidConfig): The configuration object for the denoising.
        model (Noise2VoidModule): The denoising model instance.

    Methods:
        __init__(method: str = "n2v", **config_kwargs: Any) -> None:
            Initialize the DenoiseModule.

        configurate(**config_kwargs: Any) -> Noise2VoidConfig:
            Reconfigure the denoising module with new parameters.

        train_2D(images: Union[List[np.ndarray], np.ndarray]) -> Any:
            Train the denoising model on 2D images.

        train_3D(images: Union[List[np.ndarray], np.ndarray]) -> Any:
            Train the denoising model on 3D images.

        predict(image: np.ndarray) -> Any:
            Apply the trained denoising model to denoise an image.

        load_model(path: str) -> None:
            Load a previously trained denoising model from a file.

        get_config() -> Union[Noise2VoidConfig, DenoiseConfigBase]:
            Get the current configuration of the denoising module.

    Usage:
        >>> denoiser = DenoiseModule(method="n2v")
        >>> denoiser.configurate(train_epochs=100, train_steps_per_epoch=100)
        >>> denoiser.train_2D(train_images)
        >>> denoised_image = denoiser.predict(test_image)
    """

    def __init__(self, method: str = "n2v", **config_kwargs: Any) -> None:
        """
        Initialize the DenoiseModule.

        Args:
            method: The denoising method to use. Defaults to "n2v".
            **config_kwargs: Additional keyword arguments for configuring the
            denoising method.
        """
        self.method: str = method
        self.config: Noise2VoidConfig = create_config(method, **config_kwargs)
        self.model: Noise2VoidModule = create_model(method, self.config)

    def configurate(self, **config_kwargs: Any) -> Noise2VoidConfig:
        """
        Reconfigure the denoising module with new parameters.

        This method allows updating the configuration and recreating the model
        if necessary.

        Args:
            **config_kwargs: Keyword arguments for updating the configuration.

        Returns:
            Noise2VoidConfig: The updated configuration object. For now only
            Noise2VoidConfig, must be extended as methods are added.
        """
        new_method = config_kwargs.pop("method", self.method)

        if new_method != self.method:
            # Create a new config with default values for the new method
            new_config = create_config(new_method, **config_kwargs)
        else:
            # Update existing config
            new_config = self._update_config(config_kwargs)

        if new_config != self.config:
            self.config = new_config
            self.method = new_method
            self.model = create_model(self.method, self.config)

        return self.config

    def _update_config(self, new_kwargs: Dict[str, Any]) -> Noise2VoidConfig:
        """
        Update the current configuration with new keyword arguments.

        Args:
            new_kwargs: New configuration parameters.

        Returns:
            Noise2VoidConfig: The updated configuration object. For now only
            Noise2VoidConfig, must be extended as methods are added.
        """

        updated_dict = self.config.__dict__.copy()
        updated_dict.update(new_kwargs)
        # Remove 'method' from updated_dict to avoid passing it twice
        updated_dict.pop("method", None)
        return create_config(self.method, **updated_dict)

    def train_2D(self, images: Union[List[np.ndarray], np.ndarray]) -> Any:
        """
        Train the denoising model on 2D images.

        Args:
            images (Union[np.ndarray, List[np.ndarray]]): The input 2D images for
            training.

        Returns:
            Any: The result of the training process.
        """
        return self.model.train_2D(images)

    def train_3D(self, images: Union[List[np.ndarray], np.ndarray]) -> Any:
        """
        Train the denoising model on 3D images.

        Args:
            images: The input 3D images for training.

        Returns:
            The result of the training process.
        """
        return self.model.train_3D(images)

    def predict(self, image: np.ndarray) -> Any:
        """
        Apply the trained denoising model to denoise an image.

        Args:
            image: The input image to denoise.

        Returns:
            The denoised image.
        """
        return self.model.predict(image)

    def load_model(self, path: str) -> None:
        """
        Load a previously trained denoising model from a file.

        Args:
            path (str): The file path to the saved model.
        """
        self.model.load(path)

    def get_config(self) -> Union[Noise2VoidConfig, DenoiseConfigBase]:
        """
        Get the current configuration of the denoising module.

        Returns:
            DenoiseConfigBase: The current configuration object.
        """
        return self.config
