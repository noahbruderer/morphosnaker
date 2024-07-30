from typing import Any

from ...mixin import _format_input_batch, _validate_input
from .config import Noise2VoidConfig
from .model import Noise2VoidModel


class Noise2VoidModule:
    """
    Module for the Noise2Void denoising method.

    This class encapsulates the functionality for training and applying the
    Noise2Void denoising model.

    Attributes:
        config (Noise2VoidConfig): Configuration for the Noise2Void model.
        model (Noise2VoidModel): The underlying Noise2Void model instance.
    """

    def __init__(self, config: Noise2VoidConfig) -> None:
        """
        Initialize the Noise2VoidModule.

        Args:
            config: Configuration for the Noise2Void model.
                If not provided, a default configuration will be used.
        """
        super().__init__()
        self.config = config if config else Noise2VoidConfig()
        self.model = Noise2VoidModel(self.config)
        self._format_input_batch = _format_input_batch
        self._validate_input = _validate_input

    def train_2D(self, images: Any, **kwargs: Any) -> Any:
        """
        Train the Noise2Void model on 2D images.

        Args:
            images: The input 2D images for training.

        Returns:
            The result of the training process.
        """
        formatted_images = self._format_input_batch(images, output_dims="TXYC")
        print(formatted_images[0].shape)

        return self.model.train_2D(formatted_images)

    def train_3D(self, images: Any, **kwargs: Any) -> Any:
        """
        Train the Noise2Void model on 3D images.

        Args:
            images: The input 3D images for training.

        Returns:
            The result of the training process.
        """
        formatted_images = self._format_input_batch(images, output_dims="TZXYC")
        print(formatted_images[0].shape)
        return self.model.train_3D(formatted_images)

    def predict(self, image: Any, **kwargs: Any) -> Any:
        """
        TODO UNIFY OUTPUT DIMS
        Apply the trained Noise2Void model to denoise an image.

        Args:
            image: The input image to denoise.

        Returns:
            The denoised image.
        """
        if self.config.denoising_mode == "2D":
            formatted_images = self._format_input_batch(image, output_dims="TXYC")
        elif self.config.denoising_mode == "3D":
            formatted_images = self._format_input_batch(image, output_dims="TZXYC")
        print(formatted_images[0].shape)

        return self.model.predict(formatted_images)

    def load(self, path: str) -> None:
        """
        Load a previously trained Noise2Void model from a file.

        Args:
            path: The file path to the saved model.
        """
        self.model.load(path)
