import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import numpy as np

from morphosnaker.utils import ImageProcessor


class DenoiseTrainBase(ABC):
    """Abstract base class for training denoising models."""

    @abstractmethod
    def configure(self, config: Any) -> None:
        """
        Configure the denoising method.

        Args:
            config: Configuration object for the denoising method.
        """
        pass

    @abstractmethod
    def train_2D(self, images: Any, **kwargs: Any) -> Any:
        """
        Train the model on 2D images.

        Args:
            images: Input 2D images for training.
            **kwargs: Additional keyword arguments for training.

        Returns:
            The result of the training process.
        """
        pass

    @abstractmethod
    def train_3D(self, images: Any, **kwargs: Any) -> Any:
        """
        Train the model on 3D images.

        Args:
            images: Input 3D images for training.
            **kwargs: Additional keyword arguments for training.

        Returns:
            The result of the training process.
        """
        pass


class DenoisePredictBase(ABC):
    """Abstract base class for applying denoising models."""

    @abstractmethod
    def load_model(self, path: str) -> None:
        """
        Load a trained model.

        Args:
            path: File path to the saved model.
        """
        pass

    @abstractmethod
    def predict(self, image: Any, **kwargs: Any) -> Any:
        """
        Denoise an image.

        Args:
            image: Input image to denoise.
            **kwargs: Additional keyword arguments for prediction.

        Returns:
            The denoised image.
        """
        pass


@dataclass
class DenoiseTrainingConfigBase:
    """Base configuration for training denoising models."""

    method: str
    denoising_mode: str
    trained_model_name: str = "my_model"
    train_steps_per_epoch: int = 100
    train_epochs: int = 100
    train_loss: str = "mse"
    batch_norm: bool = True
    train_batch_size: int = 128
    result_dir: str = "./results_training"
    fig_dir: str = "figures"
    image_dimensions: str = "XY"

    def __post_init__(self) -> None:
        """Set default directories if not provided."""
        self.fig_dir = os.path.join(self.result_dir, self.fig_dir)


@dataclass
class DenoiseConfigBase:
    """Base configuration for denoising models."""

    method: str
    denoising_mode: str
    trained_model_name: str = "my_model"
    result_dir: str = "./results"
    fig_dir: str = "figures"
    channel: Optional[int] = None

    def __post_init__(self) -> None:
        """Set default directories and validate configuration."""
        self.fig_dir = os.path.join(self.result_dir, self.fig_dir)
        self.validate()

    def validate(self) -> None:
        """Validate the configuration."""
        self._validate_denoising_mode()

    def _validate_denoising_mode(self) -> None:
        """Validate the denoising_mode parameter."""
        if self.denoising_mode not in ["2D", "3D"]:
            raise ValueError(
                "denoising_mode must be either '2D' or '3D', got: "
                f"{self.denoising_mode}"
            )


class DenoiseMethodModuleBase(DenoiseTrainBase, DenoisePredictBase):
    """Base class for denoising method modules.
    TODO add docstring and abstract class for the"""

    def __init__(self):
        self.image_processor = ImageProcessor()

    @abstractmethod
    def configure(self, config: DenoiseConfigBase) -> None:
        """
        Configure the denoising method.

        Args:
            config: Configuration object for the denoising method.
        """
        pass

    @abstractmethod
    def train_2D(self, images: Any, **kwargs: Any) -> Any:
        """
        Train the model on 2D images.

        Args:
            images: Input 2D images for training.
            **kwargs: Additional keyword arguments for training.

        Returns:
            The result of the training process.
        """
        pass

    @abstractmethod
    def train_3D(self, images: Any, **kwargs: Any) -> Any:
        """
        Train the model on 3D images.

        Args:
            images: Input 3D images for training.
            **kwargs: Additional keyword arguments for training.

        Returns:
            The result of the training process.
        """
        pass

    @abstractmethod
    def predict(self, image: Any, **kwargs: Any) -> Any:
        """
        Denoise an image.

        Args:
            image: Input image to denoise.
            **kwargs: Additional keyword arguments for prediction.

        Returns:
            The denoised image.
        """
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        """
        Load a trained model.

        Args:
            path: File path to the saved model.
        """
        pass

    def _format_input_batch(
        self, images: Union[np.ndarray, List[np.ndarray]], output_dims: str
    ) -> Union[np.ndarray, list[np.ndarray]]:
        """
        Format a batch of input images to the specified output dimensions.
        Args:
            images: Input images (single array or list of arrays).
            output_dims: Desired output dimensions.
        Returns:
            List of formatted images.
        """
        if isinstance(images, np.ndarray):
            return self.image_processor.format(images, output_dims)
        elif isinstance(images, list):
            return self.image_processor.format(images, output_dims)

    def _validate_input(self, image: np.ndarray, expected_dims: str) -> None:
        """
        Validate that the input image has the expected dimensions.
        Args:
            image: Input image.
            expected_dims: Expected dimensions of the input.
        Raises:
            ValueError: If the input dimensions don't match the expected dimensions.
        """
        if image.ndim != len(expected_dims):
            raise ValueError(
                f"Input image should have {len(expected_dims)} dimensions, "
                f"but has {image.ndim}"
            )
