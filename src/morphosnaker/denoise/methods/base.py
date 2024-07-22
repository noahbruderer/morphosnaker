from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, List
import os

class DenoiseTrainBase(ABC):
    """Abstract base class for training denoising model."""

    @abstractmethod
    def configure(self, config: Any) -> None:
        """Configure the denoising method."""
        pass
    
    @abstractmethod
    def train_2D(self, images: Any, **kwargs: Any) -> Any:
        """Train the model on 2D images."""
        pass
    

class DenoisePredictBase(ABC):
    """Abstract base class for training denoising model."""

    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load a trained model."""
        pass
    
    @abstractmethod
    def denoise(self, image: Any, **kwargs: Any) -> Any:
        """Denoise an image."""
        pass

    
@dataclass
class DenoiseTrainingConfigBase:
    """Base configuration for training denoising models."""

    method: str
    denoising_mode: str
    trained_model_name: str = 'my_model'
    train_steps_per_epoch: int = 100
    train_epochs: int = 100
    train_loss: str = 'mse'
    batch_norm: bool = True
    train_batch_size: int = 128
    result_dir: str = './results_training'
    fig_dir: Optional[str] = 'figures'
    image_dimensions: str = 'XY'

    def __post_init__(self) -> None:
        """Set default directories if not provided."""
        self.fig_dir = os.path.join(self.result_dir, self.fig_dir)

@dataclass
class DenoisePredictConfigBase:
    """Base configuration for prediction."""

    method: str
    trained_model_name: str = 'my_model'
    result_dir: str = './results_prediction'
    fig_dir: Optional[str] = 'figures'

    def __post_init__(self) -> None:
        """Set default directories if not provided."""
        self.fig_dir = os.path.join(self.result_dir, self.fig_dir)
            
class DenoiseMethodModuleBase(ABC):
    @abstractmethod
    def configure(self, config):
        pass

    @abstractmethod
    def train_2D(self, images, input_dims=None, **kwargs):
        pass

    @abstractmethod
    def denoise(self, image, input_dims=None, **kwargs):
        pass

    @abstractmethod
    def load_model(self, path):
        pass