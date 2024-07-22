from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Literal
from morphosnaker.denoise.methods.base import DenoiseTrainingConfigBase, DenoisePredictConfigBase

@dataclass
class Noise2VoidTrainingConfig(DenoiseTrainingConfigBase):
    """Configuration for Noise2Void training."""
    n2v_patch_shape: Tuple[int, ...] = None
    training_patch_fraction: float = 0.8
    unet_kern_size: int = 3
    n2v_perc_pix: float = 0.198
    n2v_manipulator: str = 'uniform_withCP'
    n2v_neighborhood_radius: int = 5
    structN2Vmask: Optional[List[List[int]]] = None
    denoising_mode: str = '2D'
    save_weights_only: bool = False
    channel: Optional[int] = None  # None means all channels, otherwise specify the channel index
    author: str = 'morphosnaker'
    
    def __post_init__(self) -> None:
        """Validate the configuration after initialization."""
        super().__post_init__()
        if self.n2v_perc_pix <= 0 or self.n2v_perc_pix >= 1:
            raise ValueError(f"n2v_perc_pix must be between 0 and 1, got: {self.n2v_perc_pix}")
        if self.denoising_mode not in ['2D', '3D']:
            raise ValueError(f"denoising_mode must be either '2D' or '3D', got: {self.denoising_mode}")
        if self.n2v_patch_shape is not None:
            if self.denoising_mode == '2D' and len(self.n2v_patch_shape) != 2:
                raise ValueError(f"For 2D, n2v_patch_shape must be a 2-tuple, got: {self.n2v_patch_shape}")
            if self.denoising_mode == '3D' and len(self.n2v_patch_shape) != 3:
                raise ValueError(f"For 3D, n2v_patch_shape must be a 3-tuple, got: {self.n2v_patch_shape}")

@dataclass
class Noise2VoidPredictionConfig(DenoisePredictConfigBase):
    """Configuration for Noise2Void denoising."""
    denoising_mode: Literal['2D', '3D'] = '2D'
    trained_model_name = 'my_model'
    tile_overlap: int = 16
    tile_shape: Tuple[int, int] = (64, 64)
    input_dims: str = 'YX'  # Default to 'YX'
    channel: Optional[int] = None  # None means all channels, otherwise specify the channel index
    def __post_init__(self) -> None:
        """Validate the configuration after initialization."""
        super().__post_init__()
        if len(self.tile_shape) != 2:
            raise ValueError(f"tile_shape must be a 2-tuple, got: {self.tile_shape}")
        if self.tile_overlap < 0:
            raise ValueError(f"tile_overlap must be non-negative, got: {self.tile_overlap}")