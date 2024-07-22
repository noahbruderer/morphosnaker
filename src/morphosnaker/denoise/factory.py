from typing import Union
from morphosnaker.denoise.methods.noise2void.config import Noise2VoidTrainingConfig, Noise2VoidPredictionConfig, DenoiseTrainingConfigBase
from morphosnaker.denoise.methods.base import DenoisePredictConfigBase, DenoiseTrainingConfigBase
from typing import Dict, Type

TRAINING_CONFIG_MAP: Dict[str, Type[DenoiseTrainingConfigBase]] = {
    'n2v': Noise2VoidTrainingConfig,
    # Add other methods here
}

PREDICT_CONFIG_MAP: Dict[str, Type[DenoisePredictConfigBase]] = {
    'n2v': Noise2VoidPredictionConfig,
    # Add other methods here
}

def create_training_config(method: str, **kwargs) -> DenoiseTrainingConfigBase:
    config_class = TRAINING_CONFIG_MAP.get(method)
    if config_class is None:
        raise ValueError(f"Unknown training method: {method}")
    return config_class(method=method, **kwargs)

def create_predict_config(method: str, **kwargs) -> DenoisePredictConfigBase:
    config_class = PREDICT_CONFIG_MAP.get(method)
    if config_class is None:
        raise ValueError(f"Unknown denoising method: {method}")
    return config_class(method=method, **kwargs)