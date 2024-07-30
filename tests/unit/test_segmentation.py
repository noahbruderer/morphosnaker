import numpy as np
import pytest

from morphosnaker.segmentation import Segmentation
from morphosnaker.segmentation.methods.cellpose.config import CellposeConfig


@pytest.fixture
def default_cellpose_module():
    return Segmentation(method="cellpose")


@pytest.fixture
def sample_image():
    # Create a sample 2D image (100x100 with a simple shape)
    image = np.zeros((100, 100), dtype=np.float32)
    image[25:75, 25:75] = 1.0
    return image


def test_cellpose_module_initialization():
    module = Segmentation(method="cellpose")
    assert isinstance(module.config, CellposeConfig)
    assert module.method == "cellpose"
    assert module.config.channels == (0, 0)


@pytest.mark.parametrize(
    "channels, expected", [(0, (0, 0)), ((0, 0), (0, 0)), ((1, 2), (1, 2)), (1, (1, 1))]
)
def test_cellpose_different_channels(channels, expected):
    module = Segmentation(method="cellpose", channels=channels)
    assert module.config.channels == expected


def test_cellpose_invalid_channels():
    with pytest.raises(ValueError):
        Segmentation(method="cellpose", channels=(0, 0, 0))


def test_cellpose_module_configuration(default_cellpose_module):
    new_config = default_cellpose_module.configurate(model_type="nuclei", diameter=30)
    assert isinstance(new_config, CellposeConfig)
    assert new_config.model_type == "nuclei"
    assert new_config.diameter == 30


def test_cellpose_prediction(default_cellpose_module, sample_image):
    masks = default_cellpose_module.predict(sample_image)
    assert isinstance(masks, np.ndarray)
    assert masks.shape == sample_image.shape


def test_cellpose_invalid_configuration():
    with pytest.raises(ValueError):
        Segmentation(method="cellpose", model_type="invalid_model")


def test_cellpose_3d_configuration():
    module = Segmentation(method="cellpose", do_3D=True)
    assert module.config.do_3D == True


def test_cellpose_predict_with_kwargs(default_cellpose_module, sample_image):
    masks = default_cellpose_module.predict(sample_image, diameter=50)
    assert isinstance(masks, np.ndarray)
    assert masks.shape == sample_image.shape


# This test requires a pre-trained model file
@pytest.mark.skip(reason="Requires a pre-trained model file")
def test_cellpose_load_model(default_cellpose_module):
    default_cellpose_module.load_model("path/to/pretrained/model")
    # Add assertions to check if the model was loaded correctly


def test_cellpose_get_config(default_cellpose_module):
    config = default_cellpose_module.get_config()
    assert isinstance(config, CellposeConfig)
    assert config.method == "cellpose"
