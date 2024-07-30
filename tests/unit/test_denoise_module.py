import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from morphosnaker import denoise


@pytest.fixture
def sample_image():
    return np.random.rand(1, 100, 100, 1).astype(np.float32)


def test_denoise_module_initialization():
    denoiser = denoise.DenoiseModule()
    assert denoiser is not None
    assert denoiser.method_module is None


def test_denoise_module_config():
    denoiser = denoise.DenoiseModule.config(
        "n2v",
        denoising_mode="2D",
        n2v_patch_shape=(64, 64),
        train_steps_per_epoch=1,
        train_epochs=1,
    )
    assert denoiser.method_module is not None
    assert denoiser.get_config().method == "n2v"
    assert denoiser.get_config().denoising_mode == "2D"
    assert denoiser.get_config().n2v_patch_shape == (64, 64)

    with pytest.raises(ValueError):
        denoise.DenoiseModule.config("invalid_method")


# @patch('morphosnaker.denoise.methods.noise2void.model.N2V')
# @patch('morphosnaker.denoise.methods.noise2void.model.N2VConfig')
def test_train_2D(sample_image):
    mock_model = MagicMock()
    mock_model.train.return_value = {"loss": [0.1], "val_loss": [0.2]}

    denoiser = denoise.DenoiseModule.config(
        "n2v",
        denoising_mode="2D",
        n2v_patch_shape=(64, 64),
        train_steps_per_epoch=1,
        train_epochs=1,
    )
    denoiser.train_2D([sample_image])


@patch("morphosnaker.denoise.methods.noise2void.model.N2V")
def test_denoise(mock_n2v, sample_image):
    mock_model = MagicMock()
    mock_n2v.return_value = mock_model
    mock_model.predict.return_value = np.zeros_like(sample_image)

    denoiser = denoise.DenoiseModule.config(
        "n2v",
        denoising_mode="2D",
        n2v_patch_shape=(64, 64),
        train_steps_per_epoch=1,
        train_epochs=1,
    )
    denoiser.train_2D([sample_image])
    denoised_image = denoiser.denoise(sample_image)

    assert denoised_image.shape == sample_image.shape
    assert not np.array_equal(denoised_image, sample_image)
    mock_model.predict.assert_called_once()


@patch("morphosnaker.denoise.methods.noise2void.model.N2V")
def test_save_load_model(mock_n2v, sample_image):
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model")

        mock_model = MagicMock()
        mock_n2v.return_value = mock_model

        denoiser = denoise.DenoiseModule.config(
            "n2v",
            denoising_mode="2D",
            n2v_patch_shape=(64, 64),
            train_steps_per_epoch=1,
            train_epochs=1,
            author="test_author",
            trained_model_name="test_model",
        )
        denoiser.train_2D([sample_image])
        # denoiser.save_model(model_path)

        assert os.path.exists(model_path)

        new_denoiser = denoise.DenoiseModule.config("n2v", denoising_mode="2D")
        new_denoiser.load_model(model_path)

        mock_n2v.assert_called_with(config=None, name="n2v", basedir=tmpdir)


def test_get_set_config():
    denoiser = denoise.DenoiseModule.config("n2v", denoising_mode="2D")
    initial_config = denoiser.get_config()

    new_config = type(initial_config)(
        method="n2v", denoising_mode="3D", n2v_patch_shape=(32, 64, 64)
    )
    denoiser.set_config(new_config)

    updated_config = denoiser.get_config()
    assert updated_config.method == "n2v"
    assert updated_config.denoising_mode == "3D"
    assert updated_config.n2v_patch_shape == (32, 64, 64)
