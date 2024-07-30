import os

import pytest
import tifffile

from morphosnaker import denoise, utils


@pytest.fixture
def setup_test_environment():
    base_dir = "./tests/e2e/outputs/2d_test/"
    os.makedirs(base_dir, exist_ok=True)
    path_file = "./tests/e2e/data/still_juvenile2_stack_1_img.tiff"
    return base_dir, path_file


def test_noise2void_2d(setup_test_environment):
    base_dir, path_file = setup_test_environment

    # Initialize utilities
    image_processor = utils.ImageProcessor()

    # Load and process image
    image_info = image_processor.inspect(path_file)
    assert image_info, "Image inspection failed"

    image = image_processor.load(path_file, "XYC")
    assert image is not None, "Image loading failed"

    image = image_processor.select_dimensions(image, channels=0)
    assert image.ndim == 5, f"Expected 4D image, got {image.ndim}D"

    # Initialize denoiser for training
    denoiser = denoise.DenoiseModule(
        method="n2v",
        denoising_mode="2D",
        n2v_patch_shape=(64, 64),
        train_steps_per_epoch=1,
        train_epochs=1,
        result_dir=base_dir,
        trained_model_name="n2v_test_model_2D",
    )

    # Train the model
    training_history = denoiser.train_2D([image, image])
    assert training_history is not None, "Training failed"

    # Save the model
    model_path = os.path.join(base_dir, "n2v_test_model_2D/")
    assert os.path.exists(model_path), "Model saving failed"

    # Initialize denoiser for prediction
    denoiser_predict = denoise.DenoiseModule(
        method="n2v",
        denoising_mode="2D",
        result_dir=base_dir,
        trained_model_name="n2v_test_model_2D",
        tile_shape=(128, 128),
    )

    # Load the model
    denoiser_predict.load_model("./tests/e2e/outputs/2d_test/")

    # Perform denoising
    denoised_image = denoiser_predict.predict(image)
    assert denoised_image is not None, "Denoising failed"
    # assert (
    #     denoised_image.shape == image.shape
    # ), f"Shape mismatch:{denoised_image.shape} vs {image.shape}"

    # Save the denoised image
    denoised_path = os.path.join(base_dir, "denoised_image.tiff")
    tifffile.imwrite(denoised_path, denoised_image)
    assert os.path.exists(denoised_path), "Denoised image saving failed"

    print(f"Denoised image shape: {denoised_image.shape}")

    # Test configuration getting and setting
    train_config = denoiser.get_config()
    assert train_config is not None, "Failed to get training configuration"

    predict_config = denoiser_predict.get_config()
    assert predict_config is not None, "Failed to get prediction configuration"

    # Update configuration
    updated_config = denoiser.configurate(train_epochs=2)
    assert updated_config.train_epochs == 2, "Configuration update failed"


if __name__ == "__main__":
    pytest.main([__file__])
