import os

import numpy as np

from morphosnaker import denoise


def test_noise2void_3d():
    # Create a 3D numpy array with some noise
    base_dir = "./tests/e2e/outputs/3d_test/"
    denoiser = denoise.DenoiseModule()

    np.random.seed(42)

    # Define the dimensions
    T, C, Z, Y, X = 1, 1, 64, 128, 128

    # Generate a 5D image with shape (T, C, Z, Y, X)
    # Here we use a single time point (T=1) and single channel (C=1)
    image_tcxyz = np.random.normal(0, 1, (T, C, Z, Y, X)).astype(np.float32)

    # Create a structured pattern to add to the noise
    z, y, x = np.meshgrid(
        np.linspace(0, 1, Z),
        np.linspace(0, 1, Y),
        np.linspace(0, 1, X),
        indexing="ij",
    )

    # Add the structure to the image
    # Broadcasting the structured pattern to match the shape of the image
    image_tcxyz[0, 0, ...] += 5 * np.sin(2 * np.pi * (x + y + z))

    # Normalize the image to the [0, 1] range
    image_tcxyz = (image_tcxyz - image_tcxyz.min()) / (
        image_tcxyz.max() - image_tcxyz.min()
    )

    # Now image_tcxyz has the shape (T, C, Z, Y, X)
    print("Shape of the image:", image_tcxyz.shape)
    # Setup configuration

    denoiser = denoise.DenoiseModule(
        method="n2v",
        denoising_mode="3D",
        n2v_patch_shape=(16, 32, 32),
        train_steps_per_epoch=1,
        train_epochs=1,
        result_dir=base_dir,
        trained_model_name="n2v_test_model_3D",
    )
    # Run the training
    print(image_tcxyz.shape)
    training_history = denoiser.train_3D(image_tcxyz)
    assert training_history is not None, "Training failed"

    model_path = os.path.join(base_dir, "n2v_test_model_3D/")
    assert os.path.exists(model_path), "Model saving failed"

    # Initialize denoiser for prediction
    denoiser_predict = denoise.DenoiseModule(
        method="n2v",
        denoising_mode="3D",
        result_dir=base_dir,
        trained_model_name="n2v_test_model_3D",
        tile_shape=(64, 128, 128),
    )
    denoiser_predict.load_model("./tests/e2e/outputs/3d_test/")

    # Perform denoising
    denoised_image = denoiser_predict.predict(image_tcxyz)
    assert denoised_image is not None, "Denoising failed"
    # assert (
    #     denoised_image.shape == image_tcxyz.shape
    # ), f"Shape mismatch:{denoised_image.shape} vs {image_tcxyz.shape}"


if __name__ == "__main__":
    test_noise2void_3d()
