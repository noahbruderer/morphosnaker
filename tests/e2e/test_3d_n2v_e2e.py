import numpy as np
from morphosnaker import denoise
import os

def test_noise2void_3d():
    # Create a 3D numpy array with some noise
    base_dir = './tests/e2e/outputs/3d_test/'
    model_name = 'n2v_3d_test_model'
    denoiser = denoise.DenoiseModule()

    np.random.seed(42)
    image_3d = np.random.normal(0, 1, (1, 64, 128, 128, 1)).astype(np.float32)
    # Add some structure to the noise
    z, y, x = np.meshgrid(np.linspace(0, 1, 64), np.linspace(0, 1, 128), np.linspace(0, 1, 128), indexing='ij')
    image_3d[0, ..., 0] += 5 * np.sin(2 * np.pi * (x + y + z))
    # Normalize to [0, 1] range
    image_3d = (image_3d - image_3d.min()) / (image_3d.max() - image_3d.min())

    # Setup configuration

    denoiser = denoiser.config(
        method = 'n2v',
        mode = 'train',
        trained_model_name=model_name,
        denoising_mode='3D',
        n2v_patch_shape=(16, 32, 32),
        train_epochs=1,  # Short training for test purposes
        train_steps_per_epoch=1,
        train_batch_size=2,
        n2v_perc_pix=0.198,
        result_dir=base_dir,
    )

    # Run the training
    history = denoiser.train_3D([image_3d])

    # # Check if the training history and figures were saved
    # assert os.path.exists(os.path.join(denoiser.config.fig_dir, f"{denoiser.trained_model_name}_training_history.png")), "Training history plot not saved"
    # assert os.path.exists(os.path.join(denoiser.config.fig_dir, f"{denoiser.trained_model_name}_sample_patches.png")), "Sample patches plot not saved"

    # Check if the metadata were saved
    #add .h5 assertion and metadata 

    print("Noise2Void 3D training test completed successfully!")
    denoiser_predict = denoiser.config(method = 'n2v',
                                    denoising_mode = '3D',
                                    mode = 'predict',
                                    result_dir=base_dir,
                                    trained_model_name= model_name)

    denoiser_predict.load_model(base_dir)
    denoised_image = denoiser_predict.denoise(image_3d)

if __name__ == "__main__":
    test_noise2void_3d()