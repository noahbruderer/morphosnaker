import numpy as np

from morphosnaker import denoise, utils

image_processor = utils.ImageProcessor()
denoiser = denoise.DenoiseModule()
# Load the pre-trained N2V model (replace 'path_to_model' with the actual path to your model)
base_dir = "./tests/e2e/outputs/2d_test/"
path_file = (
    "/Users/noahbruderer/Documents/Documents - Noah’s MacBook Pro -"
    " 1/Work_Documents/morphomics/data/raw_images/liveimaging_starved_juvenile_240617_0007_twoevents.tif"
)
image = image_processor.inspect(path_file)
image = image_processor.load(path_file, "TZCYX")

z_stack_start = 0
z_stack_stop = 2
img_stacks_list = []
for i in range(z_stack_start, z_stack_stop):
    image_2d = image_processor.select_dimensions(
        image, channels=[1], time_points=list(range(7, image.shape[0])), z_slices=[i]
    )
    img_stacks_list.append(image_2d)

denoiser = denoise.DenoiseModule(
    method="n2v",
    denoising_mode="2D",
    n2v_patch_shape=(64, 64),
    train_steps_per_epoch=1,
    train_epochs=1,
    result_dir=base_dir,
    trained_model_name="n2v_test_model_2D",
)

history = denoiser.train_2D(img_stacks_list)

# For prediction
denoiser_predict = denoise.DenoiseModule(
    method="n2v",
    denoising_mode="2D",
    result_dir=base_dir,
    trained_model_name="n2v_test_model_2D",
    tile_shape=(128, 128),
)

denoiser_predict.load_model(base_dir)

image_2d = image_processor.select_dimensions(
    image, channels=[1], time_points=list(range(7, image.shape[0]))
)

# Create an empty array to store the denoised images
denoised_image = np.zeros_like(image_2d)

# Loop through each time point (T)
for t in range(image_2d.shape[0]):
    # Loop through each channel (C)
    for c in range(image_2d.shape[1]):
        # Loop through each Z stack (Z)
        for z in range(image_2d.shape[2]):
            # Extract the 2D slice at this specific (T, C, Z)
            image_slice = image_processor.select_dimensions(
                image_2d, channels=[c], time_points=[t], z_slices=[z]
            )
            # Check if the image_slice is 2D
            # Denoise the 2D slice
            denoised_slice = denoiser_predict.predict(image_slice)
            # Store the denoised slice back into the appropriate position in the output
            # array
            denoised_image[t, c, z, :, :] = denoised_slice

# # For prediction
# denoiser = DenoiseModule.config('n2v', 'predict', tile_shape=(128, 128))
# denoiser.load_model('path/to/saved/model')
# denoised_image = denoiser.denoise(noisy_image)

# # Getting configurations
# train_config = denoiser.get_config('train')
# predict_config = denoiser.get_config('predict')

# # Setting configurations
# denoiser.set_config(new_train_config, 'train')
# denoiser.set_config(new_predict_config, 'predict')
