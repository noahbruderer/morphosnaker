from morphosnaker import denoise
from morphosnaker import utils
import tifffile

image_processor = utils.ImageProcessor()
denoiser = denoise.DenoiseModule()
# Load the pre-trained N2V model (replace 'path_to_model' with the actual path to your model)
base_dir = "./tests/e2e/outputs/2d_test/"
path_file = "/Users/noahbruderer/Documents/Documents - Noah’s MacBook Pro - 1/Work_Documents/morphomics/morphosnaker/tests/e2e/data/still_juvenile2_stack_1_img.tiff"

image = image_processor.inspect(path_file)
image = image_processor.load(path_file, "XYC")
image = image_processor.select_dimensions(image, channels=0)
# For training
denoiser = denoiser.config(method= 'n2v', 
                           mode = 'train', 
                           denoising_mode='2D',
                            n2v_patch_shape=(64, 64),
                            train_steps_per_epoch=1,
                            train_epochs=1,
                            result_dir=base_dir,
                            image_dimensions = 'ZYXC',
                            trained_model_name='n2v_test_model_2D')

training_history = denoiser.train_2D(image)

# For prediction
denoiser_predict = denoiser.config(method = 'n2v',
                                   denoising_mode = '2D',
                                   mode = 'predict',
                                   result_dir=base_dir,
                                   trained_model_name= 'n2v_test_model_2D',
                                   tile_shape=(128, 128))

denoiser_predict.load_model(base_dir)
denoised_image = denoiser_predict.denoise(image)
# Save the denoised image
print(denoised_image.shape)
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