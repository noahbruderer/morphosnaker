# MorphoSnaker

**Standardise your image processing!**

**MorphoSnaker** is a Python package designed for microscopy and biological imaging applications: **Image processing, Denoising, Segmentation.**


## Features

- [Image Loading and Preprocessing](#image-loading-and-preprocessing)
- [Denoising Techniques](#denoising-techniques)
<!-- - [Segmentation Tools (in development)](#segmentation-tools-in-development)
- [Flexible and Extensible Architecture](#flexible-and-extensible-architecture) -->

## Installation

### Recommended: Containerised
To ensure a consistent environment and streamline the setup process, we recommend using the MorphoSnaker Singularity image. This image includes all necessary dependencies and configurations for running MorphoSnaker.

#### Building the Singularity Image

1. **Prerequisites**:
   - Ensure that Singularity is installed on your system. For installation instructions, refer to the [Singularity installation guide](https://sylabs.io/guides/3.5/user-guide/installation.html).

2. **Prepare the Definition File**:
   - Download the image definition file https://github.com/Tatan47/morphosnaker/image/morphosnaker.def

3. **Build the Image**:
   - Use the following command to build the Singularity image. Replace `morphosnaker.sif` with the desired name for your image file.

    ```bash
    singularity build morphosnaker.sif morphosnaker.def
    ```

#### Running the Singularity Image

1. **Executing Commands**:
   - To run a Python script or other commands inside the Singularity container, use the following command format:

    ```bash
    singularity exec morphosnaker.sif python3 /path/to/your/script.py
    ```

   - Replace `/path/to/your/script.py` with the actual path to your script. This command executes the script within the container environment.

2. **Interactive Shell**:
   - If you need an interactive shell within the Singularity container, use:

    ```bash
    singularity shell morphosnaker.sif
    ```

   - This command drops you into an interactive shell session within the container, where you can manually execute commands and scripts.

#### Additional Notes

- **Data Binding**: To access data or scripts from your host system within the container, use the `--bind` option to bind mount directories:

  ```bash
  singularity exec --bind /host/path:/container/path morphosnaker.sif python3 /container/path/script.py
  ```

## Install using pip
To install MorphoSnaker, use the following command:

	```bash
	pip install git+https://github.com/Tatan47/morphosnaker.git
	```

# Tutorial

## Image Loading and Preprocessing

To quickly get started with MorphoSnaker, follow these steps:


### Load, Inspect, and Standardize Your Image

Before working with your image data, it’s essential to inspect and standardize it to ensure consistent dimensions and formatting. The standard format used here is (TCZYX), where:

- T represents time points,
- C represents the channels.
- Z represents the z-stack,
- Y and X represent the spatial dimensions,

The following steps outline how to inspect and standardize your image:

```python
from utils import ImageProcessor

# Create an instance of the ImageProcessor
loader = ImageProcessor()

# Inspect your image to gather metadata
image_info = loader.inspect("path/to/your/image.tif")

# OUTPUT:
Inspecting: path/to/your/image.tif
Raw shape: (5, 256, 256)
Number of pages: 5
Data type: uint8
Value range: min = 0.0000, max = 1.0000

Inspected 1 file(s).

# Define the input dimensions explicitly
image_dimensions = "CXY"

# Load and standardize the image to the specified dimensions
image = loader.load("path/to/your/image.tif", image_dimensions)

# output:
Loading source: '/Users/noahbruderer/Documents/Documents - Noah’s MacBook Pro - 1/Work_Documents/morphosnaker/tests/e2e/data/training_mask_2.tiff'
Loading files:   0%|                                                                                         | 0/1 [00:00<?, ?it/s]Input shape: (5, 256, 256), Input dims: CXY
After transposing existing dims: shape=(5, 256, 256), dims=CYX
After adding dimensions: shape= (1, 5, 1, 256, 256), dims=TCZYX
```

<!-- Explanation:

	1.	Create an Instance: Instantiate the ImageProcessor class, which provides methods for inspecting and loading images.
	2.	Inspect the Image: Use the inspect method to gather information about the image, such as its shape, data type, and value range. This step helps you understand the structure of your image data.
	3.	Define Input Dimensions: Explicitly define the input dimensions using a string. This step avoids the potential issues with auto-detection and ensures your image is standardized correctly.
	4.	Load and Standardize: Use the load method with the defin
  ed dimensions to standardize your image format to (TZYXC). If some dimensions do not exist, they will be created to fit this format.

This approach ensures that your image data is consistently formatted, making it easier to work with in subsequent processing steps. -->

## Denoising Techniques

### Train your model
Follow the steps below to train a 2D noise2void on one input imge, you can also input lists of images to increase your training data size.
```python
from morphosnaker import denoise, utils

#load and format your image, noise2void requires TZYXZ or TYXC and one channel at a time
image_processor = utils.ImageProcessor()
image = image_processor.load(path_file, "XYC")

#select the channel 
image = image_processor.select_dimensions(image, channels=0)

#arrange the dimensions 
image = image_processor.format(image, "TYXC")

# Set-up the denoiser, here we use the noise2void method
# Adjust the configurations as needed 

denoiser = denoise.DenoiseModule(
      method="n2v",
      denoising_mode="2D",
      n2v_patch_shape=(64, 64),
      train_steps_per_epoch=1,
      train_epochs=1,
      result_dir="/path/to/results",
      trained_model_name="your_n2v_model",
   )

# train the model
training_history = denoiser.train_2D([image, image])
```
### Denoise images

Load the trained model to denosie images.

```python

# Initialize denoiser for prediction
denoiser_predict = denoise.DenoiseModule(
   method="n2v",
   denoising_mode="2D",
   result_dir="/path/to/results",
   trained_model_name="your_n2v_model",
   tile_shape=(128, 128),
)
#load the model you have trained earlier
denoiser_predict.load_model("/path/to/results/your_n2v_model")

# Perform denoising
denoised_image = denoiser_predict.predict(image)

# save the denoised image
tifffile.imwrite("/path/to/your/images/denoised_image.tiff", denoised_image)


```


## Modules

MorphoSnaker consists of the following modules:

- Denoising
- Utils
- Segmentation (coming soon)

## Contributing

We welcome contributions! Please see our [Contributing Guide](link-to-contributing-guide) for more details.

## License

This project is licensed under the MIT License. See the [LICENSE](link-to-license-file) file for details.