# MorphoSnaker

MorphoSnaker is a comprehensive Python package for image processing, denoising, and segmentation, primarily designed for microscopy and biological imaging applications.

## Features

- Image loading and preprocessing
- Advanced denoising techniques (e.g., Noise2Void)
- Segmentation tools (in development)
- Flexible and extensible architecture

## Installation

To install MorphoSnaker, use the following command:

```bash
pip install morphosnaker
```

## Quick Start

To quickly get started with MorphoSnaker, follow these steps:

1. Import the necessary modules:

```python
from morphosnaker import denoise, utils
```

2. Load, Inspect, and Standardize Your Image

Before working with your image data, it’s essential to inspect and standardize it to ensure consistent dimensions and formatting. This process involves inspecting the image to gather metadata and then explicitly defining the input dimensions for standardization. The standard format used here is (TZYXC), where:

	•	T represents time points,
	•	Z represents the z-stack,
	•	Y and X represent the spatial dimensions, and
	•	C represents the channels.

The following steps outline how to inspect and standardize your image:

```python
from utils import ImageProcessor

# Create an instance of the ImageProcessor
loader = ImageProcessor()

# Inspect your image to gather metadata
image_info = loader.inspect("path/to/your/image.tif")

# Define the desired input dimensions explicitly
image_dimensions = "TZYXC"

# Load and standardize the image to the specified dimensions
image = loader.load("path/to/your/image.tif", image_dimensions)
```

Explanation:

	1.	Create an Instance: Instantiate the ImageProcessor class, which provides methods for inspecting and loading images.
	2.	Inspect the Image: Use the inspect method to gather information about the image, such as its shape, data type, and value range. This step helps you understand the structure of your image data.
	3.	Define Input Dimensions: Explicitly define the input dimensions using a string. This step avoids the potential issues with auto-detection and ensures your image is standardized correctly.
	4.	Load and Standardize: Use the load method with the defin
  ed dimensions to standardize your image format to (TZYXC). If some dimensions do not exist, they will be created to fit this format.

This approach ensures that your image data is consistently formatted, making it easier to work with in subsequent processing steps.

3. Denoise the image:

```python
denoiser = denoise.DenoiseModule.config('n2v', denoising_mode='2D')
denoised_image = denoiser.denoise(image)
```

4. Save the denoised image:

```python
loader.save(denoised_image, "path/to/save/denoised_image.tif")
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