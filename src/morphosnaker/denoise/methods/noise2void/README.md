# MorphoSnaker Denoising Module

## Overview

The MorphoSnaker Denoising Module is a powerful and flexible tool for denoising microscopy images. It provides a high-level interface for configuring, training, and applying various denoising methods, with a primary focus on the Noise2Void algorithm.

## Features

- Support for both 2D and 3D image denoising
- Configurable denoising parameters
- Easy-to-use interface for training and prediction
- Integration with popular deep learning frameworks
- Extensible architecture for adding new denoising methods
- Type annotations for improved code quality and IDE support

## Installation

To install the MorphoSnaker Denoising Module, use pip:

```python
pip install morphosnaker-denoise
```

## Quick Start

Here's a simple example of how to use the Denoising Module:

```python
from morphosnaker.denoise import DenoiseModule
import numpy as np

# Create a denoising module with default Noise2Void configuration
denoiser = DenoiseModule(method="n2v")

# Prepare some dummy data
train_images = np.random.rand(10, 100, 100).astype(np.float32)
noisy_image = np.random.rand(100, 100).astype(np.float32)

# Train the model on 2D images
history = denoiser.train_2D(train_images)

# Denoise a new image
denoised_image = denoiser.predict(noisy_image)

# Load a previously trained model
denoiser.load_model("path/to/saved/model")
```

# Configuration

The Denoising Module can be configured with various parameters. Here's an example of how to create a custom configuration:

```python
from morphosnaker.denoise import DenoiseModule

custom_config = {
    "denoising_mode": "2D",
    "train_epochs": 200,
    "n2v_patch_shape": (64, 64),
    "train_batch_size": 64
}

denoiser = DenoiseModule(method="n2v", **custom_config)
```

# Type Annotations
The MorphoSnaker Denoising Module uses type annotations throughout its codebase. This provides several benefits:

- Improved code readability
- Better IDE support with auto-completion and type checking
- Easier debugging and maintenance

To take full advantage of type annotations, we recommend using a type-aware IDE like PyCharm or Visual Studio Code with the Python extension.
Documentation
For detailed documentation on all available methods, configuration options, and return types, please refer to the API documentation.

# Contributing
We welcome contributions to the MorphoSnaker Denoising Module! Please see our Contributing Guide for more information on how to get started.
License
This project is licensed under the MIT License - see the LICENSE file for details.

# Contact

noah.bruderer@gmail.com

For questions, issues, or suggestions, please open an issue on our GitHub repository.
