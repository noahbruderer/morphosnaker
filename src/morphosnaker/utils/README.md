# MorphoSnaker Utils Module

The utils module in MorphoSnaker provides essential tools for image loading, inspection, and preprocessing.
The main goal is to easaly load images and always have a consistent format such as: TCZYX.  
## Features

- Flexible image loading from files or directories
- Image inspection without full loading
- Standardized dimension handling
- Channel and time point selection

## Usage

To use the utils module, follow these steps:

1. Initialize the ImageProcessor:

```python
from morphosnaker.utils import ImageProcessor

processor = ImageProcessor()
```

2. Inspect an image or directory:

```python
info = processor.inspect("path/to/image_or_directory")
```

3. Load an image or multiple images:

```python
images = processor.load("path/to/image_or_directory", input_dims='TCZYX')
```

4. Select specific channels and time points:

```python
selected_image = processor.select_dimensions(image, channels=[0, 2], time_points=[0, 1, 2])
```

5. Save an image:

```python
processor.save(processed_image, "path/to/save/processed_image.tif")
```


## Key Methods

- `inspect`: Examine image properties without full loading
- `load`: Load images with automatic dimension standardization
- `load_raw`: Load images in raw format
- `format`: format loaded images in standard into any arrangement you want 
- `select_dimensions`: Extract specific channels and time points
- `save`: Save processed images

## Supported File Formats

- TIFF (.tif, .tiff)
- NumPy arrays (.npy)

## Dimension Handling

The `ImageProcessor` automatically standardizes input images to either TYXC (for 2D) or TZYXC (for 3D) format, making it easy to work with images from various sources.

