from typing import List, Union

import numpy as np

from morphosnaker.utils import ImageProcessor


def _format_input_batch(
    images: Union[np.ndarray, List[np.ndarray]], output_dims: str
) -> Union[np.ndarray, list[np.ndarray]]:
    """
    Format a batch of input images to the specified output dimensions.
    Args:
        images: Input images (single array or list of arrays).
        output_dims: Desired output dimensions.
    Returns:
        List of formatted images.
    """
    if isinstance(images, np.ndarray):
        return ImageProcessor().format(images, output_dims)
    elif isinstance(images, list):
        return ImageProcessor().format(images, output_dims)


def _validate_input(image: np.ndarray, expected_dims: str) -> None:
    """
    Validate that the input image has the expected dimensions.
    Args:
        image: Input image.
        expected_dims: Expected dimensions of the input.
    Raises:
        ValueError: If the input dimensions don't match the expected dimensions.
    """
    if image.ndim != len(expected_dims):
        raise ValueError(
            f"Input image should have {len(expected_dims)} dimensions, "
            f"but has {image.ndim}"
        )
