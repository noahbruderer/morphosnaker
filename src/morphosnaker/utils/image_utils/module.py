from typing import Union, Optional, List
import os
import numpy as np
from .processor import ImageProcessorMethod

class ImageProcessor:
    """
    A class for loading and preprocessing images using ImageProcessorMethod.
    """

    def __init__(self, channels: Optional[Union[int, List[int]]] = None, time_points: Optional[Union[int, List[int]]] = None):
        """
        Initialize the ImageProcessor.

        Args:
            channels (Optional[Union[int, List[int]]]): Channels to load. If None, load all channels.
            time_points (Optional[Union[int, List[int]]]): Time points to load. If None, load all time points.
        """
        self.image_processor = ImageProcessorMethod(channels=channels, time_points=time_points)

    def inspect(self, source: Union[str, List[str]], max_files: int = 5):
        """
        Inspect images without fully loading them into memory.

        Args:
            source (Union[str, List[str]]): Path to a file, list of file paths, or directory.
            max_files (int): Maximum number of files to inspect if source is a directory.

        Returns:
            List[dict]: A list of dictionaries containing inspection results.

        Example:
            >>> loader = ImageProcessor()
            >>> results = loader.inspect("path/to/images")
            >>> for result in results:
            ...     print(f"File: {result['file_path']}, Shape: {result['raw_shape']}")
        """
        print(f"Inspecting source: {source!r}")
        return self.image_processor.inspect(source, max_files)

    def load(self, source: Union[str, List[str]], input_dims: str = 'auto', max_files: int = 5) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Load images without any preprocessing or modifications.

        Args:
            source (Union[str, List[str]]): Path to a file, list of file paths, or directory.
            input_dims (str): The dimensions of the input images (e.g., 'YX', 'TYX', 'TZYXC'). 
                              Use 'auto' to attempt automatic detection.
            max_files (int): Maximum number of files to load if source is a directory.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: Loaded image(s) as numpy array(s).

        Example:
            >>> loader = ImageProcessor()
            >>> images = loader.load("path/to/image.tif", input_dims='TZYXC')
            >>> print(f"Loaded image shape: {images.shape}")
        """
        print(f"Loading source: {source!r}")
        try:
            return self.image_processor.load(source, input_dims, max_files)
        except Exception as e:
            print(f"Error loading images: {str(e)}")
            raise

    def select_dimensions(self, image: np.ndarray, channels: Union[int, List[int]] = None, time_points: Union[int, List[int]] = None, z_slices: Union[int, List[int]] = None) -> np.ndarray:
        """
        Select specific channels and time points from the image while maintaining dimensionality.

        Args:
            image (np.ndarray): Input image with shape (T, Y, X, C) or (T, Z, Y, X, C).
            channels (Union[int, List[int]], optional): Channel(s) to select. If None, all channels are kept.
            time_points (Union[int, List[int]], optional): Time point(s) to select. If None, all time points are kept.

        Returns:
            np.ndarray: Image with selected channels and time points.

        Example:
            >>> loader = ImageProcessor()
            >>> image = loader.load("path/to/image.tif")
            >>> selected = loader.select_channels_and_timepoints(image, channels=0, time_points=[0, 1, 2])
            >>> print(f"Selected image shape: {selected.shape}")
        """
        return self.image_processor.select_dimensions(image, channels, time_points, z_slices)


    def load_and_preprocess(self, source: Union[str, List[str]], input_dims: str = 'auto', 
                            modifications: Optional[dict] = None) -> List[np.ndarray]:
        """
        Load, preprocess, and optionally modify images.

        Args:
            source (Union[str, List[str]]): Path to a file, list of file paths, or directory.
            input_dims (str): The dimensions of the input images (e.g., 'YX', 'TYX', 'TZYXC').
                              Use 'auto' to attempt automatic detection.
            modifications (Optional[dict]): Dictionary of modifications to apply to the images.

        Returns:
            List[np.ndarray]: List of processed numpy arrays representing the images.

        Example:
            >>> loader = ImageProcessor()
            >>> mods = {'crop_box': (0, 100, 0, 100)}
            >>> images = loader.load_and_preprocess("path/to/images", input_dims='TYXC', modifications=mods)
            >>> print(f"Processed {len(images)} images")
        """
        images = self.load(source, input_dims)
        
        if not isinstance(images, list):
            images = [images]
        
        if modifications:
            images = [self.image_processor.apply_modifications(img, modifications) for img in images]
        
        return images

    def save(self, images: Union[np.ndarray, List[np.ndarray]], file_path: Union[str, List[str]]):
        """
        Save image(s) to file(s).

        Args:
            images (Union[np.ndarray, List[np.ndarray]]): Image or list of images to save.
            file_path (Union[str, List[str]]): Path or list of paths to save the images.

        Example:
            >>> loader = ImageProcessor()
            >>> images = loader.load("path/to/image.tif")
            >>> loader.save(images, "path/to/save/processed_image.tif")
        """
        if isinstance(images, np.ndarray):
            images = [images]
            file_path = [file_path]
        
        for img, path in zip(images, file_path):
            self.image_processor.save(img, path)
            print(f"Saved image to: {path}")