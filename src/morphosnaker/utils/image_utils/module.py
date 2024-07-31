from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .methods import PlotMethod
from .processor import ImageProcessorMethod


class ImageProcessor:
    """
    A class for loading and preprocessing images using ImageProcessorMethod.
    TODO - Add more detailed description of the class and its methods.
    """

    def __init__(self) -> None:
        """Initialize the ImageProcessor."""
        self.image_processor = ImageProcessorMethod()
        self.plot = PlotMethod()

    def inspect(self, source: Union[str, List[str]], max_files: int = 5):
        """
        Inspect images without fully loading them into memory.

        Args:
            source (Union[str, List[str]]): Path to a file, list of file paths,
                or directory.
            max_files (int): Maximum number of files to inspect if source
                is a directory.

        Returns:
            List[dict]: A list of dictionaries containing inspection results.

        Example:
            >>> loader = ImageProcessor()
            >>> results = loader.inspect("path/to/images")
            >>> for result in results:
            ...     print(f"File: {result['file_path']}, Shape: "
                        f"{result['raw_shape']}")
        """
        print(f"Inspecting source: {source!r}")
        return self.image_processor.inspect(source, max_files)

    def load_raw(
        self,
        source: Union[str, List[str]],
        number_of_files: Optional[int] = None,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Load raw images without any preprocessing or modifications.

        Args:
            source (Union[str, List[str]]): Path to a file, list of file paths,
            or directory.
            number_of_files (Optional[int]): Number of files to load if source
            is a directory.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: Loaded raw image(s) as numpy
            array(s).

        Example:
            >>> loader = ImageProcessor()
            >>> images = loader.load_raw("path/to/image.tif")
            >>> print(f"Loaded raw image shape: {images.shape}")
        """
        return self.image_processor.load_raw(source, number_of_files)

    def load(
        self,
        source: Union[str, List[str]],
        input_dims: str = "auto",
        number_of_files: Optional[int] = None,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Load images and retruns standard image format TCZYX.

        Args:
            source (Union[str, List[str]]): Path to a file, list of file paths,
            or directory.
            input_dims (str): The dimensions of the input images (e.g., 'YX',
            'TYX', 'TZYXC'). Use 'auto' to attempt automatic detection.
            number_of_files (Optional[int]): Number of files to load if source
            is a directory.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: Loaded image(s) as numpy
                array(s).

        Example:
            >>> loader = ImageProcessor()
            >>> images = loader.load("path/to/image.tif", input_dims='TZYXC')
            >>> print(f"Loaded image shape: {images.shape}")
        """
        print(f"Loading source: {source!r}")
        try:
            return self.image_processor.load(source, input_dims, number_of_files)
        except Exception as e:
            print(f"Error loading images: {str(e)}")
            raise

    def standardise_image(
        self, image: np.ndarray, input_dims: str = "auto"
    ) -> np.ndarray:
        """
        Standardise the image by converting it to float32 and normalising it.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Standardised image.

        Example:
            >>> loader = ImageProcessor()
            >>> image = loader.load("path/to/image.tif")
            >>> standardised = loader.standardise_image(image)
        """
        return self.image_processor.standardise_dimensions(image, input_dims)

    def select_dimensions(
        self,
        image: np.ndarray | List[np.ndarray],
        channels: Optional[Union[int, List[int]]] = None,
        time_points: Optional[Union[int, List[int]]] = None,
        z_slices: Optional[Union[int, List[int]]] = None,
    ) -> np.ndarray | List[np.ndarray]:
        """
        Select specific channels and time points from the image while
        maintaining dimensionality.

        Args:
            image (np.ndarray): Input image with shape (T, C, Z, Y, X) or
            (T, Z, Y, X, C).
            channels (Union[int, List[int]], optional): Channel(s) to select.
            If None, all channels are kept.
            time_points (Union[int, List[int]], optional): Time point(s) to
            select. If None, all time points are kept.
            z_slices (Union[int, List[int]], optional): Z slice(s) to select.
            If None, all Z slices are kept.

        Returns:
            np.ndarray: Image with selected channels and time points.

        Example:
            >>> loader = ImageProcessor()
            >>> image = loader.load("path/to/image.tif")
            >>> selected = loader.select_dimensions(image, channels=0,
            time_points=[0, 1, 2])
            >>> print(f"Selected image shape: {selected.shape}")
        """
        return self.image_processor.select_dimensions(
            image, channels, time_points, z_slices
        )

    def format(
        self,
        image: Union[np.ndarray, List[np.ndarray]],
        output_dims: str = "TCZYX",
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Reformat the dimensions of an image array.

        This function takes an input image with dimensions in the order TCZYX
        and reformats it to the specified output dimensions. It can remove
        unnecessary dimensions and reorder the remaining ones.

        Parameters:
        -----------
        image : numpy.ndarray
        The input image array with dimensions in the order TCZYX.
        output_dims : str, optional
        A string specifying the desired output dimensions.
        Must be a subset of "TCZYX". Default is "TCZYX".

        Returns:
        --------
        numpy.ndarray
        The reformatted image array with dimensions as specified in
        output_dims.

        Raises:
        -------
        ValueError
        If output_dims contains dimensions not present in the input dimensions.

        Examples:
        ---------
        >>> img = np.random.rand(1, 3, 1, 100, 100)  # TCZYX
        >>> formatted_img = format_image_dimensions(img, output_dims="XYT")
        >>> formatted_img.shape
        (100, 100, 1)

        Notes:
        ------
        - The function assumes the input image always has dimensions in the
        order TCZYX.
        - Dimensions not specified in output_dims are removed by selecting only
        the first element along those axes.
        """
        return self.image_processor.format_images(image, output_dims)

    def crop(
        self,
        image: np.ndarray,
        crop_dims: Dict[str, Union[int, Tuple[int, int]]],
    ) -> np.ndarray:
        """
        Crop the image according to the specified dimensions.

        Args:
            image (np.ndarray): Input image.
            crop_dims (Dict[str, Union[int, Tuple[int, int]]]): Dimensions to
            crop.

        Returns:
            np.ndarray: Cropped image.
        """
        return self.image_processor.crop(image, crop_dims)

    def save(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
        file_path: Union[str, List[str]],
    ):
        """
        Save image(s) to file(s).

        Args:
            images (Union[np.ndarray, List[np.ndarray]]): Image or list of
            images to save.
            file_path (Union[str, List[str]]): Path or list of paths to save
            the images.

        Example:
            >>> loader = ImageProcessor()
            >>> images = loader.load("path/to/image.tif")
            >>> loader.save(images, "path/to/save/processed_image.tif")
        """
        if isinstance(images, np.ndarray):
            images = [images]

        if isinstance(file_path, str):
            file_path = [file_path]

        for img, path in zip(images, file_path):
            self.image_processor.save(img, path)
            print(f"Saved image to: {path}")
