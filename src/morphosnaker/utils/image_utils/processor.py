import os
from glob import glob
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tifffile
from termcolor import colored
from tqdm import tqdm

from .base import ImageProcessorBase


class ImageProcessorMethod(ImageProcessorBase):
    """
    A class for standardized image processing operations.

    This class provides methods for loading, inspecting, processing, and saving
    image data. It supports various file formats and dimension orders, and
    includes utility methods for common image processing tasks.

    Attributes:
        channels (Optional[Union[int, List[int]]]): Channels to process.
        time_points (Optional[Union[int, List[int]]]): Time points to process.
    """

    SUPPORTED_FORMATS = [".tif", ".tiff", ".npy"]

    def __init__(
        self,
    ):
        """
        Initialize the ImageProcessorMethod.
        """

    def inspect(
        self,
        source: Union[str, List[str]],
        number_of_files: int = 5,
        print_results: bool = True,
    ) -> List[Dict]:
        """
        Inspect image files without fully loading them into memory.
        Args:
            source (Union[str, List[str]]): Path to a file, directory, or list
            of file paths.
            number_of_files (int): Maximum number of files to inspect.
            print_results (bool): Whether to print the inspection results.
        Returns:
            List[Dict]: List of dictionaries containing inspection results for
            each file.
        Raises:
            Exception: If an error occurs during inspection.
        """
        try:
            file_list = self._get_file_list(source, number_of_files)
            results = []
            for file_path in tqdm(file_list, desc="Inspecting files"):
                results.append(self._inspect_single_file(file_path))

            if print_results:
                self._print_inspection_results(results)

            return results
        except Exception as e:
            print(colored(f"Error during inspection: {str(e)}", "red"))
            raise

    def load_raw(
        self,
        source: Union[str, List[str]],
        number_of_files: Optional[int] = None,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Load image(s) from the source.

        Args:
            source (Union[str, List[str]]): Path to a file, directory, or list
            of file paths.
            input_dims (str): The dimension order of the input images. Use
            'auto' for automatic detection.
            number_of_files (Optional[int]): Maximum number of files to load.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: Unprocessed image(s).

        Raises:
            Exception: If an error occurs during loading or processing.
        """
        try:
            file_list = self._get_file_list(source, number_of_files)
            return [
                self._load_file(file_path)
                for file_path in tqdm(file_list, desc="Loading files")
            ]
        except Exception as e:
            print(colored(f"Error during loading: {str(e)}", "red"))
            raise

    def load(
        self,
        source: Union[str, List[str]],
        input_dims: str = "auto",
        number_of_files: Optional[int] = None,
    ):
        try:
            file_list = self._get_file_list(source, number_of_files)
            images = [
                self._load_and_process_file(file_path, input_dims, output_dims="TCZYX")
                for file_path in tqdm(file_list, desc="Loading files")
            ]
            if images != list:
                return images[0]
            else:
                return images

        except Exception as e:
            print(colored(f"Error during loading: {str(e)}", "red"))
            raise

    def format_images(
        self,
        image: Union[np.ndarray, list[np.ndarray]],
        output_dim: str,
    ) -> Union[np.ndarray, list[np.ndarray]]:

        if isinstance(image, list):
            return [self._format_image_dimensions(img, output_dim) for img in image]
        else:
            return self._format_image_dimensions(image, output_dims=output_dim)

    def select_dimensions(
        self,
        image: np.ndarray,
        channels: Optional[Union[int, List[int]]] = None,
        time_points: Optional[Union[int, List[int]]] = None,
        z_slices: Optional[Union[int, List[int]]] = None,
    ) -> np.ndarray:
        """
        Select specific channels, time points, and Z slices from the image
        while maintaining dimensionality.

        Args:
        image (np.ndarray): Input image with shape (T, C, Z, Y, X).
        channels (Union[int, List[int]], optional): Channel(s) to select. If
        None, all channels are kept.
        time_points (Union[int, List[int]], optional): Time point(s) to select.
        If None, all time points are kept.
        z_slices (Union[int, List[int]], optional): Z slice(s) to select. If
        None, all Z slices are kept.

        Returns:
        np.ndarray: Image with selected channels, time points, and Z slices.

        Raises:
        ValueError: If the input image doesn't have the expected number of
        dimensions.
        """
        if image.ndim not in [5]:
            raise ValueError(
                "Input image must have 5 (T, C, Z, Y, X), dimensions. Load it "
                "with the ImageProcessor.load method."
            )

        # Convert single integers to lists for consistent processing
        channels = [channels] if isinstance(channels, int) else channels
        time_points = [time_points] if isinstance(time_points, int) else time_points
        z_slices = [z_slices] if isinstance(z_slices, int) else z_slices

        # Select time points
        if time_points is not None:
            image = image[time_points, ...]

        # Select Z slices if the image is 5D
        if z_slices is not None:
            image = image[:, :, z_slices, ...]

        # Select channels
        if channels is not None:
            image = image[:, channels, ...]

        return image

    def normalize(
        self, image: np.ndarray, method: str = "minmax", clip: bool = True
    ) -> np.ndarray:
        """
        Normalize the input image.

        Args:
            image (np.ndarray): Input image to be normalized.
            method (str): Normalization method. Options: "minmax", "zscore".
            Default is "minmax".
            clip (bool): Whether to clip values to [0, 1] range after
            normalization. Default is True.

        Returns:
            np.ndarray: Normalized image.

        Raises:
            ValueError: If an unsupported normalization method is specified.
        """
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        if method == "minmax":
            min_val = image.min()
            max_val = image.max()
            if min_val != max_val:
                image = (image - min_val) / (max_val - min_val)
            else:
                image = np.zeros_like(image)
        elif method == "zscore":
            mean = image.mean()
            std = image.std()
            if std != 0:
                image = (image - mean) / std
            else:
                image = np.zeros_like(image)
        else:
            raise ValueError(f"Unsupported normalization method: {method}")

        if clip:
            image = np.clip(image, 0, 1)

        return image

    def crop(
        self,
        image: np.ndarray,
        crop_specs: Dict[str, Union[int, Tuple[int, int]]],
    ) -> np.ndarray:
        """
        Crop the image along specified dimensions.

        Args:
            image (np.ndarray): Input image with shape (T, C, Z, Y, X).
            crop_specs (Dict[str, Union[int, Tuple[int, int]]]): Crop
            specifications.
            Keys should be 't', 'c', 'z', 'y', or 'x'.
            Values can be either an int (for single index) or a tuple of two
            ints (for range).
            Example: {'z': (10, 20), 'y': (100, 200), 'x': (50, 250)}

        Returns:
            np.ndarray: Cropped image.

        Raises:
            ValueError: If the image dimensions are not 5D or if crop specs are
            invalid.
        """
        if image.ndim != 5:
            raise ValueError("Image must be 5D (T, C, Z, Y, X)")

        # Map dimension names to their respective indices
        dim_map = {"t": 0, "c": 1, "z": 2, "y": 3, "x": 4}
        slices = [slice(None)] * 5  # Default slice: selects all elements

        for dim, spec in crop_specs.items():
            if dim in dim_map:
                axis = dim_map[dim]
                if isinstance(spec, int):
                    # Select a single index, making it a slice for consistency
                    slices[axis] = slice(spec, spec + 1)
                elif isinstance(spec, tuple) and len(spec) == 2:
                    # Select a range of indices
                    slices[axis] = slice(spec[0], spec[1])
                else:
                    raise ValueError(
                        f"Invalid crop specification for dimension '{{dim}}': {spec}"
                    )

        # Convert slices list to a tuple for indexing
        slice_tuple = tuple(slices)
        return image[slice_tuple]

    def save(self, image: np.ndarray, file_path: str) -> None:
        """
        Save an image to a file.

        Args:
            image (np.ndarray): Image to save.
            file_path (str): Path where the image will be saved.

        Raises:
            ValueError: If the file format is unsupported.
            Exception: If an error occurs during saving.
        """
        try:
            if file_path.endswith((".tif", ".tiff")):
                tifffile.imwrite(file_path, image)
            elif file_path.endswith(".npy"):
                np.save(file_path, image)
            else:
                raise ValueError(f"Unsupported file format for saving: {file_path}")
            print(colored(f"Successfully saved image to {file_path}", "green"))
        except Exception as e:
            print(colored(f"Error saving file {file_path}: {str(e)}", "red"))
            raise

    def _load_and_process_file(
        self, file_path: str, input_dims: str, output_dims: str = "TCZYX"
    ) -> np.ndarray:
        """
        Load and process a single image file.

        Args:
            file_path (str): Path to the image file.
            input_dims (str): The dimension order of the input image.

        Returns:
            np.ndarray: Loaded and processed image.

        Raises:
            Exception: If an error occurs during loading or processing.
        """
        try:
            image = self._load_file(file_path)
            if input_dims == "auto":
                input_dims = self._guess_dimensions(image)
            return self._process_image(image, input_dims, output_dims)
        except Exception as e:
            print(colored(f"Error processing file {file_path}: {str(e)}", "red"))
            raise

    def _process_image(
        self, image: np.ndarray, input_dims: str, output_dims: str
    ) -> np.ndarray:
        """
        Process a loaded image: select channels, standardize dimensions, and
        normalize.

        Args:
            image (np.ndarray): Input image.
            input_dims (str): The dimension order of the input image.

        Returns:
            np.ndarray: Processed image.

        Raises:
            ValueError: If channels cannot be selected from the image.
            Exception: If an error occurs during image processing.
        """
        try:
            image = self._standardize_dimensions(
                image, input_dims, output_dims=output_dims
            )
            return image
        except Exception as e:
            print(colored(f"Error processing image: {str(e)}", "red"))
            raise

    def _standardize_dimensions(
        self, image: np.ndarray, input_dims: str, output_dims: str = "TCZYX"
    ) -> np.ndarray:
        """
        Standardize image dimensions to TCZYX.

        Args:
            image (np.ndarray): Input image.
            input_dims (str): The dimension order of the input image.

        Returns:
            np.ndarray: Image with standardized dimensions.

        Raises:
            AssertionError: If the number of dimensions doesn't match the
            specified input_dims.
        """
        print(colored(f"Input shape: {image.shape}, Input dims: {input_dims}", "blue"))

        assert len(image.shape) == len(
            input_dims
        ), "Number of dimensions doesn't match the specified input_dims"
        # check if z dim is present in input_dims, if not: 2D

        # I - initiation: we order the dimensions in the output_dims order
        # (regardless if all target dims are present in input_dims or not
        existing_dims = [dim for dim in output_dims if dim in input_dims]
        transpose_order = [input_dims.index(dim) for dim in existing_dims]
        image = np.transpose(image, transpose_order)
        reordered_dims = "".join(existing_dims)
        print(
            colored(
                f"After transposing existing dims: shape={image.shape}, dims="
                f"{reordered_dims}",
                "yellow",
            )
        )
        # II: - Add missing dimensions to our input image to standardise it
        # first we map the standard dimensions to their axis
        dim_to_axis = {dim: i for i, dim in enumerate(output_dims)}
        for dim in output_dims:
            if dim not in reordered_dims:
                axis = dim_to_axis[dim]
                image = np.expand_dims(image, axis=axis)
                reordered_dims = reordered_dims[:axis] + dim + reordered_dims[axis:]
        print(
            colored(
                "After adding dimensions: shape={image.shape}, "
                f"dims={reordered_dims}",
                "yellow",
            )
        )

        return image

    def _guess_dimensions(self, image: np.ndarray) -> str:
        """
        TODO THIS HAS TO BE IMPROVED!!!

        Guess the dimension order based on the shape of the image.
        Args:
            image (np.ndarray): Input image.

        Returns:
            str: Guessed dimension order.

        Raises:
            ValueError: If unable to guess dimensions for the given shape.
        """
        shape = image.shape
        if len(shape) == 2:
            return "YX"
        elif len(shape) == 3:
            return "ZYX" if shape[0] < shape[1] and shape[0] < shape[2] else "TYX"
        elif len(shape) == 4:
            return "ZYXC" if shape[0] < shape[1] and shape[0] < shape[2] else "TYXC"
        elif len(shape) == 5:
            return "TZYXC"
        else:
            raise ValueError(f"Unable to guess dimensions for shape {shape}")

    def _get_file_list(
        self,
        source: Union[str, List[str]],
        number_of_files: Optional[int] = None,
    ) -> List[str]:
        """
        Get a list of image files from the source.
        Args:
            source (Union[str, List[str]]): Path to a file, directory, or list
            of file paths.
            number_of_files (Optional[int]): Maximum number of files to return.
        Returns:
            List[str]: List of valid file paths.
        Raises:
            ValueError: If the source is neither a valid file nor a directory.
            TypeError: If the source is not a string or list.
        """
        try:
            if isinstance(source, str):
                if os.path.isdir(source):
                    file_list = []
                    for format in self.SUPPORTED_FORMATS:
                        file_list.extend(glob(os.path.join(source, f"*{format}")))
                    file_list.sort()
                elif os.path.isfile(source):
                    file_list = [source]
                else:
                    if not os.path.exists(source):
                        raise FileNotFoundError(f"Source '{source}' does not exist.")
                    else:
                        raise ValueError(
                            f"Source '{source}' is neither a valid file nora directory."
                        )
            elif isinstance(source, list):
                file_list = []
                for item in source:
                    if os.path.isfile(item) and any(
                        item.lower().endswith(format)
                        for format in self.SUPPORTED_FORMATS
                    ):
                        file_list.append(item)
                    else:
                        print(
                            colored(
                                f"Warning: '{item}' is not a valid file or not"
                                " in supported format. Skipping.",
                                "yellow",
                            )
                        )

            valid_files = [
                f
                for f in file_list
                if os.path.isfile(f)
                and any(f.lower().endswith(format) for format in self.SUPPORTED_FORMATS)
            ]

            if len(valid_files) == 0:
                raise ValueError("No valid files found in the provided source.")

            return (
                valid_files[:number_of_files]
                if number_of_files is not None
                else valid_files
            )
        except Exception as e:
            print(colored(f"Error in _get_file_list: {str(e)}", "red"))
            raise

    def _load_file(self, file_path: str) -> np.ndarray:
        """
        Load an image file into a numpy array.

        Args:
            file_path (str): Path to the image file.

        Returns:
            np.ndarray: Loaded image.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is unsupported.
            Exception: If an error occurs during file loading.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File does not exist: {file_path}")

        try:
            if file_path.endswith(".npy"):
                return np.load(file_path)
            elif file_path.lower().endswith((".tif", ".tiff")):
                return tifffile.imread(file_path)
            else:
                raise ValueError(
                    f"Unsupported file format: {file_path}. Supported formats"
                    "are .npy,"
                    " .tif, and .tiff"
                )
        except Exception as e:
            print(colored(f"Error loading file {file_path}: {str(e)}", "red"))
            raise

    def _inspect_single_file(self, file_path: str) -> dict:
        """
        Inspect a single image file and return its properties.

        This method loads the file, extracts key information such as shape,
        data type,
        number of pages (for multi-page TIFF files), and value range.

        Args:
            file_path (str): Path to the image file.

        Returns:
            dict: A dictionary containing the file's properties.
                Keys include:
                - 'raw_shape': The shape of the loaded image array.
                - 'file_path': The path to the file.
                - 'num_pages': The number of pages (for TIFF files, 1 for
                others).
                - 'dtype': The data type of the image.
                - 'min_value': The minimum pixel value in the image.
                - 'max_value': The maximum pixel value in the image.

        Raises:
            Exception: If there's an error loading or inspecting the file.
        """
        try:
            # Load the file using the existing _load_file method
            image = self._load_file(file_path)

            # Determine the number of pages
            if file_path.lower().endswith((".tif", ".tiff")):
                with tifffile.TiffFile(file_path) as tif:
                    num_pages = len(tif.pages)
            else:
                num_pages = 1 if image.ndim == 2 else image.shape[0]

            return {
                "raw_shape": image.shape,
                "file_path": file_path,
                "num_pages": num_pages,
                "dtype": str(image.dtype),
                "min_value": float(image.min()),
                "max_value": float(image.max()),
            }
        except Exception as e:
            return {"file_path": file_path, "error": str(e)}

    def _print_inspection_results(self, results: List[dict]):
        """
        Print the inspection results for one or more files.

        This method takes the results from inspecting files and prints them
        in a formatted, colored output to the console.

        Args:
            results (List[dict]): A list of dictionaries, each containing
                the inspection results for a single file.

        Note:
            This method uses colored output, which may not be visible in all
            console environments.
        """
        for result in results:
            if "error" in result:
                print(
                    colored(
                        f"Error inspecting {result['file_path']}: {result['error']}",
                        "red",
                    )
                )
            else:
                print(colored(f"\nInspecting: {result['file_path']}", "cyan"))
                print(colored(f"Raw shape: {result['raw_shape']}", "blue"))
                print(colored(f"Number of pages: {result['num_pages']}", "yellow"))
                print(colored(f"Data type: {result['dtype']}", "yellow"))
                print(
                    colored(
                        f"Value range: min = {result['min_value']:.4f}, max ="
                        f" {result['max_value']:.4f}",
                        "yellow",
                    )
                )

        print(colored(f"\nInspected {len(results)} file(s).", "green"))

    def _format_image_dimensions(self, image, output_dims: str = "TCZYX"):

        input_dims = "TCZYX"
        output_dims = output_dims.upper()

        if not set(output_dims).issubset(set(input_dims)):
            raise ValueError("Output dimensions must be a subset of input dimensions")

        # Create a slicer that keeps only the dimensions we want
        slicer = tuple(slice(None) if dim in output_dims else 0 for dim in input_dims)

        # Apply the slicer
        result = image[slicer]

        # Create the transpose order based on the dimensions we kept
        kept_dims = [dim for dim in input_dims if dim in output_dims]
        transpose_order = [kept_dims.index(dim) for dim in output_dims]

        # Transpose to the desired order
        result = np.transpose(result, transpose_order)

        return result
