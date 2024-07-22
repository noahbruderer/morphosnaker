import os
from glob import glob
import numpy as np
import tifffile
from .base import ImageProcessorBase
from typing import Union, Optional, List, Tuple
from termcolor import colored
from tqdm import tqdm

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

    def __init__(self, channels: Optional[Union[int, List[int]]] = None, time_points: Optional[Union[int, List[int]]] = None):
        """
        Initialize the ImageProcessorMethod.

        Args:
            channels (Optional[Union[int, List[int]]]): Channels to process. If None, all channels are processed.
            time_points (Optional[Union[int, List[int]]]): Time points to process. If None, all time points are processed.
        """
        self.channels = channels if isinstance(channels, list) or channels is None else [channels]
        self.time_points = time_points if isinstance(time_points, list) or time_points is None else [time_points]

    def _get_file_list(self, source: Union[str, List[str]], max_files: Optional[int] = None) -> List[str]:
        """
        Get a list of image files from the source.

        Args:
            source (Union[str, List[str]]): Path to a file, directory, or list of file paths.
            max_files (Optional[int]): Maximum number of files to return.

        Returns:
            List[str]: List of file paths.

        Raises:
            ValueError: If the source is neither a valid file nor a directory.
            TypeError: If the source is not a string or list.
        """
        try:
            if isinstance(source, str):
                if os.path.isdir(source):
                    file_list = sorted(glob(os.path.join(source, "*.tif")) + 
                                    glob(os.path.join(source, "*.tiff")) +
                                    glob(os.path.join(source, "*.npy")))
                    return file_list[:max_files] if max_files is not None else file_list
                elif os.path.isfile(source):
                    return [source]
                else:
                    if not os.path.exists(source):
                        raise FileNotFoundError(f"Source '{source}' does not exist.")      
                    else:
                        raise ValueError(f"Source '{source}' is neither a valid file nor a directory.")
            elif isinstance(source, list):
                return source[:max_files] if max_files is not None else source
            else:
                raise TypeError(f"Expected str or list, got {type(source)}")
        except Exception as e:
            print(colored(f"Error in _get_file_list: {str(e)}", "red"))
            raise

        except Exception as e:
            print(colored(f"Error in _get_file_list: {str(e)}", "red"))
            raise

    def inspect(self, source: Union[str, List[str]], max_files: int = 5, print_results: bool = True) -> List[dict]:
        """
        Inspect image files without fully loading them into memory.

        Args:
            source (Union[str, List[str]]): Path to a file, directory, or list of file paths.
            max_files (int): Maximum number of files to inspect.
            print_results (bool): Whether to print the inspection results.

        Returns:
            List[dict]: List of dictionaries containing inspection results for each file.

        Raises:
            Exception: If an error occurs during inspection.
        """
        try:
            file_list = self._get_file_list(source, max_files)
            results = []
            for file_path in tqdm(file_list, desc="Inspecting files"):
                if os.path.isfile(file_path) and file_path.lower().endswith(('.tif', '.tiff', '.npy')):
                    results.append(self._inspect_single_file(file_path))
                else:
                    results.append({"file_path": file_path, "error": "Not a supported file or doesn't exist"})

            if print_results:
                self._print_inspection_results(results)

            return results
        except Exception as e:
            print(colored(f"Error during inspection: {str(e)}", "red"))
            raise

    def load(self, source: Union[str, List[str]], input_dims: str = 'auto', max_files: Optional[int] = None) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Load image(s) from the source.

        Args:
            source (Union[str, List[str]]): Path to a file, directory, or list of file paths.
            input_dims (str): The dimension order of the input images. Use 'auto' for automatic detection.
            max_files (Optional[int]): Maximum number of files to load.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: Loaded and processed image(s).

        Raises:
            Exception: If an error occurs during loading or processing.
        """
        try:
            file_list = self._get_file_list(source, max_files)
            if len(file_list) == 1:
                return self._load_and_process_file(file_list[0], input_dims)
            else:
                return [self._load_and_process_file(file_path, input_dims) for file_path in tqdm(file_list, desc="Loading files")]
        except Exception as e:
            print(colored(f"Error during loading: {str(e)}", "red"))
            raise


    def select_dimensions(self, image: np.ndarray, 
                        channels: Union[int, List[int]] = None, 
                        time_points: Union[int, List[int]] = None,
                        z_slices: Union[int, List[int]] = None) -> np.ndarray:
        """
        Select specific channels, time points, and Z slices from the image while maintaining dimensionality.
        
        Args:
        image (np.ndarray): Input image with shape (T, Y, X, C) or (T, Z, Y, X, C).
        channels (Union[int, List[int]], optional): Channel(s) to select. If None, all channels are kept.
        time_points (Union[int, List[int]], optional): Time point(s) to select. If None, all time points are kept.
        z_slices (Union[int, List[int]], optional): Z slice(s) to select. If None, all Z slices are kept.
        
        Returns:
        np.ndarray: Image with selected channels, time points, and Z slices.
        
        Raises:
        ValueError: If the input image doesn't have the expected number of dimensions.
        """
        if image.ndim not in [4, 5]:
            raise ValueError("Input image must have 4 (T, Y, X, C) or 5 (T, Z, Y, X, C) dimensions.")

        # Convert single integers to lists for consistent processing
        channels = [channels] if isinstance(channels, int) else channels
        time_points = [time_points] if isinstance(time_points, int) else time_points
        z_slices = [z_slices] if isinstance(z_slices, int) else z_slices

        # Select time points
        if time_points is not None:
            image = image[time_points]

        # Select Z slices if the image is 5D
        if z_slices is not None and image.ndim == 5:
            image = image[:, z_slices]

        # Select channels
        if channels is not None:
            image = image[..., channels]

        # Ensure dimensions are maintained
        if image.ndim == 3:  # (Y, X, C) -> (1, Y, X, C)
            image = image[np.newaxis, ...]
        elif image.ndim == 4 and image.shape[-1] != 1:  # (T, Z, Y, X) -> (T, Z, Y, X, 1)
            image = np.expand_dims(image, axis=-1)

        return image


    def _inspect_single_file(self, file_path: str) -> dict:
        """
        Inspect a single image file and return its properties.

        This method loads the file, extracts key information such as shape, data type,
        number of pages (for multi-page TIFF files), and value range.

        Args:
            file_path (str): Path to the image file.

        Returns:
            dict: A dictionary containing the file's properties.
                Keys include:
                - 'raw_shape': The shape of the loaded image array.
                - 'file_path': The path to the file.
                - 'num_pages': The number of pages (for TIFF files, 1 for others).
                - 'dtype': The data type of the image.
                - 'min_value': The minimum pixel value in the image.
                - 'max_value': The maximum pixel value in the image.

        Raises:
            Exception: If there's an error loading or inspecting the file.
        """
        try:
            image = self._load_file(file_path)
            raw_shape = image.shape
            
            if file_path.lower().endswith(('.tif', '.tiff')):
                with tifffile.TiffFile(file_path) as tif:
                    num_pages = len(tif.pages)
                    first_page = tif.pages[0]
                    dtype = first_page.dtype
            else:
                num_pages = 1
                dtype = image.dtype

            min_val = image.min()
            max_val = image.max()

            return {
                "raw_shape": raw_shape,
                "file_path": file_path,
                "num_pages": num_pages,
                "dtype": str(dtype),
                "min_value": float(min_val),
                "max_value": float(max_val)
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
                print(colored(f"Error inspecting {result['file_path']}: {result['error']}", "red"))
            else:
                print(colored(f"\nInspecting: {result['file_path']}", "cyan"))
                print(colored(f"Raw shape: {result['raw_shape']}", "blue"))
                print(colored(f"Number of pages: {result['num_pages']}", "yellow"))
                print(colored(f"Data type: {result['dtype']}", "yellow"))
                print(colored(f"Value range: min = {result['min_value']:.4f}, max = {result['max_value']:.4f}", "yellow"))

        print(colored(f"\nInspected {len(results)} file(s).", "green"))
        
    def _load_and_process_file(self, file_path: str, input_dims: str) -> np.ndarray:
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
            if input_dims == 'auto':
                input_dims = self._guess_dimensions(image)
            return self._process_image(image, input_dims)
        except Exception as e:
            print(colored(f"Error processing file {file_path}: {str(e)}", "red"))
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
            if file_path.endswith('.npy'):
                return np.load(file_path)
            elif file_path.lower().endswith(('.tif', '.tiff')):
                return tifffile.imread(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}. Supported formats are .npy, .tif, and .tiff")
        except Exception as e:
            print(colored(f"Error loading file {file_path}: {str(e)}", "red"))
            raise

    def _process_image(self, image: np.ndarray, input_dims: str) -> np.ndarray:
        """
        Process a loaded image: select channels, standardize dimensions, and normalize.

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
            if self.channels is not None:
                if image.ndim < 3 or image.shape[-1] <= max(self.channels):
                    raise ValueError(f"Cannot select channels {self.channels} from image with shape {image.shape}")
                image = image[..., self.channels]

            image = self._standardize_dimensions(image, input_dims)

            if image.dtype != np.float32 or image.max() > 1.0:
                image = image.astype(np.float32)
                min_val = image.min()
                max_val = image.max()
                if min_val != max_val:
                    image = (image - min_val) / (max_val - min_val)
                else:
                    image = np.zeros_like(image)

            return image
        except Exception as e:
            print(colored(f"Error processing image: {str(e)}", "red"))
            raise

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
            if file_path.endswith(('.tif', '.tiff')):
                tifffile.imwrite(file_path, image)
            elif file_path.endswith('.npy'):
                np.save(file_path, image)
            else:
                raise ValueError(f"Unsupported file format for saving: {file_path}")
            print(colored(f"Successfully saved image to {file_path}", "green"))
        except Exception as e:
            print(colored(f"Error saving file {file_path}: {str(e)}", "red"))
            raise

    def _standardize_dimensions(self, image: np.ndarray, input_dims: str) -> np.ndarray:
        """
        Standardize image dimensions to TYXC (for 2D) or TZYXC (for 3D).

        Args:
            image (np.ndarray): Input image.
            input_dims (str): The dimension order of the input image.

        Returns:
            np.ndarray: Image with standardized dimensions.

        Raises:
            AssertionError: If the number of dimensions doesn't match the specified input_dims.
        """
        print(colored(f"Input shape: {image.shape}, Input dims: {input_dims}", "blue"))

        assert len(image.shape) == len(input_dims), "Number of dimensions doesn't match the specified input_dims"

        is_2d = 'Z' not in input_dims
        target_dims = 'TYXC' if is_2d else 'TZYXC'

        for dim in target_dims:
            if dim not in input_dims:
                image = np.expand_dims(image, axis=-1 if dim in 'ZC' else 0)
                input_dims = ('T' + input_dims) if dim == 'T' else (input_dims + dim)

        print(colored(f"After adding dimensions: shape={image.shape}, dims={input_dims}", "yellow"))

        transpose_order = [input_dims.index(dim) for dim in target_dims]
        print(colored(f"Transpose order: {transpose_order}", "yellow"))

        final_image = np.transpose(image, transpose_order)
        print(colored(f"Final shape: {final_image.shape}", "green"))

        return final_image

    def _guess_dimensions(self, image: np.ndarray) -> str:
        """
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
            return 'YX'
        elif len(shape) == 3:
            return 'ZYX' if shape[0] < shape[1] and shape[0] < shape[2] else 'TYX'
        elif len(shape) == 4:
            return 'ZYXC' if shape[0] < shape[1] and shape[0] < shape[2] else 'TYXC'
        elif len(shape) == 5:
            return 'TZYXC'
        else:
            raise ValueError(f"Unable to guess dimensions for shape {shape}")

    def crop(self, image: np.ndarray, crop_box: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Crop the spatial dimensions of the image.

        Args:
            image (np.ndarray): Input image.
            crop_box (Tuple[int, int, int, int]): Crop box coordinates (y_start, y_end, x_start, x_end).

        Returns:
            np.ndarray: Cropped image.

        Raises:
            ValueError: If the image dimensions are not 4D or 5D.
        """
        y_start, y_end, x_start, x_end = crop_box
        if image.ndim == 4:  # (T, Y, X, C)
            return image[:, y_start:y_end, x_start:x_end, :]
        elif image.ndim == 5:  # (T, Z, Y, X, C)
            return image[:, :, y_start:y_end, x_start:x_end, :]
        else:
            raise ValueError("Image must be 4D (T, Y, X, C) or 5D (T, Z, Y, X, C)")

    def normalize(self, image: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize the image using the specified method.

        Args:
            image (np.ndarray): Input image.
            method (str): Normalization method ('minmax' or 'zscore').

        Returns:
            np.ndarray: Normalized image.

        Raises:
            ValueError: If an unsupported normalization method is specified.
        """
        if method == 'minmax':
            return (image - image.min()) / (image.max() - image.min())
        elif method == 'zscore':
            return (image - np.mean(image)) / np.std(image)
        else:
            raise ValueError(f"Unsupported normalization method: {method}")

    def apply_modifications(self, image: np.ndarray, modifications: dict) -> np.ndarray:
        """
        Apply a series of modifications to the image.

        Args:
            image (np.ndarray): Input image.
            modifications (dict): Dictionary of modifications to apply.

        Returns:
            np.ndarray: Modified image.
        """
        if 'crop_box' in modifications:
            image = self.crop(image, modifications['crop_box'])
        if 'normalize' in modifications:
            image = self.normalize(image, modifications['normalize'])
        # Add more modifications as needed
        return image