import numpy as np
from .segmentation_methods import CellposeSegmentation  # Import StarDistSegmentation if needed
from ..model_training.model_training_methods import CellposeTrainModel


class Segmentation:
    """
    Segmentation class to handle different segmentation methods.
    """

    def __init__(self, gpu: bool = True):
        """
        Initialize the Segmentation instance.

        Parameters:
        gpu (bool): Whether to use GPU for segmentation. Defaults to True.
        """
        self.cellpose_segmenter = CellposeSegmentation()
        self.cellpose_train_model = CellposeTrainModel()

        # self.stardist_segmenter = StarDistSegmentation(gpu=gpu)

    def _validate_parameters(self, segmenter, parameters: dict):
        """
        Validate the provided parameters against the required parameters for the segmenter.

        Parameters:
        segmenter (SegmentationBase): The segmentation method instance.
        parameters (dict): Dictionary of parameters to validate.

        Raises:
        ValueError: If any required parameter is missing.
        """
        required_params = segmenter.required_parameters()
        for param in required_params:
            if param not in parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
    def cellpose_segment(self, image: np.ndarray, **parameters):
        """
        Perform segmentation using the Cellpose method.

        Parameters:
        image (np.ndarray): The input image to segment.
        **parameters: Additional parameters required for Cellpose segmentation.

        Returns:
        tuple: Segmentation results including masks, flows, styles, and diameters.

        Raises:
        ValueError: If any required parameter is missing.
        """
        self._validate_parameters(self.cellpose_segmenter, parameters)
        self.cellpose_segmenter.set_parameters(**parameters)
        return self.cellpose_segmenter.segment(image)

    # Uncomment and complete this when StarDistSegmentation is defined
    # def stardist(self, image: np.ndarray, **parameters):
    #     """
    #     Perform segmentation using the StarDist method.
    #
    #     Parameters:
    #     image (np.ndarray): The input image to segment.
    #     **parameters: Additional parameters required for StarDist segmentation.
    #
    #     Returns:
    #     np.ndarray: Segmentation labels.
    #
    #     Raises:
    #     ValueError: If any required parameter is missing.
    #     """
    #     self._validate_parameters(self.stardist_segmenter, parameters)
    #     self.stardist_segmenter.set_parameters(**parameters)
    #     return self.stardist_segmenter.segment(image)