import numpy as np
from typing import List, Optional
from .abc import SegmentationBase
from cellpose import models as cellpose_models
# from stardist.models import StarDist2D  # Uncomment when needed

class CellposeSegmentation(SegmentationBase):
    """
    CellposeSegmentation class for segmenting cells using the Cellpose model.
    Inherits from SegmentationBase.
    """

    def __init__(self):
        """
        Initialize the CellposeSegmentation instance.

        Parameters:
        gpu (bool): Whether to use GPU for segmentation. Defaults to False.
        """
        self.parameters = {}

    def set_parameters(self, **parameters):
        """
        Set model-specific parameters for Cellpose segmentation.

        Parameters:
        model_type (str): Type of model to use ('cyto' or 'cyto3'). Defaults to 'cyto3'.
        diameter (Optional[int]): Estimated diameter of cells. Defaults to None.
        channels (List[int]): List of channels to use. Defaults to [0, 0].
        flow_threshold (Optional[float]): Flow threshold for Cellpose. Defaults to None.
        do_3D (bool): Whether to perform 3D segmentation. Defaults to False.
        stitch_threshold (Optional[float]): Stitch threshold for 3D segmentation. Defaults to None.
        """
        self.parameters['gpu'] = parameters.get('gpu', False)
        self.parameters['model_type'] = parameters.get('model_type', 'cyto3')
        self.parameters['diameter'] = parameters.get('diameter', None)
        self.parameters['channels'] = parameters.get('channels', [0, 0])
        self.parameters['flow_threshold'] = parameters.get('flow_threshold', None)
        self.parameters['do_3D'] = parameters.get('do_3D', False)
        self.parameters['stitch_threshold'] = parameters.get('stitch_threshold', 0)

        self.model = cellpose_models.Cellpose(gpu=parameters['gpu'], model_type=self.parameters['model_type'])

    def required_parameters(self) -> List[str]:
        """
        Return a list of required parameters for Cellpose segmentation.

        Returns:
        List[str]: List of required parameter names.
        """
        return ['model_type', 'channels']

    def segment(self, image: np.ndarray):
        """
        Perform segmentation on the provided image using Cellpose.

        Parameters:
        image (np.ndarray): The input image to segment.

        Returns:
        tuple: Segmentation results including masks, flows, styles, and diameters.
        """
        if not self.model:
            raise ValueError("Model parameters not set. Call 'set_parameters' first.")
        
        masks, flows, styles, diams = self.model.eval(
            image,
            diameter=self.parameters['diameter'],
            channels=self.parameters['channels'],
            flow_threshold=self.parameters['flow_threshold'],
            do_3D=self.parameters['do_3D'],
            stitch_threshold=self.parameters['stitch_threshold']
        )
        return masks, flows, styles, diams

# Uncomment when StarDistSegmentation is needed
# class StarDistSegmentation(SegmentationBase):
#     """
#     StarDistSegmentation class for segmenting cells using the StarDist model.
#     Inherits from SegmentationBase.
#     """
# 
#     def set_parameters(self, **parameters):
#         """
#         Set model-specific parameters for StarDist segmentation.
# 
#         Parameters:
#         model_name (str): Name of the pretrained model to use. Defaults to '2D_versatile_fluo'.
#         """
#         self.model_name = parameters.get('model_name', '2D_versatile_fluo')
#         self.model = StarDist2D.from_pretrained(self.model_name)
# 
#     def required_parameters(self) -> List[str]:
#         """
#         Return a list of required parameters for StarDist segmentation.
# 
#         Returns:
#         List[str]: List of required parameter names.
#         """
#         return ['model_name']
# 
#     def segment(self, image: np.ndarray):
#         """
#         Perform segmentation on the provided image using StarDist.
# 
#         Parameters:
#         image (np.ndarray): The input image to segment.
# 
#         Returns:
#         np.ndarray: Segmentation labels.
#         """
#         labels, _ = self.model.predict_instances(image)
#         return labels