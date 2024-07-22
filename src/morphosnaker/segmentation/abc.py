from abc import ABC, abstractmethod
import numpy as np


class SegmentationBase(ABC):
    def __init__(self, gpu=True):
        self.gpu = gpu

    @abstractmethod
    def set_parameters(self, **parameters):
        """
        Define and set model-specific parameters.
        """
        pass

    @abstractmethod
    def segment(self, image):
        """
        Perform segmentation on the provided image.
        """
        pass