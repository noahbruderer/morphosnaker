# image_utils/base.py

from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np


class ImageProcessorBase(ABC):
    @abstractmethod
    def __init__(
        self,
        channels: Optional[Union[int, List[int]]] = None,
        time_points: Optional[Union[int, List[int]]] = None,
    ):
        pass

    # @abstractmethod
    # def load(
    #     self,
    #     source: Union[str, List[str], "np.ndarray", List["np.ndarray"]],
    #     input_dims: str,
    #     max_files: int = 5,
    # ):  # -> Union[np.ndarray, List[np.ndarray]]:
    #     pass

    @abstractmethod
    def save(self, image: "np.ndarray", file_path: str) -> None:
        pass
