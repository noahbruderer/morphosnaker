from typing import Dict, List, Set

import numpy as np
from scipy.ndimage import binary_dilation


class NeighborhoodExtractor:
    def __init__(self, connectivity: int = 1):
        """
        Initialize the NeighborhoodExtractor.

        Args:
            connectivity (int): Defines the connectivity of the neighborhood.
                1 for 4-connectivity in 2D or 6-connectivity in 3D,
                2 for 8-connectivity in 2D or 26-connectivity in 3D.
        """
        self.connectivity = connectivity

    def extract(self, labeled_image: np.ndarray) -> Dict[int, List[int]]:
        """
        Extract neighborhood information from a labeled image.

        Args:
            labeled_image (np.ndarray): A 2D or 3D labeled image where each cell has a
            unique integer label.

        Returns:
            Dict[int, List[int]]: A dictionary where keys are cell labels and values are
            lists of neighboring cell labels.
        """
        dims = labeled_image.ndim
        if dims not in [2, 3]:
            raise ValueError("Input image must be 2D or 3D")

        neighborhood = {}
        unique_labels = np.unique(labeled_image)
        unique_labels = unique_labels[unique_labels != 0]  # Exclude background

        for label in unique_labels:
            neighbors = self._find_neighbors(labeled_image, label)
            neighborhood[label] = list(neighbors)

        return neighborhood

    def _find_neighbors(self, labeled_image: np.ndarray, label: int) -> Set[int]:
        """
        Find the neighbors of a given cell in the labeled image.

        Args:
            labeled_image (np.ndarray): The labeled image.
            label (int): The label of the cell to find neighbors for.

        Returns:
            Set[int]: A set of labels of neighboring cells.
        """
        cell_mask = labeled_image == label
        dilated = self._dilate(cell_mask)
        neighbor_mask = dilated & (labeled_image != label)
        neighbors = set(np.unique(labeled_image[neighbor_mask])) - {0}
        return neighbors

    def _dilate(self, mask: np.ndarray) -> np.ndarray:
        """
        Dilate the input mask based on the connectivity.

        Args:
            mask (np.ndarray): Input binary mask.

        Returns:
            np.ndarray: Dilated mask.
        """
        dims = mask.ndim
        if dims == 2:
            structure = (
                np.ones((3, 3))
                if self.connectivity == 2
                else np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            )
        elif dims == 3:
            structure = (
                np.ones((3, 3, 3))
                if self.connectivity == 2
                else np.array(
                    [
                        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    ]
                )
            )
        else:
            raise ValueError("Unsupported dimensionality for dilation")

        # Perform dilation using scipy's efficient binary_dilation
        dilated_mask = binary_dilation(mask, structure=structure)
        return dilated_mask
