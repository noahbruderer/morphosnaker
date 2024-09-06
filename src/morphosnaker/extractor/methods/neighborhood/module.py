# extractor/methods/neighborhood.py

from typing import Dict, List, Set

import numpy as np


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
        # dims = labeled_image.ndim
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
            return self._dilate_2d(mask)
        elif dims == 3:
            return self._dilate_3d(mask)
        return mask  # CHECK THIS

    def _dilate_2d(self, mask: np.ndarray) -> np.ndarray:
        dilated = np.copy(mask)
        rows, cols = mask.shape
        for i in range(rows):
            for j in range(cols):
                if mask[i, j]:
                    if self.connectivity == 1:
                        neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                    else:  # connectivity == 2
                        neighbors = [
                            (i - 1, j),
                            (i + 1, j),
                            (i, j - 1),
                            (i, j + 1),
                            (i - 1, j - 1),
                            (i - 1, j + 1),
                            (i + 1, j - 1),
                            (i + 1, j + 1),
                        ]
                    for ni, nj in neighbors:
                        if 0 <= ni < rows and 0 <= nj < cols:
                            dilated[ni, nj] = True
        return dilated

    def _dilate_3d(self, mask: np.ndarray) -> np.ndarray:
        dilated = np.copy(mask)
        depths, rows, cols = mask.shape
        for d in range(depths):
            for i in range(rows):
                for j in range(cols):
                    if mask[d, i, j]:
                        if self.connectivity == 1:
                            neighbors = [
                                (d - 1, i, j),
                                (d + 1, i, j),
                                (d, i - 1, j),
                                (d, i + 1, j),
                                (d, i, j - 1),
                                (d, i, j + 1),
                            ]
                        else:  # connectivity == 2
                            neighbors = [
                                (d + dd, i + di, j + dj)
                                for dd in [-1, 0, 1]
                                for di in [-1, 0, 1]
                                for dj in [-1, 0, 1]
                                if not (dd == di == dj == 0)
                            ]
                        for nd, ni, nj in neighbors:
                            if 0 <= nd < depths and 0 <= ni < rows and 0 <= nj < cols:
                                dilated[nd, ni, nj] = True
        return dilated
