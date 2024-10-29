from typing import Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label


class MaskCurator:
    @staticmethod
    def fuse_small_objects(mask: np.ndarray, max_artifact_size: int) -> np.ndarray:
        """
        Identify small unsegmented objects and fuse them with the nearest neighbor.

        Args:
            mask (np.ndarray): Input segmentation mask.
            max_artifact_size (int): Maximum size of objects to be considered artifacts.

        Returns:
            np.ndarray: Updated mask with small objects fused.
        """
        # Ensure mask is 2D
        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

        # Identify small objects
        small_objects = label(np.logical_not(mask))
        sizes = np.bincount(small_objects.flatten())
        small_labels = np.where(sizes <= max_artifact_size)[0]
        small_labels = small_labels[small_labels != 0]  # Exclude background

        # Dilate small objects to find neighbors
        dilated = ndi.binary_dilation(np.isin(small_objects, small_labels))
        neighbor_labels = mask[dilated]

        # Assign each small object to a random neighbor
        for obj_label in small_labels:
            obj_mask = small_objects == obj_label
            obj_neighbors = neighbor_labels[obj_mask]
            valid_neighbors = obj_neighbors[obj_neighbors > 0]
            if len(valid_neighbors) > 0:
                new_label = np.random.choice(valid_neighbors)
                mask[obj_mask] = new_label

        return mask

    @staticmethod
    def label_large_holes(
        mask: np.ndarray, min_hole_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify large unsegmented holes and label them as new cells.

        Args:
            mask (np.ndarray): Input segmentation mask.
            min_hole_size (int): Minimum size of holes to be labeled as new cells.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Updated mask and a mask of newly labeled cells.
        """
        # Identify holes
        holes = np.logical_not(mask)
        labeled_holes, _ = label(holes, return_num=True)
        sizes = np.bincount(labeled_holes.flatten())

        # Identify large holes
        large_hole_labels = np.where(sizes >= min_hole_size)[0]
        large_hole_labels = large_hole_labels[
            large_hole_labels != 0
        ]  # Exclude background

        # Label large holes as new cells
        new_cell_mask = np.zeros_like(mask)
        new_label = mask.max() + 1
        for hole_label in large_hole_labels:
            hole_mask = labeled_holes == hole_label
            mask[hole_mask] = new_label
            new_cell_mask[hole_mask] = new_label
            new_label += 1

        return mask, new_cell_mask

    @classmethod
    def curate(
        cls, mask: np.ndarray, max_artifact_size: int, min_hole_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply curation methods to the mask.

        Args:
            mask (np.ndarray): Input segmentation mask (2D or 3D).
            max_artifact_size (int): Maximum size of objects to be considered artifacts.
            min_hole_size (int): Minimum size of holes to be labeled as new cells.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Curated mask and a mask of newly labeled cells.
        """
        print(f"Debug: Input mask shape = {mask.shape}")
        print(f"Debug: Input mask dtype = {mask.dtype}")
        print(f"Debug: Input mask min = {mask.min()}, max = {mask.max()}")

        if mask.ndim == 2:
            curated_mask = cls.fuse_small_objects(mask.copy(), max_artifact_size)
            curated_mask, new_cell_mask = cls.label_large_holes(
                curated_mask, min_hole_size
            )
        elif mask.ndim == 3:
            curated_mask = np.zeros_like(mask)
            new_cell_mask = np.zeros_like(mask)
            for i in range(mask.shape[0]):
                print(f"Debug: Processing slice {i}")
                curated_slice = cls.fuse_small_objects(
                    mask[i].copy(), max_artifact_size
                )
                curated_slice, new_cell_slice = cls.label_large_holes(
                    curated_slice, min_hole_size
                )
                curated_mask[i] = curated_slice
                new_cell_mask[i] = new_cell_slice
        else:
            raise ValueError(f"Expected 2D or 3D mask, got shape {mask.shape}")

        return curated_mask, new_cell_mask
