from typing import Any, List

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects
from skimage.segmentation import clear_border, watershed


class MaskCurator:
    @staticmethod
    def remove_small_objects(
        mask: np.ndarray, min_size: int = 64, connectivity: int = 1
    ) -> np.ndarray:
        """
        Remove small objects from the mask.
        """
        return remove_small_objects(mask, min_size=min_size, connectivity=connectivity)

    @staticmethod
    def fill_holes(mask: np.ndarray) -> np.ndarray:
        """
        Fill holes in the mask.
        """
        return ndi.binary_fill_holes(mask)

    @staticmethod
    def clear_border(mask: np.ndarray, buffer_size: int = 0) -> np.ndarray:
        """
        Clear objects connected to the border of the mask.
        """
        return clear_border(mask, buffer_size=buffer_size)

    @staticmethod
    def separate_touching_objects(
        mask: np.ndarray, min_distance: int = 1
    ) -> np.ndarray:
        """
        Attempt to separate touching objects using distance transform and watershed.
        """
        distance = ndi.distance_transform_edt(mask)
        local_max = ndi.maximum_filter(distance, size=min_distance) == distance
        markers, _ = ndi.label(local_max)
        labels = watershed(-distance, markers, mask=mask)
        return labels

    @staticmethod
    def label_unsegmented_holes(mask: np.ndarray, min_hole_size: int) -> np.ndarray:
        """
        Label regions that are 'holes' (i.e., unsegmented areas) based on a minimum size.
        """
        # Invert mask to find holes
        inverted_mask = np.logical_not(mask)

        # Label connected components in the inverted mask
        labeled_holes, num_features = ndi.label(inverted_mask)

        # Remove small holes (those smaller than min_hole_size)
        labeled_holes = remove_small_objects(labeled_holes, min_size=min_hole_size)

        # Return the updated mask where holes are given a new label
        return labeled_holes

    @staticmethod
    def fuse_small_holes_with_neighbors(
        mask: np.ndarray, max_hole_size: int
    ) -> np.ndarray:
        """
        Fuse small holes with their neighboring regions, creating continuous regions.
        """
        # Invert mask to detect holes
        inverted_mask = np.logical_not(mask)

        # Fill small holes (holes with size smaller than max_hole_size)
        fused_mask = ndi.binary_fill_holes(inverted_mask)
        fused_mask = remove_small_objects(fused_mask, min_size=max_hole_size)

        # Invert back to return the final mask with holes filled
        return np.logical_not(fused_mask)

    @classmethod
    def curate(cls, mask: np.ndarray, methods: List[str], **kwargs: Any) -> np.ndarray:
        """
        Apply a series of curation methods to the mask.
        """
        curated_mask = mask.copy()
        for method in methods:
            if hasattr(cls, method):
                method_kwargs = kwargs.get(method, {})
                curated_mask = getattr(cls, method)(curated_mask, **method_kwargs)
            else:
                raise ValueError(f"Unknown curation method: {method}")
        return curated_mask
