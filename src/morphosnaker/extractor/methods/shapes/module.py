from typing import Any, Dict, Optional

import numpy as np
from skimage.measure import regionprops  # type: ignore


class ShapeFactorExtractor:
    def __init__(self, mode: str = "3D"):
        if mode not in ["2D", "3D"]:
            raise ValueError("Mode must be either '2D' or '3D'")
        self.mode = mode

    def extract(
        self, labeled_image: np.ndarray, embryo_name: Optional[str] = None
    ) -> Dict[int, Dict[str, Any]]:
        unique_labels = np.unique(labeled_image)
        unique_labels = unique_labels[unique_labels != 0]  # Exclude background

        # Get the bounding box of the entire embryo
        embryo_mask = np.isin(labeled_image, unique_labels)
        embryo_props = regionprops(embryo_mask.astype(int))[0]
        bbox_min = embryo_props.bbox[:3] if self.mode == "3D" else embryo_props.bbox[:2]

        shape_factors = {}
        for label in unique_labels:
            cell_data = self._calculate_shape_factors(
                labeled_image, label, bbox_min, embryo_name
            )
            if cell_data:
                shape_factors[label] = cell_data

        return shape_factors

    def _calculate_shape_factors(
        self,
        image: np.ndarray,
        label_id: int,
        bbox_min: tuple,
        embryo_name: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        try:
            mask = image == label_id
            region = regionprops(mask.astype(int))[0]
            centroid = region.centroid

            # Calculate shape factors
            major_axis_length = region.major_axis_length
            minor_axis_length = region.minor_axis_length
            area = region.area
            convex_area = region.convex_area

            aspect_ratio = (
                major_axis_length / minor_axis_length
                if minor_axis_length != 0
                else np.nan
            )
            compactness = (area ** (1.5)) / convex_area if convex_area != 0 else np.nan
            roundness = (
                (4 * area) / (np.pi * (major_axis_length**2))
                if major_axis_length != 0
                else np.nan
            )
            elongation = (
                major_axis_length / minor_axis_length
                if minor_axis_length != 0
                else np.nan
            )

            # Adjust centroid coordinates
            centroid_adjusted = np.array(centroid) - np.array(bbox_min)
            local_centroid = np.array(centroid) - np.array(region.bbox[: len(centroid)])

            region_data = {
                "area": area,
                "solidity": region.solidity,
                "major_axis_length": major_axis_length,
                "minor_axis_length": minor_axis_length,
                "equiv_diameter": region.equivalent_diameter,
                "extent": region.extent,
                "feret_diameter_max": region.feret_diameter_max,
                "bounding_box_area": region.bbox_area,
                "aspect_ratio": aspect_ratio,
                "compactness": compactness,
                "roundness": roundness,
                "elongation": elongation,
                "convex_area": convex_area,
            }

            # Add centroid data
            for i, dim in enumerate(["x", "y", "z"][: len(centroid)]):
                region_data[f"centroid_{dim}_image"] = centroid[i]
                region_data[f"centroid_{dim}_embryo"] = centroid_adjusted[i]
                region_data[f"centroid_{dim}_cell"] = local_centroid[i]

            if self.mode == "3D":
                region_data["volume"] = np.sum(mask)

            if embryo_name:
                region_data["Embryo"] = embryo_name

            return region_data

        except Exception as e:
            print(f"Error processing cell {label_id}: {str(e)}")
            return None
