from .methods.curration.module import MaskCurator
from .methods.neighborhood.module import NeighborhoodExtractor
from .methods.shapes.module import ShapeFactorExtractor


class Extractor:
    @staticmethod
    def neighborhood():
        return NeighborhoodExtractor()

    @staticmethod
    def shape_factors():
        return ShapeFactorExtractor()

    @staticmethod
    def mask_curator():
        return MaskCurator()
