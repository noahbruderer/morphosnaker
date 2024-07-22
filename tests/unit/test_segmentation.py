import os
import tempfile
import numpy as np
import pytest

# Adjust the import paths to be relative to the current directory structure
from morphosnaker.segmentation.experiment_manager import SegmentationExperimentManager
from morphosnaker.segmentation.segmentation_wrapper import Segmentation

# Mock Segmentation class for testing
class MockSegmentation:
    def __init__(self, gpu=True):
        self.gpu = gpu

    def load_image(self, image_path):
        return np.random.rand(100, 100)  # Mock image

    def cellpose(self, image, **parameters):
        return np.ones_like(image)  # Mock segmentation result

    def stardist(self, image, **parameters):
        return np.ones_like(image)  # Mock segmentation result

@pytest.fixture
def setup_experiment():
    with tempfile.TemporaryDirectory() as temp_dir:
        experiment_name = "test_experiment"
        input_path = temp_dir
        output_dir = temp_dir
        gpu = False

        # Create a mock input image file
        input_image_path = os.path.join(input_path, "test_image.npy")
        np.save(input_image_path, np.random.rand(100, 100))

        experiment_manager = SegmentationExperimentManager(
            experiment_name=experiment_name,
            input_path=input_image_path,
            output_dir=output_dir,
            gpu=gpu,
            verbose=False
        )

        yield experiment_manager, input_image_path

def test_cellpose_segmentation(setup_experiment):
    experiment_manager, input_image_path = setup_experiment
    segmentation_instance = MockSegmentation(gpu=False)
    result = experiment_manager.save_experiment_setup(
        segmentation_instance=segmentation_instance,
        method="cellpose",
        image_path=input_image_path,
        model_type='cyto',
        diameter=30,
        channels=[0, 0]
    )

    assert result is not None
    assert np.array_equal(result, np.ones((100, 100)))
    assert os.path.exists(os.path.join(experiment_manager.output_dir, "test_image_segmented.tiff"))

def test_stardist_segmentation(setup_experiment):
    experiment_manager, input_image_path = setup_experiment
    segmentation_instance = MockSegmentation(gpu=False)
    result = experiment_manager.save_experiment_setup(
        segmentation_instance=segmentation_instance,
        method="stardist",
        image_path=input_image_path,
        model_name='2D_versatile_fluo'
    )

    assert result is not None
    assert np.array_equal(result, np.ones((100, 100)))
    assert os.path.exists(os.path.join(experiment_manager.output_dir, "test_image_segmented.tiff"))