import os
import tempfile
import numpy as np
import pytest

# Adjust the import paths to be relative to the current directory structure
from morphosnaker.segmentation.experiment_manager import SegmentationExperimentManager
from morphosnaker.segmentation.segmentation_wrapper import Segmentation


@pytest.fixture
def real_image_setup():
    with tempfile.TemporaryDirectory() as temp_dir:
        experiment_name = "integration_test_experiment"
        input_path = temp_dir
        output_dir = temp_dir
        gpu = False

        # Create a real sample input image file (e.g., a small synthetic image)
        input_image_path = os.path.join(input_path, "real_test_image.npy")
        real_image = np.random.rand(100, 100)
        np.save(input_image_path, real_image)

        experiment_manager = SegmentationExperimentManager(
            experiment_name=experiment_name,
            input_path=input_image_path,
            output_dir=output_dir,
            gpu=gpu,
            verbose=False
        )

        segmentation = Segmentation(gpu=gpu)

        yield experiment_manager, segmentation, input_image_path

def test_real_image_cellpose_segmentation(real_image_setup):
    experiment_manager, segmentation_instance, input_image_path = real_image_setup
    result = experiment_manager.save_experiment_setup(
        segmentation_instance=segmentation_instance,
        method="cellpose",
        image_path=input_image_path,
        model_type='cyto',
        diameter=30,
        channels=[0, 0]
    )

    assert result is not None
    assert os.path.exists(os.path.join(experiment_manager.output_dir, "real_test_image_segmented.tiff"))

def test_real_image_stardist_segmentation(real_image_setup):
    experiment_manager, segmentation_instance, input_image_path = real_image_setup
    result = experiment_manager.save_experiment_setup(
        segmentation_instance=segmentation_instance,
        method="stardist",
        image_path=input_image_path,
        model_name='2D_versatile_fluo'
    )

    assert result is not None
    assert os.path.exists(os.path.join(experiment_manager.output_dir, "real_test_image_segmented.tiff"))