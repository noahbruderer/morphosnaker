import os
import tempfile
import unittest

import numpy as np
import tifffile

from morphosnaker.utils import ImageProcessor


class TestImageProcessor(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.TemporaryDirectory()

        # Create some temporary image files
        self.tif_file_2d = os.path.join(self.test_dir.name, "test_image_2d.tif")
        self.tif_file_3d = os.path.join(self.test_dir.name, "test_image_3d.tif")
        self.npy_file = os.path.join(self.test_dir.name, "test_image.npy")

        # Create test TIFF files
        self.tif_image_2d = np.random.randint(0, 255, (5, 256, 256), dtype=np.uint8)
        tifffile.imwrite(self.tif_file_2d, self.tif_image_2d)

        self.tif_image_3d = np.random.randint(0, 255, (5, 10, 256, 256), dtype=np.uint8)
        tifffile.imwrite(self.tif_file_3d, self.tif_image_3d)

        # Create a test NPY file
        self.npy_image = np.random.rand(10, 128, 128).astype(np.float32)
        np.save(self.npy_file, self.npy_image)

        # Create ImageProcessor instance
        self.processor = ImageProcessor()

    def tearDown(self):
        # Cleanup the temporary directory
        self.test_dir.cleanup()

    def test_inspect_tif_file_2d(self):
        results = self.processor.inspect(self.tif_file_2d)
        self.assertEqual(len(results), 1)
        self.assertIn("raw_shape", results[0])
        self.assertEqual(results[0]["raw_shape"], (5, 256, 256))
        self.assertEqual(results[0]["dtype"], "uint8")

    def test_inspect_tif_file_3d(self):
        results = self.processor.inspect(self.tif_file_3d)
        self.assertEqual(len(results), 1)
        self.assertIn("raw_shape", results[0])
        self.assertEqual(results[0]["raw_shape"], (5, 10, 256, 256))
        self.assertEqual(results[0]["dtype"], "uint8")

    def test_inspect_npy_file(self):
        results = self.processor.inspect(self.npy_file)
        self.assertEqual(len(results), 1)
        self.assertIn("raw_shape", results[0])
        self.assertEqual(results[0]["raw_shape"], (10, 128, 128))
        self.assertEqual(results[0]["dtype"], "float32")

    def test_load_tif_file_2d(self):
        images = self.processor.load(self.tif_file_2d, input_dims="TYX")
        image = images[0] if isinstance(images, list) else images
        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(image.shape, (5, 1, 1, 256, 256))  # TCZYX order
        np.testing.assert_array_almost_equal(image[:, 0, 0, :, :], self.tif_image_2d)

    def test_load_tif_file_3d(self):
        images = self.processor.load(self.tif_file_3d, input_dims="TZYX")
        image = images[0] if isinstance(images, list) else images
        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(image.shape, (5, 1, 10, 256, 256))  # TCZYX order
        np.testing.assert_array_almost_equal(image[:, 0, :, :, :], self.tif_image_3d)

    def test_load_npy_file(self):
        images = self.processor.load(self.npy_file, input_dims="CYX")
        image = images[0] if isinstance(images, list) else images
        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(image.shape, (1, 10, 1, 128, 128))  # TCZYX order
        np.testing.assert_array_almost_equal(image[0, :, 0, :, :], self.npy_image)

    def test_inspect_directory(self):
        results = self.processor.inspect(self.test_dir.name)
        self.assertEqual(len(results), 3)
        shapes = [result["raw_shape"] for result in results if "raw_shape" in result]
        self.assertIn((5, 256, 256), shapes)
        self.assertIn((5, 10, 256, 256), shapes)
        self.assertIn((10, 128, 128), shapes)

    def test_error_handling_invalid_file(self):
        with self.assertRaises(FileNotFoundError):
            self.processor.load("non_existent_file.tif", input_dims="TYX")

    def test_error_handling_invalid_input_dims(self):
        with self.assertRaises(AssertionError):
            self.processor.load(self.tif_file_2d, input_dims="INVALID")


if __name__ == "__main__":
    unittest.main()
