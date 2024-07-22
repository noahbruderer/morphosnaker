import os
import tempfile
import unittest
import numpy as np
import tifffile
from morphosnaker import utils

class TestImageLoader(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.TemporaryDirectory()
        # Create some temporary image files
        self.tif_file = os.path.join(self.test_dir.name, 'test_image.tif')
        self.npy_file = os.path.join(self.test_dir.name, 'test_image.npy')
        
        # Create a test TIFF file
        self.tif_image = np.random.randint(0, 1, (5, 256, 256), dtype=np.uint8)
        tifffile.imwrite(self.tif_file, self.tif_image)
        
        # Create a test NPY file
        self.npy_image = np.random.rand(10, 128, 128).astype(np.float32)
        np.save(self.npy_file, self.npy_image)
        
        # Create ImageLoader instance
        self.loader = utils.ImageProcessor()

    def tearDown(self):
        # Cleanup the temporary directory
        self.test_dir.cleanup()

    def test_inspect_tif_file(self):
        results = self.loader.inspect(self.tif_file)
        self.assertEqual(len(results), 1)
        self.assertIn('raw_shape', results[0])
        self.assertEqual(results[0]['raw_shape'], (5, 256, 256))
        self.assertEqual(results[0]['dtype'], 'uint8')

    def test_inspect_npy_file(self):
        results = self.loader.inspect(self.npy_file)
        self.assertEqual(len(results), 1)
        self.assertIn('raw_shape', results[0])
        self.assertEqual(results[0]['raw_shape'], (10, 128, 128))
        self.assertEqual(results[0]['dtype'], 'float32')

    def test_load_tif_file(self):
        image = self.loader.load(self.tif_file, input_dims='TYX')
        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(image.shape, (5, 256, 256, 1))  # Note: added channel dimension
        np.testing.assert_array_equal(image[..., 0], self.tif_image)

    def test_load_npy_file(self):
        image = self.loader.load(self.npy_file, input_dims='TYX')
        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(image.shape, (10, 128, 128, 1))  # Note: added channel dimension
        np.testing.assert_array_almost_equal(image[..., 0], self.npy_image)

    def test_inspect_directory(self):
        results = self.loader.inspect(self.test_dir.name)
        self.assertEqual(len(results), 2)
        shapes = [result['raw_shape'] for result in results if 'raw_shape' in result]
        self.assertIn((5, 256, 256), shapes)
        self.assertIn((10, 128, 128), shapes)

    def test_load_directory(self):
        images = self.loader.load(self.test_dir.name, input_dims='TYX', max_files=2)
        self.assertIsInstance(images, list)
        self.assertEqual(len(images), 2)
        shapes = [image.shape for image in images]
        self.assertIn((5, 256, 256, 1), shapes)
        self.assertIn((10, 128, 128, 1), shapes)

    def test_load_and_preprocess(self):
        images = self.loader.load_and_preprocess(self.tif_file, input_dims='TYX', 
                                                 modifications={'crop_box': (10, 100, 10, 100)})
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].shape, (5, 90, 90, 1))

    def test_load_with_different_input_dims(self):
        image = self.loader.load(self.tif_file, input_dims='TYX')
        self.assertEqual(image.shape, (5, 256, 256, 1))  # TYXC order

    def test_error_handling_invalid_file(self):
        with self.assertRaises(FileNotFoundError):
            self.loader.load('non_existent_file.tif', input_dims='TYX')

    def test_error_handling_invalid_input_dims(self):
        with self.assertRaises(AssertionError):
            self.loader.load(self.tif_file, input_dims='INVALID')

if __name__ == '__main__':
    unittest.main()