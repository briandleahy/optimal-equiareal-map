import unittest

import numpy as np

from optimalmap import CoordinateTransform
from transformimage import ImageTransformer


TOLS = {'atol': 1e-11, 'rtol': 1e-11}
MEDTOLS = {'atol': 1e-6, 'rtol': 1e-6}


class TestImageTransformer(unittest.TestCase):
    def test_transform_pixel_locations_returns_correct_shape(self):
        image_transformer = create_image_transformer()
        xy_transformed = image_transformer._transform_pixel_locations()
        npix = np.prod(image_transformer.image.shape[:2])
        self.assertEqual(xy_transformed.shape[0], npix)

    def test_transform_pixel_locations_returns_correct_values(self):
        image_transformer = create_image_transformer()
        xy_transformed = image_transformer._transform_pixel_locations()
        true_xy = np.array([
            [-3830.12260646,  4074.75163398],
            [-2569.52165095,  2930.69424839],
            [-1672.48408151,  2141.656535  ],
            [-1047.19835921,  1592.69724966]])
        predicted_xy = xy_transformed[:true_xy.shape[0]]
        self.assertTrue(np.allclose(true_xy, predicted_xy, **MEDTOLS))

    def test_get_bounding_box_size_for_encloses_points(self):
        np.random.seed(3)
        points = np.random.randn(10, 2) * 100
        image_transformer = create_image_transformer()
        bbox = image_transformer._get_bounding_box_size_for(points)
        # These need to check with a swappng of 0, 1 to 1, 0:
        self.assertGreaterEqual(bbox[0], points[:, 1].ptp())
        self.assertGreaterEqual(bbox[1], points[:, 0].ptp())

    def test_get_bounding_box_size_for_is_minimal(self):
        np.random.seed(3)
        points = np.random.randn(10, 2) * 100
        image_transformer = create_image_transformer()
        bbox = image_transformer._get_bounding_box_size_for(points)
        self.assertLessEqual(bbox[0] - 1, points[:, 1].ptp())
        self.assertLessEqual(bbox[1] - 1, points[:, 0].ptp())

    def test_get_bounding_box_size_for_returns_tuple(self):
        np.random.seed(3)
        points = np.random.randn(10, 2) * 100
        image_transformer = create_image_transformer()
        bbox = image_transformer._get_bounding_box_size_for(points)
        self.assertEqual(type(bbox), tuple)


def create_image_transformer():
    image = np.ones([22, 68, 4])
    transform = create_coordinate_transform()
    return ImageTransformer(image, transform)


def create_coordinate_transform():
    np.random.seed(10)

    transform = CoordinateTransform(degree=(6, 6))
    params = np.random.randn(transform.params.size) * 0.1
    transform.update(params)
    return transform


if __name__ == '__main__':
    unittest.main()
