import unittest

import numpy as np

from optimalmap import CoordinateTransform
from transformimage import ImageTransformer


TOLS = {'atol': 1e-11, 'rtol': 1e-11}
MEDTOLS = {'atol': 1e-6, 'rtol': 1e-6}


class TestImageTransformer(unittest.TestCase):
    def test_transform_pixel_locations_returns_correct_shape(self):
        image_transformer = create_image_transformer()
        transformed_xy = image_transformer._transform_pixel_locations()
        correct_shape = image_transformer.image.shape[:2]
        self.assertEqual(transformed_xy[0].shape, correct_shape)
        self.assertEqual(transformed_xy[1].shape, correct_shape)

    def test_transform_pixel_locations_min_value_is_zero(self):
        image_transformer = create_image_transformer()
        transformed_xy = image_transformer._transform_pixel_locations()
        correct_shape = image_transformer.image.shape[:2]
        self.assertEqual(transformed_xy[0].min(), 0)
        self.assertEqual(transformed_xy[1].min(), 0)

    def test_get_bounding_box_size_for_encloses_points(self):
        np.random.seed(3)
        points = mimic_transformed_points()
        image_transformer = create_image_transformer()
        bbox = image_transformer._get_bounding_box_size_for(points)
        for i in range(2):
            self.assertGreaterEqual(bbox[i], points[i].max())

    def test_get_bounding_box_size_for_is_minimal(self):
        np.random.seed(3)
        points = mimic_transformed_points()
        image_transformer = create_image_transformer()
        bbox = image_transformer._get_bounding_box_size_for(points)
        for i in range(2):
            self.assertLessEqual(bbox[i] - 1, points[i].max())

    def test_get_bounding_box_size_for_returns_tuple(self):
        np.random.seed(3)
        points = np.random.randn(10, 2) * 100
        image_transformer = create_image_transformer()
        bbox = image_transformer._get_bounding_box_size_for(points)
        self.assertEqual(type(bbox), tuple)


def mimic_transformed_points():
    points = np.random.rand(2, 10) * 100
    points -= points.min(axis=1).reshape(-1, 1)
    return points

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
