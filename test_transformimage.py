import unittest

import numpy as np

from optimalmap import CoordinateTransform
from transformimage import ImageTransformer


TOLS = {'atol': 1e-11, 'rtol': 1e-11}
MEDTOLS = {'atol': 1e-6, 'rtol': 1e-6}


class TestImageTransformer(unittest.TestCase):
    def test_get_transformed_pixel_locations_returns_correct_shape(self):
        image_transformer = create_image_transformer()
        xy_transformed = image_transformer._get_transformed_pixel_locations()
        npix = np.prod(image_transformer.image.shape[:2])
        self.assertEqual(xy_transformed.shape[0], npix)

    def test_get_transformed_pixel_locations_returns_correct_values(self):
        image_transformer = create_image_transformer()
        xy_transformed = image_transformer._get_transformed_pixel_locations()
        true_xy = np.array([
            [-3830.12260646,  4074.75163398],
            [-2569.52165095,  2930.69424839],
            [-1672.48408151,  2141.656535  ],
            [-1047.19835921,  1592.69724966]])
        predicted_xy = xy_transformed[:true_xy.shape[0]]
        self.assertTrue(np.allclose(true_xy, predicted_xy, **MEDTOLS))




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
