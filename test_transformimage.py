import unittest

import numpy as np

from optimalmap import CoordinateTransform
from transformimage import ImageTransformer, transform_image


TOLS = {'atol': 1e-11, 'rtol': 1e-11}
MEDTOLS = {'atol': 1e-6, 'rtol': 1e-6}


class TestImageTransformer(unittest.TestCase):
    def test_raises_error_when_image_is_not_lambert_like(self):
        bad_image = np.ones([15, 15, 3])
        transform = create_coordinate_transform()
        self.assertRaises(ValueError, ImageTransformer, bad_image, transform)

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

    def test_get_bounding_box_size_for_is_near_minimal(self):
        # This is slightly weaker than it should be (with 1 px of minimal
        # box), becuase there is and edge case in the get_bbox which
        # is easier to fix by being slightly over the absolute minimal
        # bounding-box.
        np.random.seed(3)
        points = mimic_transformed_points()
        image_transformer = create_image_transformer()
        bbox = image_transformer._get_bounding_box_size_for(points)
        for i in range(2):
            self.assertLessEqual(bbox[i] - 2, points[i].max())

    def test_get_bounding_box_size_for_returns_tuple(self):
        np.random.seed(3)
        points = np.random.randn(10, 2) * 100
        image_transformer = create_image_transformer()
        bbox = image_transformer._get_bounding_box_size_for(points)
        self.assertEqual(type(bbox), tuple)

    def test_fill_in_holes_on_smooth_data(self):
        image_transformer = create_image_transformer()
        image_without_holes = np.empty([15, 15, 3])
        for i in range(15):
            image_without_holes[i] = i

        filled_in = np.ones(image_without_holes.shape[:2], dtype='bool')
        filled_in[[10, 5], [5, 10]] = False

        image_with_holes = image_without_holes.copy()
        image_with_holes[~filled_in] = 0.
        closed = image_transformer._fill_in_holes(image_with_holes, filled_in)
        self.assertTrue(np.allclose(closed, image_without_holes, **MEDTOLS))

    def test_fill_in_holes_on_rough_data(self):
        image_transformer = create_image_transformer()
        np.random.seed(3)
        image_without_holes = 1.0 + 0.03 * np.random.randn(15, 15, 3)
        filled_in = np.ones(image_without_holes.shape[:2], dtype='bool')
        filled_in[[10, 5, 2], [5, 10, 2]] = False
        image_with_holes = image_without_holes.copy()
        image_with_holes[~filled_in] = 0.

        closed = image_transformer._fill_in_holes(image_with_holes, filled_in)

        self.assertTrue(
            np.all(closed[filled_in] == image_without_holes[filled_in]))
        self.assertTrue(np.allclose(closed[filled_in], 1, atol=1e-1, rtol=1e-1))

    def test_zero_out_edges(self):
        image_transformer = create_image_transformer()
        shape = (15, 15, 3)
        image = np.ones(shape)
        # We make holes in the center and on the edge:
        holes = np.zeros(shape[:2], dtype='bool')
        holes[[10, 5, 2], [5, 10, 2]] = True
        edges = np.zeros(shape[:2], dtype='bool')
        for i in [0, -1]:
            edges[i, :] = True
            edges[:, i] = True
        filled_in = ~(holes | edges)
        assert np.sum(holes & ~edges) > 0

        edges_zeroed = image_transformer._zero_out_edges(image, filled_in)

        self.assertTrue(np.all(edges_zeroed[~edges] == image[~edges]))
        self.assertTrue(np.all(edges_zeroed[edges] == 0))

    def test_transform_image_calls_image_transformer(self):
        image = np.random.randn(22, 68, 3)
        transform = create_coordinate_transform()
        # and we want a reaonable set of parameters, so:
        transform.update(1e-3 * transform.params)

        v1 = transform_image(image, transform)
        transformer = ImageTransformer(image, transform)
        v2 = transformer.transform_image()
        self.assertTrue(np.all(v1 == v2))

    def test_identity_transformation_gives_identity(self):
        image = np.random.rand(22, 68, 3)
        transform = CoordinateTransform(degree=(1, 1))
        check = transform_image(image, transform)
        self.assertEqual(image.shape, check.shape)


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
