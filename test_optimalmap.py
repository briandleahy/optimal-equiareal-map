import unittest

import numpy as np

from optimal_map import *

TOLS = {'atol': 1e-15, 'rtol': 1e-15}
SOFTTOLS = {'atol': 1e-3, 'rtol': 1e-3}

# these are more system-level tests than TDD-style unit tests, since
# I wrote this up before I read clean code :)

class TestCoordinateTransform(unittest.TestCase):
    def test_transformation_intializes_to_identity(self):
        np.random.seed(72)
        ct = CoordinateTransform()
        x = np.random.randn(21)
        y = np.random.randn(x.size)
        new_x, new_y = ct.eval(x, y)
        self.assertTrue(np.allclose(new_x, x, **TOLS))
        self.assertTrue(np.allclose(new_y, y, **TOLS))

    def test_update_always_keeps_dc_term_0(self):
        np.random.seed(72)
        ct = CoordinateTransform()
        self.assertTrue(np.all(ct._coeffs[0, 0, :] == 0))
        params = ct.params
        ct.update(np.random.randn(*params.shape))
        self.assertTrue(np.all(ct._coeffs[0, 0, :] == 0))

    def test_params_returns_copy(self):
        np.random.seed(72)
        ct = CoordinateTransform()
        self.assertTrue(np.all(ct._coeffs[0, 0, :] == 0))
        ct.update(np.random.randn(*ct.params.shape))
        p1 = ct.params
        p2 = ct.params
        self.assertTrue(np.all(p1 == p2))
        self.assertTrue(p1 is not p2)

    def test_update_changes_the_evaluated_coordinates(self):
        np.random.seed(72)
        ct = CoordinateTransform()
        x = np.random.randn(21)
        y = np.random.randn(x.size)

        params = ct.params
        ct.update(np.random.randn(*params.shape))
        new_x, new_y = ct.eval(x, y)

        self.assertFalse(np.allclose(new_x, x, **SOFTTOLS))
        self.assertFalse(np.allclose(new_y, y, **SOFTTOLS))

    def _test_evaluate_metric_returns_delta_when_coeffs_are_one(self):
        ct = CoordinateTransform()
        np.random.seed(72)
        xold, yold = np.random.randn(2, 10)

        # Checking that the test is correct; not part of test:
        xnew, ynew = ct.eval(xold, yold)
        assert np.allclose(xnew, xold, **TOLS)
        assert np.allclose(ynew, yold, **TOLS)
        # done checking

        metric = ct.evaluate_metric(xold, yold)
        correct = np.eye(2).reshape(2, 2, 1)
        self.assertTrue(np.allclose(metric - correct, 0, **TOLS))

if __name__ == '__main__':
    unittest.main()

