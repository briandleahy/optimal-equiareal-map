import unittest

import numpy as np

from optimal_map import *

TOLS = {'atol': 1e-15, 'rtol': 1e-15}
MEDTOLS = {'atol': 1e-6, 'rtol': 1e-6}
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

    def test_eval_dXdx_when_identity_transformation(self):
        ct = CoordinateTransform()
        np.random.seed(72)
        xold, yold = np.random.randn(2, 10)

        # Checking that the test is correct; not part of test:
        xnew, ynew = ct.eval(xold, yold)
        assert np.allclose(xnew, xold, **TOLS)
        assert np.allclose(ynew, yold, **TOLS)
        # done checking

        dxdx_matrix = ct.eval_dXdx('X', 'x')
        dydx_matrix = ct.eval_dXdx('Y', 'x')
        dxdy_matrix = ct.eval_dXdx('X', 'y')
        dydy_matrix = ct.eval_dXdx('Y', 'y')

        dxdx = np.polynomial.polynomial.polyval2d(xold, yold, dxdx_matrix)
        dydx = np.polynomial.polynomial.polyval2d(xold, yold, dydx_matrix)
        dxdy = np.polynomial.polynomial.polyval2d(xold, yold, dxdy_matrix)
        dydy = np.polynomial.polynomial.polyval2d(xold, yold, dydy_matrix)

        self.assertTrue(np.allclose(dxdx, 1, **TOLS))
        self.assertTrue(np.allclose(dydx, 0, **TOLS))
        self.assertTrue(np.allclose(dxdy, 0, **TOLS))
        self.assertTrue(np.allclose(dydy, 1, **TOLS))

    def test_eval_dXdx_via_finite_difference(self):
        ct = CoordinateTransform()
        np.random.seed(72)
        ct.update(np.random.randn(ct.params.size))
        x, y = np.random.randn(2, 10)
        dx = 1e-7

        dxdx_matrix = ct.eval_dXdx('X', 'x')
        dydx_matrix = ct.eval_dXdx('Y', 'x')
        dxdy_matrix = ct.eval_dXdx('X', 'y')
        dydy_matrix = ct.eval_dXdx('Y', 'y')
        dxdx = np.polynomial.polynomial.polyval2d(x, y, dxdx_matrix)
        dydx = np.polynomial.polynomial.polyval2d(x, y, dydx_matrix)
        dxdy = np.polynomial.polynomial.polyval2d(x, y, dxdy_matrix)
        dydy = np.polynomial.polynomial.polyval2d(x, y, dydy_matrix)

        numerical_dxdx = (ct.eval(x + dx, y)[0] - ct.eval(x, y)[0]) / dx
        numerical_dxdy = (ct.eval(x, y + dx)[0] - ct.eval(x, y)[0]) / dx
        numerical_dydx = (ct.eval(x + dx, y)[1] - ct.eval(x, y)[1]) / dx
        numerical_dydy = (ct.eval(x, y + dx)[1] - ct.eval(x, y)[1]) / dx

        self.assertTrue(np.allclose(dxdx, numerical_dxdx, **MEDTOLS))
        self.assertTrue(np.allclose(dydx, numerical_dydx, **MEDTOLS))
        self.assertTrue(np.allclose(dxdy, numerical_dxdy, **MEDTOLS))
        self.assertTrue(np.allclose(dydy, numerical_dydy, **MEDTOLS))

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


class TestLambertQuadrature(unittest.TestCase):
    def test_setup(self):
        nxpts = 25
        quad = LambertCylindricalQuadrature(nxpts=nxpts)
        self.assertTrue(quad.nxpts == nxpts)
        self.assertTrue(quad.nypts < np.ceil(nxpts / 3 + 1))

    def test_pts_returns_copy(self):
        nxpts = 25
        quad = LambertCylindricalQuadrature(nxpts=nxpts)
        pts = quad.pts
        self.assertTrue(np.allclose(pts, quad._xypts, **TOLS))
        self.assertTrue(pts is not quad._xypts)

    def test_pts_is_correct_size(self):
        nxpts = 25
        quad = LambertCylindricalQuadrature(nxpts=nxpts)
        pts = quad.pts
        self.assertTrue(pts.shape == (quad.nxpts * quad.nypts, 2))

    def test_integrate_on_constant(self):
        quad = LambertCylindricalQuadrature()
        ones = lambda x: np.ones(x.shape[0])
        surface_area = quad.integrate(ones)

        self.assertTrue(np.isclose(surface_area, 4 * np.pi, **TOLS))

    def test_integrate_on_mean_zero_function(self):
        quad = LambertCylindricalQuadrature()
        cos_theta = lambda x: x[:, 1]
        should_be_zero = quad.integrate(cos_theta)

        self.assertTrue(np.isclose(should_be_zero, 0, **TOLS))

    def test_as_sumofsquares_returns_correct_shape(self):
        quad = LambertCylindricalQuadrature()
        ones = lambda x: np.ones(x.shape[0])
        as_sumofsquares = quad.as_sumofsquares(ones)
        pts = quad.pts
        self.assertTrue(as_sumofsquares.size == pts.shape[0])

    def test_as_sumofsquares_on_constant(self):
        quad = LambertCylindricalQuadrature()
        ones = lambda x: np.ones(x.shape[0])
        as_sumofsquares = quad.as_sumofsquares(ones)
        surface_area = np.sum(as_sumofsquares**2)

        self.assertTrue(np.isclose(surface_area, 4 * np.pi, **TOLS))


if __name__ == '__main__':
    unittest.main()

