import unittest

import numpy as np

from optimalmap import *

TOLS = {'atol': 1e-15, 'rtol': 1e-15}
MEDTOLS = {'atol': 1e-6, 'rtol': 1e-6}
SOFTTOLS = {'atol': 1e-3, 'rtol': 1e-3}


class TestCoordinateTransform(unittest.TestCase):
    def test_transformation_intializes_to_identity(self):
        np.random.seed(72)
        ct = CoordinateTransform()
        x = np.random.randn(21)
        y = np.random.randn(x.size)
        new_x, new_y = ct.evaluate(x, y)
        self.assertTrue(np.allclose(new_x, x, **TOLS))
        self.assertTrue(np.allclose(new_y, y, **TOLS))

    def test_coeffs_are_correct_shape(self):
        np.random.seed(72)
        degree = (5, 6)
        correct_shape = tuple([d + 1 for d in degree])
        ct = CoordinateTransform(degree=degree)
        self.assertEqual(ct._coeffs[0].shape, correct_shape)
        self.assertEqual(ct._coeffs[1].shape, correct_shape)

    def test_create_mask_returns_correct_shape(self):
        np.random.seed(3)
        degree = tuple(np.random.randint(0, high=10, size=2))
        ct = CoordinateTransform(degree=degree)
        mask = ct._create_mask_for_parameters()
        self.assertEqual(mask.shape, ct._coeffs.shape)

    def test_update_always_keeps_dc_term_0(self):
        np.random.seed(72)
        ct = CoordinateTransform()
        self.assertTrue(np.all(ct._coeffs[:, 0, 0] == 0))
        params = ct.params
        ct.update(np.random.randn(*params.shape))
        self.assertTrue(np.all(ct._coeffs[:, 0, 0] == 0))

    def test_params_returns_copy(self):
        np.random.seed(72)
        ct = CoordinateTransform()
        ct.update(np.random.randn(*ct.params.shape))
        p1 = ct.params
        p2 = ct.params
        self.assertTrue(np.all(p1 == p2))
        self.assertTrue(p1 is not p2)

    def test_params_are_correct_size(self):
        degree = (6, 7)
        ct = CoordinateTransform(degree=degree)
        self.assertEqual(ct.params.size, ct._coeffs.size - 4)

    def test_update_changes_the_evaluated_coordinates(self):
        np.random.seed(72)
        ct = CoordinateTransform()
        x = np.random.randn(21)
        y = np.random.randn(x.size)

        params = ct.params
        ct.update(np.random.randn(*params.shape))
        new_x, new_y = ct.evaluate(x, y)

        self.assertFalse(np.allclose(new_x, x, **SOFTTOLS))
        self.assertFalse(np.allclose(new_y, y, **SOFTTOLS))

    def test_evaluate_derivative_when_identity_transformation(self):
        ct = CoordinateTransform()
        np.random.seed(72)
        xold, yold = np.random.randn(2, 10)

        # Checking that the test is correct; not part of test:
        xnew, ynew = ct.evaluate(xold, yold)
        assert np.allclose(xnew, xold, **TOLS)
        assert np.allclose(ynew, yold, **TOLS)
        # done checking

        dxdx_matrix = ct.evaluate_derivative('X', 'x')
        dydx_matrix = ct.evaluate_derivative('Y', 'x')
        dxdy_matrix = ct.evaluate_derivative('X', 'y')
        dydy_matrix = ct.evaluate_derivative('Y', 'y')

        dxdx = np.polynomial.polynomial.polyval2d(xold, yold, dxdx_matrix)
        dydx = np.polynomial.polynomial.polyval2d(xold, yold, dydx_matrix)
        dxdy = np.polynomial.polynomial.polyval2d(xold, yold, dxdy_matrix)
        dydy = np.polynomial.polynomial.polyval2d(xold, yold, dydy_matrix)

        self.assertTrue(np.allclose(dxdx, 1, **TOLS))
        self.assertTrue(np.allclose(dydx, 0, **TOLS))
        self.assertTrue(np.allclose(dxdy, 0, **TOLS))
        self.assertTrue(np.allclose(dydy, 1, **TOLS))

    def test_evaluate_derivative_via_finite_difference(self):
        ct = CoordinateTransform()
        np.random.seed(72)
        ct.update(np.random.randn(ct.params.size))
        x, y = np.random.randn(2, 10)
        dx = 1e-8

        dxdx_matrix = ct.evaluate_derivative('X', 'x')
        dydx_matrix = ct.evaluate_derivative('Y', 'x')
        dxdy_matrix = ct.evaluate_derivative('X', 'y')
        dydy_matrix = ct.evaluate_derivative('Y', 'y')
        dxdx = np.polynomial.polynomial.polyval2d(x, y, dxdx_matrix)
        dydx = np.polynomial.polynomial.polyval2d(x, y, dydx_matrix)
        dxdy = np.polynomial.polynomial.polyval2d(x, y, dxdy_matrix)
        dydy = np.polynomial.polynomial.polyval2d(x, y, dydy_matrix)

        numerical_dxdx = (
            (ct.evaluate(x + dx, y)[0] - ct.evaluate(x, y)[0]) / dx)
        numerical_dxdy = (
            (ct.evaluate(x, y + dx)[0] - ct.evaluate(x, y)[0]) / dx)
        numerical_dydx = (
            (ct.evaluate(x + dx, y)[1] - ct.evaluate(x, y)[1]) / dx)
        numerical_dydy = (
            (ct.evaluate(x, y + dx)[1] - ct.evaluate(x, y)[1]) / dx)

        self.assertTrue(np.allclose(dxdx, numerical_dxdx, **MEDTOLS))
        self.assertTrue(np.allclose(dydx, numerical_dydx, **MEDTOLS))
        self.assertTrue(np.allclose(dxdy, numerical_dxdy, **MEDTOLS))
        self.assertTrue(np.allclose(dydy, numerical_dydy, **MEDTOLS))


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
        self.assertEqual(pts.shape, (quad.nxpts * quad.nypts, 2))

    def test_wts_returns_copy(self):
        nxpts = 25
        quad = LambertCylindricalQuadrature(nxpts=nxpts)
        wts = quad.wts
        self.assertTrue(np.allclose(wts, quad._xywts, **TOLS))
        self.assertTrue(wts is not quad._xywts)

    def test_wts_is_correct_size(self):
        nxpts = 25
        quad = LambertCylindricalQuadrature(nxpts=nxpts)
        self.assertEqual(quad.pts.shape, quad.wts.shape + (2,))

    def test_sqrt_wts_is_correct_size(self):
        nxpts = 25
        quad = LambertCylindricalQuadrature(nxpts=nxpts)
        self.assertEqual(quad.wts.shape, quad.sqrt_wts.shape)

    def test_sqrt_wts_returns_copy(self):
        nxpts = 25
        quad = LambertCylindricalQuadrature(nxpts=nxpts)
        self.assertTrue(quad.sqrt_wts is not quad._xywts_sqrt)

    def test_sqrt_wts_returns_correct_values(self):
        nxpts = 25
        quad = LambertCylindricalQuadrature(nxpts=nxpts)
        self.assertTrue(np.allclose(quad.sqrt_wts, np.sqrt(quad.wts), **TOLS))

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

    def test_integrate_as_sumofsquares_returns_correct_shape(self):
        quad = LambertCylindricalQuadrature()
        ones = lambda x: np.ones(x.shape[0])
        as_sumofsquares = quad.integrate_as_sumofsquares(ones)
        pts = quad.pts
        self.assertTrue(as_sumofsquares.size == pts.shape[0])

    def test_integrate_as_sumofsquares_on_constant(self):
        quad = LambertCylindricalQuadrature()
        ones = lambda x: np.ones(x.shape[0])
        as_sumofsquares = quad.integrate_as_sumofsquares(ones)
        surface_area = np.sum(as_sumofsquares**2)

        self.assertTrue(np.isclose(surface_area, 4 * np.pi, **TOLS))


class TestLambertProjection(unittest.TestCase):
    def test_init_gives_diagonal_metric(self):
        np.random.seed(72)
        xypts = np.random.randn(200, 2)
        lambert = LambertProjection(xypts)
        self.assertTrue(np.all(lambert.metric[:, 0, 1] == 0))
        self.assertTrue(np.all(lambert.metric[:, 1, 0] == 0))

    def test_init_gives_metric_with_determinant_of_one(self):
        np.random.seed(72)
        xypts = np.random.randn(200, 2)
        lambert = LambertProjection(xypts)
        det = np.linalg.det(lambert.metric)
        self.assertTrue(np.allclose(det, 1, **TOLS))


class TestMetricCostEvaluator(unittest.TestCase):
    def test_calculate_metric_when_no_transformation(self):
        fitter = MetricCostEvaluator()
        old_metric = fitter.lambert_projection.metric
        new_metric = fitter._calculate_metric()
        self.assertTrue(np.allclose(old_metric, new_metric, **TOLS))

    def test_params(self):
        fitter = MetricCostEvaluator(area_penalty=1.0)
        self.assertTrue(np.all(fitter.params == fitter.transform.params))

    def test_update_calls_transform(self):
        fitter = MetricCostEvaluator()
        transform = fitter.transform
        old_params = transform.params.copy()

        np.random.seed(72)
        new_params = np.random.randn(old_params.size)
        fitter.update(new_params)
        updated_params = transform.params
        self.assertTrue(np.all(new_params == updated_params))

    def test_update_area_penalty(self):
        fitter = MetricCostEvaluator(area_penalty=1.0)
        assert fitter.area_penalty == 1.0
        new_penalty = 2.0
        fitter.update_area_penalty(new_penalty)
        self.assertTrue(fitter.area_penalty == new_penalty)

    def test_call_uses_area_penalty(self):
        fitter = MetricCostEvaluator(area_penalty=1.0)
        equiareal_params = fitter.params
        params = (
            equiareal_params + 1e-2 * np.random.randn(equiareal_params.size))

        fitter.update_area_penalty(0.0)
        vals_0 = fitter.call(params)
        fitter.update_area_penalty(2.0)
        vals_2 = fitter.call(params)
        self.assertTrue(np.linalg.norm(vals_0) < np.linalg.norm(vals_2))

    def test_area_penalty_does_not_effect_equiareal(self):
        fitter = MetricCostEvaluator(area_penalty=1.0)
        params = fitter.params

        fitter.update_area_penalty(0.0)
        vals_0 = fitter.call(params)
        fitter.update_area_penalty(2.0)
        vals_2 = fitter.call(params)
        self.assertTrue(
            np.isclose(np.linalg.norm(vals_0), np.linalg.norm(vals_2), **TOLS))

    @unittest.expectedFailure  # fails because of singularity at the poles
    def test_cost_is_invariant_to_nquadpts(self):
        fitter10 = MetricCostEvaluator(area_penalty=3.0, nquadpts=10)
        fitter50 = MetricCostEvaluator(area_penalty=3.0, nquadpts=50)
        params = fitter10.params

        cost10 = np.sum(fitter10.call(params)**2)
        cost50 = np.sum(fitter50.call(params)**2)
        self.assertAlmostEqual(cost10, cost50, places=7)

    @unittest.expectedFailure  # fails because of singularity at the poles
    def test_area_cost_is_invariant_to_nquadpts(self):
        fitter10 = MetricCostEvaluator(area_penalty=3.0, nquadpts=10)
        fitter20 = MetricCostEvaluator(area_penalty=3.0, nquadpts=20)
        params = fitter10.params
        # We want the area to be not-conserved, so:
        np.random.seed(2)
        params += np.random.randn(params.size) * 1e-2
        fitter10.update(params)
        fitter20.update(params)

        metric_error10, area_error10 = fitter10.calculate_metric_residuals()
        metric_error20, area_error20 = fitter20.calculate_metric_residuals()

        area_cost_10 = np.sum(area_error10**2)
        area_cost_20 = np.sum(area_error20**2)
        self.assertAlmostEqual(area_cost_10, area_cost_20, places=7)


class TestMisc(unittest.TestCase):
    def test_l2av_with_ones(self):
        t = np.ones(10)
        self.assertTrue(np.isclose(l2av(t), 1.0, **TOLS))

    def test_l2av_with_twos(self):
        t = np.ones(10) * 2
        self.assertTrue(np.isclose(l2av(t), 4.0, **TOLS))


if __name__ == '__main__':
    unittest.main()

