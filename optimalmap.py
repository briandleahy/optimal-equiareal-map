import itertools

import numpy as np


polyval2d = np.polynomial.polynomial.polyval2d  # shorter alias


class CoordinateTransform(object):
    def __init__(self, degree=(5, 5)):
        """
        Represents a coordinate transformation as a polynomial.

        Since we don't care about overall shifts (x -> x + a) we start
        at linear order in the polynomial by keeping the zeroth term
        zero.
        """
        self.degree = degree
        shp = [2] + (np.array(degree)+1).tolist()
        self._coeffs = np.zeros(shp, dtype='float')  # 2 for x,y
        self._mask = np.ones(shp, dtype='bool')
        self._mask[0, 0] = False
        # Then we want to start at a reasonable param value (X=x etc)
        self._coeffs[0, 1, 0] = 1
        self._coeffs[1, 0, 1] = 1  # FIXME check!!!

    def update(self, params):
        self._coeffs[self._mask] = params.copy()

    @property
    def params(self):
        return self._coeffs[self._mask].copy()

    def evaluate(self, x, y):
        """New coordinates as a function of the old"""
        ans = [polyval2d(x, y, c) for c in self._coeffs]
        return ans

    def evaluate_derivative(self, x_new='X', x_old='x'):
        """returns the matrix that, when polyval'd, gives dX/dx where
        X is the new coordinate and x the old

        Parameters
        ----------
        x_new : {"X", "Y"}
        x_old : {"x", "y"}
        """
        coef_ind = {'X': 0, 'Y': 1}[x_new.upper()]
        aij_ind = {'x': 0, 'y': 1}[x_old.lower()]
        shp = {'x': (-1, 1), 'y': (1, -1)}[x_old.lower()]
        aij = self._coeffs[coef_ind]
        t = np.arange(aij.shape[aij_ind]).reshape(shp)
        return np.roll(aij * t, -1, axis=aij_ind)


class LambertCylindricalQuadrature(object):
    def __init__(self, nxpts=30):
        self.nxpts = nxpts
        self.nypts = int(nxpts / np.pi) + 1
        self._setup_pts()

    def _setup_pts(self):
        # x runs from -pi, pi; y from -1, 1. So we need to adjust px, wx:
        px, wx = np.polynomial.legendre.leggauss(self.nxpts)
        px *= np.pi
        wx *= np.pi
        py, wy = np.polynomial.legendre.leggauss(self.nypts)
        xp, yp = np.meshgrid(px, py, indexing='ij')
        self._xypts = np.array([[x, y] for x, y in zip(xp.flat, yp.flat)])
        self._xywts = np.outer(wx, wy).ravel()
        self._xywts_sqrt = np.sqrt(self._xywts)

    def integrate(self, func):
        fxy = func(self._xypts)
        return np.sum(fxy * self._xywts)

    def integrate_as_sumofsquares(self, func):
        """Returns the integral as a set of pts such that \int f(x)^2 =
        sum (ans)^2 where ans is the output of this function"""
        fxy = func(self._xypts)
        return np.ravel(fxy * self._xywts_sqrt)

    @property
    def pts(self):
        return self._xypts.copy()


class LambertProjection(object):
    def __init__(self, xypts):
        """xypts = [N, 2] = (phi, theta)"""
        self.metric = np.zeros([xypts.shape[0], 2, 2])
        sin2theta = 1 - xypts[:, 1]**2
        self.metric[:, 0, 0] = 1.0 / sin2theta
        self.metric[:, 1, 1] = sin2theta
        # -- we don't need to regularize sin(2theta) with a +eps b/c
        # the legendre points aren't selected at y=+-1


class FittingWrapper(object):
    def __init__(self, nquadpts=30, degree=(5, 5), area_penalty=1.):
        # 1. the quadrature points
        self.quadobj = LambertCylindricalQuadrature(nxpts=nquadpts)
        # 2. the original metric
        self.g0 = LambertProjection(self.quadobj.pts)
        # 3. the new function
        self.transform = CoordinateTransform(degree=degree)
        self.area_penalty = area_penalty

    def calculate_metric_residuals(self):
        newmetric = self._calculate_metric()
        # 4. The deviation from perfection:
        deviation_from_isometry = newmetric - np.eye(2).reshape(1, 2, 2)
        # 5. The equiareal penalties
        deviation_from_equiareal = np.linalg.det(newmetric) - 1.
        return deviation_from_isometry.ravel(), deviation_from_equiareal

    def _calculate_metric(self):
        # 1. The old metric
        oldmetric = self.g0.metric
        # 2. The transformation matrix dX/dx
        xy = self.quadobj.pts
        dXdx = np.zeros([xy.shape[0], 2, 2], dtype='float')
        for a, b in itertools.product([0, 1], [0, 1]):
            aij = self.transform.evaluate_derivative(
                x_new='XY'[a], x_old='xy'[b])
            dXdx[:, a, b] = polyval2d(xy[:, 0], xy[:, 1], aij)
        # 3. The new metric
        newmetric = np.einsum('...ij,...ik,...jl', oldmetric, dXdx, dXdx)
        return newmetric

    def update(self, params):
        self.transform.update(params)

    def call(self, params):
        # 1. update params:
        self.update(params)
        # 2. metric, residuals
        dg, da = self.calculate_metric_residuals()
        return np.hstack([dg, self.area_penalty * da])

    def update_area_penalty(self, new_penalty):
        self.area_penalty = new_penalty

    @property
    def params(self):
        return self.transform.params


def l2av(x):
    return np.mean(x*x)

