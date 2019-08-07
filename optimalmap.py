# FIXME
# The problem here is that the metric diverges like 1/sin(2theta) at the
# poles (so a 1/x singularity), which is _not integrable_. So while
# you don't get infinities, you do get something which diverges as you
# add more points. This is why you get weirdness with the maps as the degree
# goes higher.
# The solution is simple: optimize the metric on the punctured sphere,
# with a certain range away from the poles. Doing something like +- 5 deg
# latitude gets you out of the regions of interest everywhere (within
# antarctic mainland and within the arctic ocean.) -- northernmost land
# is 83d40m northn.
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
        self._mask = self._create_mask_for_parameters()
        # Then we want to start at a reasonable param value (X=x etc)
        self._coeffs[0, 1, 0] = 1
        self._coeffs[1, 0, 1] = 1

    def update(self, params):
        self._coeffs[self._mask] = params.copy()

    @property
    def params(self):
        return self._coeffs[self._mask].copy()

    def evaluate(self, x, y):
        """New coordinates as a function of the old"""
        return [polyval2d(x, y, c) for c in self._coeffs]

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

    def _create_mask_for_parameters(self):
        # We mask out the DC terms and rotations:
        mask = np.ones(self._coeffs.shape, dtype='bool')
        mask[:, 0, 0] = False
        mask[0, 0, 1] = False
        mask[1, 1, 0] = False
        return mask


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


class MetricCostEvaluator(object):
    def __init__(self, nquadpts=30, degree=(5, 5), area_penalty=1.):
        self.quadobj = LambertCylindricalQuadrature(nxpts=nquadpts)
        self.lambert_projection = LambertProjection(self.quadobj.pts)
        self.transform = CoordinateTransform(degree=degree)
        self.area_penalty = area_penalty

    def call(self, params):
        """Returns a vector whose sum-of-squares is a cost"""
        self.update(params)
        dg, da = self.calculate_metric_residuals()
        return np.hstack([dg, self.area_penalty * da])

    def update_area_penalty(self, new_penalty):
        self.area_penalty = new_penalty

    @property
    def params(self):
        return self.transform.params

    def update(self, params):
        self.transform.update(params)

    def calculate_metric_residuals(self):
        new_metric = self._calculate_metric()
        deviation_from_isometry = new_metric - np.eye(2).reshape(1, 2, 2)
        deviation_from_equiareal = np.linalg.det(new_metric) - 1.
        deviation_from_isometry *= self.quadobj._xywts_sqrt.reshape(-1, 1, 1)
        deviation_from_equiareal *= self.quadobj._xywts_sqrt
        return deviation_from_isometry.ravel(), deviation_from_equiareal

    def _calculate_metric(self):
        old_metric = self.lambert_projection.metric
        # 2. The transformation matrix dX/dx
        xy = self.quadobj.pts
        dXdx = np.zeros([xy.shape[0], 2, 2], dtype='float')
        for a, b in itertools.product([0, 1], [0, 1]):
            aij = self.transform.evaluate_derivative(
                x_new='XY'[a], x_old='xy'[b])
            dXdx[:, a, b] = polyval2d(xy[:, 0], xy[:, 1], aij)
        # 3. The new metric
        new_metric = np.einsum('...ij,...ik,...jl', old_metric, dXdx, dXdx)
        return new_metric



def l2av(x):
    return np.mean(x*x)

