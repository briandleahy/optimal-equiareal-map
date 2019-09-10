import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

from optimalmap import MetricCostEvaluator
from transformimage import transform_image


def find_optimal_map(area_penalty, degree=12, quiet=False, nquadpts=None):
    if nquadpts is None:
        nquadpts = 6 * degree
    cost_evaluator = MetricCostEvaluator(
        degree=(degree, degree), area_penalty=area_penalty, nquadpts=nquadpts)
    # Then we change the transformation to start from Gall-Peters:
    # (otherwise it gets stuck)
    cost_evaluator.transform._coeffs[0, 1, 0] = 0.691312
    cost_evaluator.transform._coeffs[1, 0, 1] = 1.445993
    # done changing to Gall-Peters:
    initial_guess = cost_evaluator.params
    if not quiet:
        print('Area penalty={}'.format(area_penalty))
        print('\tInitial Cost: \t{:.6f}'.format(
            np.sum(cost_evaluator.call(initial_guess)**2)))
    fit_result = leastsq(cost_evaluator.call, initial_guess)
    cost_evaluator.update(fit_result[0])
    dg, da = cost_evaluator.calculate_metric_residuals()
    if not quiet:
        print('\tFinal Cost:\t{:.6f}'.format(
            np.sum(cost_evaluator.call(fit_result[0])**2)))
        print('\tEquiareal to \t{:.6f}'.format(np.sum(da**2)))
        print('\t``Flat`` to \t{:.6f}'.format(np.sum(dg**2)))
    return cost_evaluator


def main():
    area_penalty = 30.0
    cost_evaluator = find_optimal_map(area_penalty)
    transform = cost_evaluator.transform
    old_image = plt.imread('./lambert-cropped.png')
    basename = './params-degree={}-penalty={}'.format(
        transform.degree[0], round(area_penalty))
    np.savetxt(basename + '.csv', transform.params, delimiter=',')
    new_image = transform_image(old_image, transform)
    plt.imsave(basename + '.jpg', new_image)
    return new_image, transform


if __name__ == '__main__':
    main()
    # degree = 12: 1 min 24 s. Increasing the degree to 20 doesn't make
    # any significant differences except for longer run time.

