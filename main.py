from scipy.optimize import leastsq
import matplotlib.pyplot as plt

from optimalmap import *
from transformimage import *


def find_optimal_map(area_penalty, degree=6, quiet=False, nquadpts=None):
    if nquadpts is None:
        nquadpts = 12 * degree
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
    # area_penalties = [1, 30., 2e2]
    area_penalties = [30.]
    all_cost_evaluators = [find_optimal_map(p) for p in area_penalties]
    transformations = [c.transform for c in all_cost_evaluators]
    old_image = plt.imread('./lambert-cropped.png')
    for penalty, transform in zip(area_penalties, transformations):
        basename = './params-degree={}-penalty={}'.format(
            transform.degree[0], round(penalty))
        np.savetxt(basename + '.csv', transform.params, delimiter=',')
        new_image = transform_image(old_image, transform)
        plt.imsave(basename + '.png', new_image)


if __name__ == '__main__':
    main()

