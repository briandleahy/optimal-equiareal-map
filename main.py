import argparse

import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

import optimalmap
from transformimage import transform_image


def main(mode='sanson', degree=8, nquadpts=None, area_penalty=30.0,
         quiet=False):
    if nquadpts is None:
        nquadpts = 8 * degree + 1
    cost_evaluator = find_optimal_map(
        area_penalty,
        mode=mode,
        degree=degree,
        nquadpts=nquadpts,
        quiet=quiet)
    if mode == 'sanson':
        transform = optimalmap.ChainTransform([
            optimalmap.LambertToSansonTransform(),
            cost_evaluator.transform])
    elif mode == 'lambert':
        transform = cost_evaluator.transform
    else:
        raise ValueError("Invalid mode")
    old_image = plt.imread('lambert-cropped.png')
    new_image = transform_image(old_image, transform)

    basename = './{}-degree={}-penalty={}-nquadpts={}'.format(
        mode,
        degree,
        round(area_penalty),
        nquadpts)
    np.savetxt(
        basename + '.csv',
        cost_evaluator.transform.params,
        delimiter=',')
    plt.imsave(basename + '.jpg', new_image)
    return new_image, transform


def find_optimal_map(area_penalty, degree=8, nquadpts=85, quiet=False,
                     mode='sanson'):
    cost_evaluator = optimalmap.MetricCostEvaluator(
        projection_name=mode,
        degree=(degree, degree),
        area_penalty=area_penalty,
        nquadpts=nquadpts)
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


parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    help=("{sanson, lambert}. The base map projection to start from. "
          "Currently only `sanson` and `lambert` are supported. Default is"
          " sanson"),
    type=str,
    default="sanson")
parser.add_argument(
    "--degree",
    help="int. The polynomial degree of the transform. Default is 8",
    type=int,
    default=8)
parser.add_argument(
    "--nquadpts",
    help=("int. The number of quadrature points for evaluating the cost"
          "function. Higher values mean more robustness, at the cost of"
          " more computation time (quadratinc in `nquadpts`). Default is 8 * "
          "`degree` + 1."),
    type=int,
    default=None)
parser.add_argument(
    "--area_penalty",
    help=("float. The Lagrange multiplier enforcing the penalty for"
          " departures from equiareal maps. Higher penalties means maps"
          " closer to equiareal. Default is 30.0, which produces a map"
          " that is visually indistinguishable from one with higher"
          " penalties."),
    type=float,
    default=30.0)
parser.add_argument(
    "--quiet",
    help=("bool. Set to True to avoid printing information about the"
          " fits. Default is False."),
    type=bool,
    default=False)

args = parser.parse_args()

if __name__ == '__main__':
    main(**vars(args))

