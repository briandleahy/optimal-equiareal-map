from scipy.optimize import leastsq
import matplotlib.pyplot as plt

from optimalmap import *
from transformimage import *


DEGREE = 6


def get_maps(area_penalties, quiet=False):
    all_cost_evaluators = []
    for area_penalty in area_penalties:
        cost_evaluator = MetricCostEvaluator(
            degree=(DEGREE, DEGREE), area_penalty=area_penalty,
            nquadpts=12 * DEGREE)
        cost_evaluator.update_area_penalty(area_penalty)
        p0 = (
            cost_evaluator.params if len(all_cost_evaluators) == 0
            else all_cost_evaluators[-1].params)
        ans = leastsq(cost_evaluator.call, p0)
        cost_evaluator.update(ans[0])
        dg, da = cost_evaluator.calculate_metric_residuals()
        if not quiet:
            print('Area penalty={}'.format(area_penalty))
            print('\tEquiareal to \t{:.6f}'.format(l2av(da)))
            print('\t``Flat`` to \t{:.6f}'.format(l2av(dg)))
        all_cost_evaluators.append(cost_evaluator)
    return all_cost_evaluators


def main():
    area_penalties = [1, 30., 2e2]
    all_cost_evaluators = get_maps(area_penalties)
    transformations = [c.transform for c in all_cost_evaluators]
    old_image = plt.imread('./lambert-cropped.png')
    for penalty, transform in zip(area_penalties, transformations):
        basename = './params-degree={}-penalty={}'.format(
            DEGREE, round(penalty))
        np.savetxt(basename + '.csv', transform.params, delimiter=',')
        new_image = transform_image(old_image, transform)
        plt.imsave(basename + '.png', new_image)


if __name__ == '__main__':
    main()

