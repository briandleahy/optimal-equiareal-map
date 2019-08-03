from scipy.optimize import leastsq
import matplotlib.pyplot as plt

from optimalmap import *
from transformimage import *


def get_maps(area_penalties, quiet=False):
    all_cost_evaluators = []
    for area_penalty in area_penalties:
        cost_evaluator = MetricCostEvaluator(
            degree=(6, 6), area_penalty=area_penalty, nquadpts=75)
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


def calculate_images_for(area_penalties):
    all_cost_evaluators = get_maps(area_penalties)

    old_im = plt.imread('./lambert-cropped.png')
    # old_im = plt.imread('./lambert-cropped-small.png')
    new_ims = [transform_image(old_im, cost_evaluator.transform)
               for cost_evaluator in all_cost_evaluators]
    return new_ims


def main():
    area_penalties = [0.5, 30., 2e2]
    new_images = calculate_images_for(area_penalties)
    for ap, ni in zip(area_penalties, new_images):
        plt.imsave('./optimal-equiareal-area_penalty={}.png'.format(ap), ni)


if __name__ == '__main__':
    main()

