from scipy.optimize import leastsq
import matplotlib.pyplot as plt

from optimalmap import *
from transformimage import *


def get_maps(area_penalties):
    all_fws = []
    for ap in area_penalties:
        fw = MetricCostEvaluator(degree=(6, 6), area_penalty=ap, nquadpts=75)
        print('Area penalty={}'.format(ap))
        fw.update_area_penalty(ap)
        p0 = fw.params if len(all_fws) == 0 else all_fws[-1].params
        ans = leastsq(fw.call, p0)
        fw.update(ans[0])
        dg, da = fw.calculate_metric_residuals()
        print('Equiareal to \t{:.6f}'.format(l2av(da)))
        print('``Flat`` to \t{:.6f}'.format(l2av(dg)))
        all_fws.append(fw)
    return all_fws


def main():
    area_penalties = [0.5, 30., 2e2]
    all_fws = get_maps(area_penalties)
    # Now we transform:
    old_im = plt.imread('./lambert-cropped.png')
    # old_im = plt.imread('./lambert-cropped-small.png')
    new_ims = [transform_image(old_im, fw.transform) for fw in all_fws]
    for ap, ni in zip(area_penalties, new_ims):
        plt.imsave('./optimal-equiareal-area_penalty={}.png'.format(ap), ni)

if __name__ == '__main__':
    main()

