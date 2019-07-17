import itertools
import numpy as np
from scipy.interpolate import griddata


def px1_mapto_px2(img, transform, xlims, ylims, center=True):
    """
    Calculates a transformation from one image to another

    img : the image to transform to
    transform : the transofrmation
    px_dist_ratio : functions which take image coordinates to the
        transformation's coordinates

    This won't work since the function needs to know about the image
    size....... you need a function that gets the transformed image
    coordinates.
    """
    yold, xold = [np.linspace(*lims, s) for lims, s in zip(
            [ylims, xlims], img.shape[:2])]
    if center:
        xold -= xold.mean()
        yold -= yold.mean()
    xyold = np.array([[x, y] for x, y in itertools.product(xold, yold)])
    xnew, ynew = transform.eval(xyold[:, 0], xyold[:, 1])
    xynew = np.zeros([xnew.size, 2])
    xscale = img.shape[1] / np.diff(xlims)
    yscale = img.shape[0] / np.diff(ylims)
    xynew[:, 0] = xnew * xscale
    xynew[:, 1] = ynew * yscale
    return xynew


def transform_image(old_im, transform):
    # get the colors, old points:
    xlims = (-np.pi, np.pi)
    ylims = (-1., 1.)
    transformed_points = px1_mapto_px2(
        old_im, transform, xlims, ylims, center=True)
    r, g, b = [old_im[:, :, i].T.ravel() for i in range(3)]
    # get the new aspect ratio, image size
    aspect_ratio = (
        transformed_points[:, 0].ptp() / transformed_points[:, 1].ptp())
    new_shp = np.ceil(transformed_points.ptp(axis=0)[::-1])
    new_im = np.zeros(new_shp.astype('int').tolist() + [3])
    # Now get the x, y values for the new image:
    x0, y0 = [np.arange(t.min(), t.max() + 1, 1) for t in transformed_points.T]
    xynew = np.array(
        [[x, y] for x, y in itertools.product(x0, y0)],
        dtype='int')
    xynewim = xynew - xynew.min(axis=0)
    # raise ValueError
    for i, c in enumerate([r, g, b]):
        new_im[xynewim[:, 1], xynewim[:, 0], i] = griddata(
            transformed_points, c, xynew, method='linear', fill_value=0)
    return new_im


