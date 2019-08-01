"""
TODO :
Figure out why it's cropped at the poles

Actually just redo this. Transform the pixels from the old image to the new
image, update the colors, then use that to create an interpolator.
"""
import itertools
import numpy as np
from scipy.interpolate import griddata


class ImageTransformer(object):
    def __init__(self, image, transformation):
        self.image = image
        self.transformation = transformation

    def transform_image(self):
        transformed_points = self._transform_pixel_locations()
        new_image_shape = self._get_bounding_box_size_for(transformed_points)
        new_image = np.zeros(new_image_shape + (3,))

        # 1. Assign the points at
        tx, ty = np.round(transformed_points).astype('int')
        for i in range(3):
            new_image[tx, ty, i] = self.image[..., i]
        not_filled_in = np.ones(new_image_shape, dtype='bool')
        not_filled_in[tx, ty] = False

        # 2. Do a quick grey closing of each channel:
        for i in range(3):
            new_image[..., i] = grey_closing(new_image[..., i], size=3)

        # 3. Finally re-assign the values that we know:
        for i in range(3):
            new_image[tx, ty, i] = self.image[..., i]
        return new_image

    def _transform_pixel_locations(self):
        # +- pi, +- 1 are hard-coded for the Lambert projection
        xold = np.linspace(-np.pi, np.pi, self.image.shape[1]).reshape(1, -1)
        yold = np.linspace(-1,     1,     self.image.shape[0]).reshape(-1, 1)
        transformed_x, transformed_y = self.transformation.evaluate(xold, yold)

        xscale = self.image.shape[1] / xold.ptp()
        yscale = self.image.shape[0] / yold.ptp()

        coordinates_and_scales = (
            [transformed_x, xscale],
            [transformed_y, yscale],
            )
        for coord, scale in coordinates_and_scales:
            coord -= coord.min()
            coord *= scale
        return transformed_x, transformed_y

    @classmethod
    def _get_bounding_box_size_for(cls, transformed_points):
        bbox_shape = [t.ptp() for t in transformed_points]
        bbox_int = np.ceil(bbox_shape).astype('int')
        return tuple(bbox_int)


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
    xnew, ynew = transform.evaluate(xyold[:, 0], xyold[:, 1])
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
    # get the new aspect ratio, image size
    new_shp = np.ceil(transformed_points.ptp(axis=0)[::-1])
    new_im = np.zeros(new_shp.astype('int').tolist() + [3])
    # Now get the x, y values for the new image:
    x0, y0 = [np.arange(t.min(), t.max() + 1, 1) for t in transformed_points.T]
    xynew = np.array(
        [[x, y] for x, y in itertools.product(x0, y0)],
        dtype='int')
    xynewim = xynew - xynew.min(axis=0)
    # raise ValueError
    # Pack each set of r, g, b value into the new image, using griddata
    # and the new coordinates of the image.
    r, g, b = [old_im[:, :, i].T.ravel() for i in range(3)]
    for i, c in enumerate([r, g, b]):
        new_im[xynewim[:, 1], xynewim[:, 0], i] = griddata(
            transformed_points, c, xynew, method='linear', fill_value=0)
    return new_im

