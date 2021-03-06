"""
An alternative way to do this is to find the inverse of the transformation
directly. For the 1D case, we can do it as follows.

We write the inverse function as a sum of basis functions:

 .. math ::

    f^{-1}(xz) = \\sum_k a_k \\phi_k(x)

Then we know that

  .. math ::

    f^{-1}(f(x)) = x
    \\sum_k a_k \\phi_k(f(x)) - x = 0

Which means that we can write this as a linear least squares problem:
  .. math ::

    \\sum_i (\\sum_k a_k \\phi_k(f(x_i)) - x_i)^2 = 0
which is a linear least-squares problem for a_k.

I'm a little worried about points which lack an inverse though (i.e. points
whose inverse is outside the domain of the map), so I'll do it later.
"""
import itertools

import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import binary_closing
from skimage.morphology import remove_small_holes, disk


def transform_image(old_im, transform):
    transformer = ImageTransformer(old_im, transform)
    return transformer.transform_image()


class ImageTransformer(object):
    def __init__(self, image, transformation):
        if image.shape[1] < 2.9 * image.shape[0]:
            msg = """This assumes a Lambert projection, with the
                     x-direction along image.shape[1]. It seems that
                     the image does not meet this criterion. Try
                     transposing the image."""
            raise ValueError(msg)
        self.image = image
        self.transformation = transformation
        # Hard-coded for lambert projection:
        self.range_x = (-np.pi, np.pi)
        self.range_y = (-1, 1)

    def transform_image(self):
        transformed_points = self._transform_pixel_locations()
        new_image_shape = self._get_bounding_box_size_for(transformed_points)
        new_image = np.zeros(new_image_shape + (3,))

        # 1. Assign the points at
        t0, t1 = np.round(transformed_points).astype('int')
        for i in range(3):
            new_image[t0, t1, i] = self.image[..., i]
        filled_in = np.zeros(new_image_shape, dtype='bool')
        filled_in[t0, t1] = True

        new_image = self._fill_in_holes(new_image, filled_in)
        new_image = self._zero_out_edges(new_image, filled_in)
        return new_image

    def _transform_pixel_locations(self):
        ny, nx = self.image.shape[:2]
        xflat = np.linspace(*self.range_x, nx + 1)[:-1]
        yflat = np.linspace(*self.range_y, ny + 1)[:-1]
        yold, xold = np.meshgrid(yflat, xflat, indexing='ij')
        transformed_x, transformed_y = self.transformation.evaluate(xold, yold)

        yscale = self.image.shape[0] / np.ptp(self.range_y)
        xscale = self.image.shape[1] / np.ptp(self.range_x)

        coordinates_and_scales = (
            [transformed_x, xscale],
            [transformed_y, yscale],
            )
        for coord, scale in coordinates_and_scales:
            coord -= coord.min()
            coord *= scale
        return transformed_y, transformed_x

    @classmethod
    def _fill_in_holes(cls, new_image, filled_in):
        # We do a Barnes-like interpolant, as follows:
        # 1. Make a kernel
        kernel_width = max(7, int(0.03 * max(new_image.shape)))
        t_kernel = np.linspace(-7, 7, kernel_width)
        x_kernel = t_kernel.reshape(-1, 1)
        y_kernel = t_kernel.reshape(1, -1)
        kernel = np.exp(-0.5*(x_kernel**2 + y_kernel**2))
        kernel /= kernel.sum()
        # 2. get a smoothed version of the image:
        smoothed = new_image.copy()
        for i in range(3):
            smoothed[..., i] = fftconvolve(
                new_image[..., i], kernel, mode='same')
        # 3. Get a smoothed version of the mask:
        mask_smoothed = fftconvolve(
            filled_in.astype('float'), kernel, mode='same')
        # 4. re-scale the smoothed image by the smoothed mask,
        #    so we don't have dips in the brightness at the gaps:
        for i in range(3):
            smoothed[..., i] /= (mask_smoothed + 1e-10)
        # 4. Finally re-assign the values that we know:
        for i in range(3):
            smoothed[filled_in, i] = new_image[filled_in, i]
        return smoothed

    @classmethod
    def _zero_out_edges(cls, new_image, filled_in):
        # 1. Pad out ``filled_in`` so we dont remove any edges, which
        #    aren't holes. It also means we can get a robust shape for
        #    internal holes:
        pad = max(1, int(0.02 * max(new_image.shape)) + 1)
        padded = np.pad(filled_in, pad, 'constant', constant_values=False)
        hole_cutoff_size = 2 * np.sum(filled_in.shape)  # the perimeter
        # 2, Removing holes in mask and image:
        center = remove_small_holes(padded, hole_cutoff_size)
        # 3. Remove any stripes, which arise from stretching the image
        #    in 1D and skipping an entire row of pixels:
        circle = disk(pad)
        edges = ~binary_closing(center, circle)
        # 4. Unpad and zero out image
        zero_these_out = edges[pad:-pad, pad:-pad]  # unpadding
        new_image[zero_these_out] = 0.
        return new_image

    @classmethod
    def _get_bounding_box_size_for(cls, transformed_points):
        bbox_shape = [t.ptp() + 1 for t in transformed_points]
        bbox_int = np.ceil(bbox_shape).astype('int')
        return tuple(bbox_int)

