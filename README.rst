Finding the minimal-distortion equiareal map of the world.

The big picture.
================

Making a map of the world is always about tradeoffs between which pieces of information to convey accurately and how much to prioritize aesthetics over accuracy. No map can do a perfect job at conveying everything. However, when I look at a map of the world, there are a few things I don't care about -- I'm not looking at a world map to chart a trajectory or to measure distances between two countries. Usually when I look at a map, I want the map to convey two things about the world and its landmasses:

*  their shapes.
*  their sizes.

Unfortunately, the mathematics of creating maps tells us that it's *impossible* to create a map that accurately represents both shapes and sizes. But that doesn't mean that all maps are equal. In this codebase, I'll try to find a balance between shape and size accuracy to create a "good" map of the world. To do this, we'll need to combine differential geometry, numerical analysis, and numerical optimization techniques. I'll try to give a high-level overview of how I'm doing this in the next few paragraphs, assuming some technical competence in math and numerical analysis. If you want to see more detail, look at the code or reach out to me!

Also, fair warning -- I'm not a cartographer, just a wandering physicist who likes differential geometry, computation, and world maps. So I'm more motivated by finding a good tradeoff between correct shapes and correct sizes than I am in highlighting features of the Earth's geography. Which means that this might not be the best map for you to print out and put on your wall ;)


The Differential Geometry
==========================

To talk about shapes and sizes in maps, we first need a mathematical language to describe maps. That language is differential geometry. Differential geometry is a full upper level mathematics course, which I am not going to pretend to cover in full here. But I will give a quick overview of what we need to know for making a map. (If you want to learn differential geometry, I'll enthusiastically plug `this book <https://www.amazon.com/Differential-Geometry-Dover-Books-Mathematics/dp/0486667219/>`_.)

At its most abstract, a map of the world is a function which takes 2 variables as input (*i.e.* the *x* and *y* coordinates on the sheet of paper) and outputs a vector in 3-dimensional space (*i.e.* the location of that point on the Earth's surface). Since there is some space for confusion on whether the word "map" is referring to the abstract function or the physical sheet of paper, I'll be explicit and say "map function" when I am referring to the abstract function and the "printed map" when I'm referring to the physical sheet of paper.

Our end goal is to choose a map function where shapes and sizes on the printed map are proportional to the shapes and sizes on the Earth's surface. To do this, we need a way to relate distances on the Earth's surface to those on the printed map. We start with a description of real, physical distances on the Earth. We can represent each point on the Earth's surface by a three-dimensional vector :math:`\vec{x}`, which we'll write a a function of two coordinates :math:`(u^1, u^2)`. (Note that I'm using a superscript here, not an exponent. I'll use parentheses when I want to denote an exponent.) If we write the coordinates in shorthand as :math:`(u^1, u^2) = u^\alpha`, we can denote each point on the Earth's surface compactly as :math:`\vec{x}(u^\alpha)`. You can think of the coordinates :math:`u^\alpha` as latitude and longitude or :math:`\theta` and :math:`\phi`, or -- more useful here -- as the :math:`x` and :math:`y` coordinates on the printed map.

Let's consider the distance on the Earth's surface between the point :math:`\vec{x}(u^\alpha)` and :math:`\vec{x}(u^\alpha + du^\alpha)`, where :math:`du^\alpha` is small in some sense. We can write this distance :math:`s` as:

 ..  math::

    (ds)^2 = \|\vec{x}(u^\alpha + du^\alpha) - \vec{x}(u^\alpha)\|^2

Taylor expanding :math:`\vec{x}(u^\alpha + du^\alpha)` gives

 ..  math::

    (ds)^2 = \frac {\partial \vec{x}} {\partial u^1} \, \cdot \, \frac {\partial \vec{x}} {\partial u^1} \, (du^1)^2 + 2 \frac {\partial \vec{x}} {\partial u^1} \, \cdot \, \frac {\partial \vec{x}} {\partial u^2} {du^1 du^2} + \frac {\partial \vec{x}} {\partial u_2} \frac {\partial \vec{x}} {\partial u_2} (du^2)^2

If we use the convention that repeated indices are summed over, we can write this compactly as

 ..  math::

    (ds)^2 = g_{\alpha \beta} du^\alpha du^\beta

where we have introduced the *metric*

 ..  math::

    g_{\alpha \beta} = \frac {\partial \vec{x}} {\partial u_\alpha} \, \cdot \, \frac {\partial \vec{x}} {\partial u_\beta}
    \;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;(1)

The metric :math:`g_{\alpha \beta}` in general changes from point to point. As you can see from its definition, at each point the metric can be represented as a symmetric matrix. If the map perfectly represented the world, then all the distances between all the points would be proportional, and the metric would be the identity:

 ..  math::

    \mathrm{Perfect,\,flat\,metric} = \left( \begin{array}{cc} 1 & 0 \\ 0 & 1 \end{array} \right)

The metric turns out to *the* most important thing in differential geometry. The metric determines not only distances, but also angles and areas of objects. For instance, if we draw a square on the printed map, with a lower-left corner at :math:`u^\alpha` and an upper right corner at :math:`u^\alpha + du^\alpha`, the area of the corresponding region on the Earth's surface is :math:`\sqrt{g} \, du^1 du^2`, where :math:`g` is the determinant of the metric. Likwise, if we draw two lines, each starting at :math:`u^\alpha` but one ending at :math:`u^\alpha + dv^\alpha` and the other ending at :math:`u^\alpha + dt^\alpha`, the angle between those two lines on the Earth's surface is :math:`\cos \, \theta = g_{\alpha \beta} dt^\alpha dv^\beta`.

What happens if we repeat this and make a map of the map? If we define two new coordinates, :math:`\tilde{u}^1(u^1, u^2)` and :math:`\tilde{u}^2(u^1, u^2)`, such that the functions :math:`\tilde{u}^1(u^1, u^2)` and :math:`\tilde{u}^2(u^1, u^2)` are invertible, what is the distance between the two points :math:`(\tilde{u}^1, \tilde{u}^2)` and :math:`(\tilde{u}^1 + d\tilde{u}^1, \tilde{u}^2 + d\tilde{u}^2)`?

Since the transformation :math:`(u^1, u^2) \rightarrow (\tilde{u}^1, \tilde{u}^2)` is invertible, the points :math:`(\tilde{u}^1, \tilde{u}^2)`  and :math:`(\tilde{u}^1 + d\tilde{u}^1, \tilde{u}^2 + d\tilde{u}^2)` each correspond to exactly one point on the original :math:`(u^1, u^2)` map. As such, we can find the distance between the corresponding points in terms of :math:`(u^1, u^2)`, which is

 ..  math::

    (ds)^2 = g_{\alpha \beta} du^\alpha du^\beta

Next, from the chain rule, we can write

 ..  math::

    du^\alpha = \frac{\partial u^\alpha} {\partial \tilde{u}^\gamma} \, d\tilde{u}^\gamma

Substituting this in we find the distance between the points :math:`(\tilde{u}^1, \tilde{u}^2)` and :math:`(\tilde{u}^1 + d\tilde{u}^1, \tilde{u}^2 + d\tilde{u}^2)` as:

 ..  math::

    (ds)^2 = g_{\alpha \beta} \frac{\partial u^\alpha} {\partial \tilde{u}^\gamma} \frac{\partial u^\beta} {\partial \tilde{u}^\delta} d\tilde{u}^\gamma d\tilde{u}^\delta

which looks suspiciously like equation (1), but with a transformed metric


 ..  math::

    \tilde{g}_{\gamma \delta} = g_{\alpha \beta} \frac{\partial u^\alpha} {\partial \tilde{u}^\gamma} \frac{\partial u^\beta} {\partial \tilde{u}^\delta}
    \;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;(2)

In other words, once we calculate the metric in one coordinate system, we can relate it to the metric in any other coordinate system by matrix multiplication with the derivatives -- we don't need to recalculate the metric from scratch! In fact, the most poweful way to think about a metric is just as an object that transforms this way under coordinate transformations. In differential geometry there are a whole slough of objects that transform this way; these objects are known as *covariant tensors*. But that knowledge isn't necessary for what we'll do here.

Setting up the Problem
======================

Now, at this stage, it should be obvious how to create the perfect map: Find a coordinate transformation such that the metric :math:`g_{\alpha \beta}` is the identity. Unfortunately, a `classic theorem <https://en.wikipedia.org/wiki/Theorema_Egregium>`_ of differential geometry says that it is not possible to map the surface of a sphere to a flat plane without deformations. The reason has to do with the fact that the `Gaussian curvature <https://en.wikipedia.org/wiki/Gaussian_curvature>`_ depends only on the metric. If it were possible to map the sphere to the plane without distortion, then the metric on the map would be equal to the metric on the sphere, which would imply that the Gaussian curvature of the sphere is equal to that of the map. But the Gaussian curvature of a plane is 0 and the Gaussian curvature of a sphere is positive. So it's not posible to map a sphere to a plane without distortion.

However, it *is* possible to create a conformal map of a sphere onto a plane, which preserves the angles between any two lines -- the classic `Mercator projection <https://en.wikipedia.org/wiki/Mercator_projection>`_ is an example of a conformal map. And it *is* possible to create an equiareal map of a sphere onto a plane, preserving the area of any shape -- some examples of equiareal projections are the `Lambert cylindrical <https://en.wikipedia.org/wiki/Lambert_cylindrical_equal-area_projection>`_ and the `Mollweide <https://en.wikipedia.org/wiki/Mollweide_projection>`_. It's just *not* possible to map a sphere onto a plane with a projection that is both conformal and equiareal. More importantly, though, there are *many* different conformal map projections, and *many* different conformal map projections.


What I will do here is look for the "best" equiareal map projection. Specifically, I will look for the equiareal map projection that minimizes the distortion, as measured by some cost function. We'll find the best map projection numerically. First, I'll need to parameterize the space of possible maps. Second, I'll want a cost function that allows for fast numerical computation and optimization. Third, once I have a parameterization of the map function and a cost function for the map function, I can search through that parameter space to find the best map function. I'll choose the cost function and the parameterization of the maps with efficient computation in mind.

First, I need to create and parameterize the map function. I'll do this by transforming the coordinates from a normal map -- I'll start with the coordinates from the Lambert cylindrical projection :math:`(L_x, L_y)`, since it's already equiareal. Then, I'll define the map coordinates :math:`(x, y)` through two polynomials

 ..  math::

    x = \sum_{mn} \theta_{xmn} \times (L_x)^m (L_y)^n
    \;\;\;\;\;\;\;\;\;\;\;\;\;\;\; (3a)
    \\
    y = \sum_{mn} \theta_{ymn} \times (L_x)^m (L_y)^n
    \;\;\;\;\;\;\;\;\;\;\;\;\;\;\; (3b)

where the coefficients :math:`\theta = \{\theta_{xmn}, \theta_{ymn} \}` of the polynomials parameterize the map function. Using a polynomial allows me to quickly evaluate derivatives :math:`dx/dL_x`, which I can use with equation (2) to rapidly evaluate the metric in the new coordinates.

Next, I need a cost function. To penalize deviations from non-conformality, I'll take the sum of the squares of the difference between the metric and a flat metric:

 ..  math::

    \int \, dx\, dy \, \sum_{\alpha, \beta} \left( g_{\alpha \beta} - \delta_{\alpha \beta} \right)^2

I also need to constrain the map to be equiareal, *i.e.* constraing the determinant :math:`g=1`. I'll do this using a `Lagrange multiplier <https://en.wikipedia.org/wiki/Lagrange_multiplier>`_ :math:`\lambda` times another sum of squares, to give the total cost function as:

 ..  math::

    C(\theta) = \int \, \Big[ \sum_{\alpha, \beta} \left( g_{\alpha \beta} - \delta_{\alpha \beta} \right)^2 + \lambda (g - 1)^2 \Big] \, dx\, dy


When optimizing the map function's parameters, I'll need to evaluate this cost function many times, so I want to evaluate this integral as efficiently as possible. I'll do so by using `Gauss-Legendre quadrature <https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss%E2%80%93Legendre_quadrature>`_ over each of the variables x and y, to give

 ..  math::

    C(\theta) = \sum_{ij} w_{i} w_j \Bigg\{\sum_{\alpha, \beta} \Big[g_{\alpha, \beta}(x_i, y_j; \, \theta) - \delta_{\alpha \beta}\Big]^2 + \lambda \Big[ g(x_i, y_j; \, \theta) - 1\Big] ^2 \Bigg\}
    \;\;\;\;\;\;\;\;\;\;\;\;\;\;\; (4)

where the points :math:`(x_i, y_i)` and weights :math:`(w_i, w_j)` are definted by the Gaussian quadrature rules.

Solving the Problem
===================

At this point I have a parameterization of the map function, an easy way to calculate the metric, and a cost function which is minimized when the map function has minimal distortion in some sense. Now we just need to find the polynomial coefficients :math:`\theta` that minimize the cost function.

To efficiently minimize the cost function, I'll use the `Levenberg-Marquardt algorithm <https://en.wikipedia.org/wiki/Levenberg-Marquardt_algorithm>`_. The Levenberg-Marquardt algorithm is a very fast way to find local optima of a cost function that can be written as a sum of squares. Briefly, the Levenberg-Marquardt algorithm uses the structure of the cost function and first derivative information to approximate the cost function's second derivatives. As such, it converges very fast when the initial guess is near the optimal value. Since my optimal map problem is fairly simple, I'll just use Levenberg-Marquardt as implemented in `scipy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares>`_.

Finally, for aesthetic and numerical reasons I'll constrain the space of possible map functions a bit. Parameterizing the map function by a polynomial is a little redundant, as it allows for overall translations and rotations which don't change the shape or size of objects on the printed map. To eliminate overall translations, I'll set :math:`\theta_{x00}=0` and :math:`\theta_{y00}=0` in equations 3a-b. As an imperfect way to eliminate overall rotations, I'll also set :math:`\theta_{y01}=0` and :math:`\theta_{y10}=0`. This isn't perfect though, as the cubic and higher order terms end up looking like a rotation. So to stop the higher-order terms from rotating the map, I've also set :math:`\theta_{xkl}=0` when *l* is odd and :math:`\theta_{ykl}=0` when *k* is odd.  (A better solution would be to use orthogonal polynomials like `Legendre <https://en.wikipedia.org/wiki/Legendre_polynomials>`_ or `Chebyshev <https://en.wikipedia.org/wiki/Chebyshev_polynomials>`_ polynomials as a basis, but this makes calculating the derivatives needed for equation 2 just a little more complicated.)

You can see all of this in code in the ``optimalmap.py`` file. The ``CoordinateTransform`` class is responsible for transforming coordinates given a set of parameters, the ``MetricCostEvaluator`` class is responsible for evaluating a cost for a given set of parameters and a ``CoordinateTransform`` (both directly and by returning a residuals vector to be squared and summed for the cost). Finally, there are two helper classes, ``LambertCylindricalQuadrature`` and ``LambertProjection`` for calculating the integrals over the map's domain. Once initialized, the best parameters are just found via a ``scipy.optimize.leastsq`` call, as shown in ``main.py``.


The Results
============

So, what does it look like? After a call to ``scipy.optimize.leastsq``, in a minute or so on my machine I get a set of parameters which describe the map function which minimizes shape and size distortion. I then need to render the map. Normally, one would query a pixel (i, j) in the image to render and ask what color that should be. However, that requires knowing the inverse map from pixel (i, j) back to the world coordinates. But we don't have the inverse map, we only have the forward map. To avoid computing the inverse, what I do instead is calculate the forward map and interpolate onto pixels. This interpolation-based code is in the ``transformimage.ImageTransformer`` class. (A more elegant way would be to compute the inverse as a polynomial approximant, which can be done pretty quickly. But it's a little more work on the surface.)

Doing all this for a map function parameterized by two 12 :math:`\times` 12 degree polynomials (180 total parameters after constraining a few to zero) gives this map:

  .. image:: params-degree=12-penalty=30.jpg
     :scale: 50 %
     :align: center

I'll let the result speak for itself.


Next Steps?
===========
One obvious problem with the picture above is that the poles are still mapped to a line, still giving some shape distortion at the poles. In fact, the shape distortion in this map is still infinite at the poles! This problem happens because I started from the Lambert projection (in equation 3), and the Lambert projection has infinite shape distortion at the poles (although the area at the poles is still correct in the Lambert projection). This singularity doesn't affect my numerical approach here too much, because the quadrature nodes I used in equation 4 are never exactly at the poles, so numerically the distortion is always finite. A solution to this problem would be to start from a map function that has no singularities in the metric, such as the `Mollweide <https://en.wikipedia.org/wiki/Mollweide_projection>`_ or the `Sanson <https://en.wikipedia.org/wiki/Sinusoidal_projection>`_ projection, rather than using the Lambert projection with its singularities. Perhaps I'll do this later, perhaps not.

