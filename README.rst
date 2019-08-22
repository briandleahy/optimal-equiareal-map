Finding the minimal-distortion equiareal map of the world.

The big picture.
================

Making a map of the world is always about tradeoffs between which pieces of information to convey accurately and how much to prioritize aesthetics over accuracy. No map can do a perfect job at conveying everything. However, when I look at a map of the world, there are a few things I don't care about -- I'm not looking at a world map to chart a trajectory or to measure distances between two countries. Usually when I look at a map, I want the map to convey two things about the world and its landmasses:

*  Shapes.
*  Sizes.

Unfortunately, the mathematics of creating maps tells us that it's *impossible* to create a map that accurately represents both shapes and sizes. But that doesn't mean that all attempts a making map are equal. In this codebase, I'll try to find a balance between these two desirables to create a "good" map of the world. To do this, we'll need to combine differential geometry, numerical analysis, and numerical optimization techniques. I'll try to give a high-level overview of how I'm doing this in the next few paragraphs, assuming some technical competence in math and numerical analysis. If you want to see more detail, look at the code or reach out to me!

Also, fair warning -- I'm not a cartographer, just a wandering physicist who likes differential geometry, computation, and world maps. So I'm more motivated by finding a good tradeoff between an equiareal and conformal map than I am in highlighting features of the Earth's geography. Which means that I'm really not advocating for you to print this map out and put it on your wall ;)


The Differential Geometry
==========================

To talk about shapes and sizes in maps, we first need a mathematical language to describe maps. That language is differential geometry, the same mathematical basis for general relativity. Differential geometry is a full upper level mathematics course, which I am not going to pretend to cover in full here. But I will give a quick overview of what we need to know for mapping. (If you want to learn differential geometry, I'll enthusiastically plug `this book <https://www.amazon.com/Differential-Geometry-Dover-Books-Mathematics/dp/0486667219/>`_.)

We start with a description of real, physical distances on the Earth. We can represent each point on the Earth's surface by a three-dimensional vector :math:`\vec{x}`, which we'll index with coordinates :math:`(u^1, u^2)`. (Note that I'm using a superscript here, not an exponent. I'll use parentheses when I want to denote an exponent.) If we write the coordinates in shorthand as :math:`(u^1, u^2) = u^\alpha`, we can denote each point on the Earth's surface compactly as :math:`\vec{x}(u^\alpha)`. You can think of the coordinates :math:`u^\alpha` as latitude and longitude or :math:`\theta` and :math:`\phi`, or -- more useful here -- as the :math:`x` and :math:`y` coordinates of a map.

Our end goal is to make a map where shapes and sizes on the map are the same as shapes and sizes on the Earth's surface. To do this, we need a way to relate distances on the Earth's surface to those on the map. Let's consider the distance on the Earth's surface between the point :math:`\vec{x}(u^\alpha)` and :math:`\vec{x}(u^\alpha + v^\alpha)`, where :math:`v^\alpha` is small in some sense. We can write this distance :math:`s` as:

 ..  math::

    (s)^2 = \|\vec{x}(u_\alpha + v_\alpha) - \vec{x}(u_\alpha)\|^2

Taylor expanding :math:`\vec{x}(u^\alpha + v^\alpha)` gives

 ..  math::

    (s)^2 = \frac {\partial \vec{x}} {\partial u_1} \, \cdot \, \frac {\partial \vec{x}} {\partial u_1} \, (d_1)^2 + 2 \frac {\partial \vec{x}} {\partial u_1} \, \cdot \, \frac {\partial \vec{x}} {\partial u_2} {d_1 d_2} + \frac {\partial \vec{x}} {\partial u_2} \frac {\partial \vec{x}} {\partial u_2} (d_2)^2


We can write this compactly as

 ..  math::

    (s)^2 = g_{\alpha \beta} v^\alpha d^\beta

where we sum over repeated indices, and where we have introduced the *metric*

 ..  math::

    g_{\alpha \beta} = \frac {\partial \vec{x}} {\partial u_\alpha} \, \cdot \, \frac {\partial \vec{x}} {\partial u_\beta}

The metric :math:`g_{\alpha \beta}` in general changes from point to point in the map. As you can see from its definition, at each point on the map the metric can be represented as a symmetric matrix, although there is a slightly more powerful way to think about the metric, as we'll see a little later. If the map perfectly represented the world, then all the distances between all the points would be proportional, and the metric would be the identity:

 ..  math::

    \mathrm{Perfect,\,flat\,metric} = \left( \begin{array}{cc} 1 & 0 \\ 0 & 1 \end{array} \right)

The metric turns out to *the* most important thing in differential geometry. The metric determines not only distances, but also angles and areas of objects. For instance, if we draw a square on the map, with a lower-left corner at :math:`u^\alpha` and an upper right corner at :math:`u^\alpha + v^\alpha`, the area of the corresponding region on the Earth's surface is :math:`\sqrt{g} \, dv^1 dv^2`, where :math:`\sqrt{g}` is the determinant of the metric. Likwise, if we draw two lines, each starting at :math:`u^\alpha` but one ending at :math:`u^\alpha + v^\alpha` and the other ending at :math:`u^\alpha + t^\alpha`, the angle between those two lines on the Earth's surface is :math:`\theta = \mathrm{arccos}(g_{\alpha \beta} t^\alpha v^\beta)`.

TODO something about maps here? -- there is a conceptual leap here which is that the distances on the map are the distance in the plane.... Also we need to talk about transforming metrics, conformal maps, equiareal maps, the combination of the two gives a perfect map.

Now, at this stage, it should be obvious how to create the perfect map: Find a coordinate transformation such that the metric :math:`g_{\alpha \beta}` is the identity. Unfortunately, a `classic theorem <https://en.wikipedia.org/wiki/Theorema_Egregium>`_ of differential geometry proves that it is not possible to map the surface of a sphere to a flat plane without deformations. The reason has to do with the Gaussian curvature, which depends only on the metric. If it were possible to map the sphere to the plane without distortion, then the metric on the map would be equal to the metric on the sphere, which would imply that the Gaussian curvature of the sphere is equal to that of the map. But the Gaussian curvature of a plane is 0 and the Gaussian curvature of a sphere is positive. So it's not posible to map a sphere to a plane without distortion.

However, it *is* possible to create a conformal map of a sphere onto a plane, which preserves the angles between any two lines -- the classic `Mercator projection <https://en.wikipedia.org/wiki/Mercator_projection>`_ is an example of a conformal map. And it *is* possible to create an equiareal map of a sphere onto a plane, preserving the area of any shape -- some examples of equiareal projections are the `Lambert cylindrical <https://en.wikipedia.org/wiki/Lambert_cylindrical_equal-area_projection>`_ and the `Mollweide <https://en.wikipedia.org/wiki/Mollweide_projection>`_. It's just *not* possible to map a sphere onto a plane with a projection that is both conformal and equiareal. More importantly, though, there are *many* different conformal map projections, and *many* different conformal map projections.


The Solution
============

What I will do here is look for "the best" equiareal map projection. Specifically, I will look for the equiareal map projection that minimizes the distortion, as measured by some cost function. We'll find the best map projection numerically, so we'll want a cost function that allows for fast numerical computation and optimization. Then we'll parameterize the space of possible maps and search through that parameter space to find the best map. I'll choose the cost function and the parameterization of the maps with efficient computation in mind.

To penalize deviations from non-conformality, we take the sum of the squares of the difference between the metric and a flat metric:

 ..  math::

    \int \, dx\, dy \, \sum_{\alpha, \beta} \left( g_{\alpha \beta} - \delta_{\alpha \beta} \right)^2

We also need to constrain the map to be equiareal. To do this, we use a Lagrange multiplier times another sum of squares, to give the total cost function as:

 ..  math::

    C(\theta) = \int \, dx\, dy \, \sum_{\alpha, \beta} \left( g_{\alpha \beta} - \delta_{\alpha \beta} \right)^2 + \lambda (g - 1)^2

We need to efficiently evaluate this integral over a 2D range of points. We do this using Gauss-Legendre quadrature. 


d.  Gaussian quadrature to make it converge rapidly.
c.  Cost function as sum of squares to make it numerically simple.

3.  How do we parameterize the distribution?

    a.  Polynomial = linear, easy to calculate derivatives
    b.  Remove some degenerate constraints (piston, rotation, possibly even)

