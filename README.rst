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

Our end goal is to make a map where shapes and sizes on the map are the same as shapes and sizes on the Earth's surface. To do this, we need a way to relate distances on the Earth's surface to those on the map. Let's consider the distance on the Earth's surface between the point :math:`\vec{x}(u^\alpha)` and :math:`\vec{x}(u^\alpha + d^\alpha)`, where :math:`d^\alpha` is small in some sense. We can write this distance :math:`s` as:

 ..  math::

    (s)^2 = \|\vec{x}(u_\alpha + d_\alpha) - \vec{x}(u_\alpha)\|^2

Taylor expanding :math:`\vec{x}(u_\alpha + d_\alpha)` gives

 ..  math::

    (s)^2 = \frac {\partial \vec{x}} {\partial u_1} \, \cdot \, \frac {\partial \vec{x}} {\partial u_1} \, (d_1)^2 + 2 \frac {\partial \vec{x}} {\partial u_1} \, \cdot \, \frac {\partial \vec{x}} {\partial u_2} {d_1 d_2} + \frac {\partial \vec{x}} {\partial u_2} \frac {\partial \vec{x}} {\partial u_2} (d_2)^2


We can write this compactly as

 ..  math::

    (s)^2 = g_{\alpha \beta} d^\alpha d^\beta

where we sum over repeated indices, and where we have introduced the *metric*

 ..  math::

    g_{\alpha \beta} = \frac {\partial \vec{x}} {\partial u_\alpha} \, \cdot \, \frac {\partial \vec{x}} {\partial u_\beta}

The metric turns out to *the* most important thing in differential geometry. The metric determines not only distances, but also angles and areas of objects. For instance, the area of the paralleliped ...

What do I want to discuss?
 - concept of a map.
 - metric
 - getting the element of the area from the metric
 - getting shapes from the metric
 - transforming a metric.

A little algebra shows that we can calculate distances


1.  The problem: maps, equiareal, conformal, metrics, etc
    The solution is a tradeoff.

2.  How do we pick that tradeoff?

    a.  Cost function for non-conformality
    b.  Constraint for equiareal: Lagrange multiplier
    c.  Cost function as sum of squares to make it numerically simple.
    d.  Gaussian quadrature to make it converge rapidly.

3.  How do we parameterize the distribution?

    a.  Polynomial = linear, easy to calculate derivatives
    b.  Remove some degenerate constraints (piston, rotation, possibly even)

