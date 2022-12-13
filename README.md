Geometric Calculus
==================

This library provides geometric calculus functionality in python

Arguably, 'the geometric derivative is all you need'; it truly is the swiss army knife of partial differential equations.

Note that this library is not currently written with the purpose of being suitable for any production purposes; but rather to serve as an exploratory playground, and a way to facilitate the spread of interesting ideas.

It contains a submodule for both continuous and discrete geometric calculus, both interesting in their own right.


Examples
--------
Since vector calculus is a subset of geometric calculus, a wide range of physical models can be expressed using the geometric derivative in a concise, coordinate independent, and dimension independent manner. However, geometric calculus also permits the formulation of more 'exotic' types of equations:

<img src="discrete/article/31_compact_even_sps_mass_xy.gif" width="256" height="256"/> 


Discrete
--------
We implement the discrete geometric derivative over cubic grids in a variety of frameworks. Currently, there are numpy, JAX and openCL implementations, in varying stages of development.


Continuous
----------
The 'continuous' package provides functionality for working with continuous geometric calculus expressions.

This leverages JAX autodiff and numga to construct correct and compilable expressions, in any dimension and signature.

There are many ways to leverage such a system; but the one currently implemented is to use 'parameterized universal function approximators' (or 'neural networks' if you will) as function spaces to solve PDEs by optimizing their parameters directly. A small mvp JAX framework for stochastically sampling domains and optimization using gradient descent is implemented.

Using a single continuous function approximation stands in contrast to more classical function approximation schemes such as finite elements. This is an emerging approach and can be a little tricky to get to work but it compares favorably to traditional methods in high dimensional spaces, or if trying to solve problems for continuous ranges of input parameters, or trying to model equations with nonstandard terms, where no proven effective alternative methods are known.

Notes
-----
Note that the subpackages for continuous and discrete geometric calculus, while having strong conceptual overlap, do not currently share any code. The shared geometric algebra logic resides in the sister-project [numga](https://github.com/EelcoHoogendoorn/numga).

The reason they are included here in the same project is mostly a reflection of my own learning journey, to be able to explore the similarities and differences. Perhaps in the future a split might be more appropriate.
