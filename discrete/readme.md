# Readme

This submodule contains Discrete Geometric Calculus functionality.

This is currently restricted to discrete spaces formed from grids of n-cubes, with exclusively periodic boundary conditions.

This keeps the code nice and simple, while allowing for efficient stencil-based computations, without imposing many practical constraints. Alternative boundary conditions or curved computational domains can simply be encoded in the parameters of the field equations themselves, which is typically a quite competitive solution in terms of performance and convergence, compared to trying to make your mesh elements conform to the specifics of the modelling problem at hand. That being said, other valid approaches to discrete geometric calculus exist; this is just what's implemented here.

Note that the motivation for working with n-cube grids, is that on such grids, all partial derivatives that may appear in valid geometric-calculus expressions, can be written as simple upwind and downwind first order difference operators, stating relationships between n-cubes and their n-1 boundary cubes.


## TODO

so much cleanup and unification... need some kind of clean way to compose code for pure geometric derivative timestepping, vs tacking on other terms, and folding metric variations in the object structure

want numga-multivector interop.
want the ability to set up plane waves and so on

Implement eigensolver; preferaby using explitly periodic time axis. got an mvp; but it is quite slow and brittle

at least give nonlinear a brief spin in new jax context
being able to get some form of soliton going would be so cool
wrt nonlinear; had more interesting results with quad terms; but dimensionally wrong

https://en.wikipedia.org/wiki/Sine-Gordon_equation
sine-gordon looking really neat. if we can get something like that running in nd awesome. if not for physics, at least for quantum pong. those bound elastic states are amazing; havnt gotten very close yet though

