So far, the performance of jax.roll based implementation being slower than the numpy implementation on CPU, is a little disappointing.

Covolution based solutions are also not looking very satisfying, presenting limits in dimensionality, and a bunch of hackery.

Something like loopy seems like the best bet to get decent performance, without losing generality. pystella may serve as a nice inspiration there.

loopy can be pretty dense; but this is a pretty gentle intro: https://github.com/zachjweiner/pystella/blob/main/examples/codegen-tutorial.ipynb