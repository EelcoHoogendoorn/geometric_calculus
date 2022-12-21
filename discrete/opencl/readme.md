So far, the performance of jax.roll based implementation is a little disappointing.

Covolution based solutions are also not looking very satisfying, presenting limits in dimensionality, and a bunch of hackery.

Something like loopy seems like the best bet to get decent performance, without losing generality. pystella may serve as a nice inspiration there.

loopy can be pretty dense; but this is a pretty gentle intro: https://github.com/zachjweiner/pystella/blob/main/examples/codegen-tutorial.ipynb

so far my naive implementation shows a few x speedup over compiled jax or numpy. not bad, for a start, given the room for improvement. interesting to note that my overheating laptop is still getting smoked by my mobile phone running the shadertoy implementation... guess that shows the power of texture caching in the hardware. also the lack of polish of opencl compared to jax is annoying...

need to work metric and mass terms into loopy