"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = m * phi * I = m * phi * xyt,
in the even subalgebra of x+y+t-

Note that the leapfrog scheme is broken;
the update rule is no longer one of spacelike updated in terms of timelike, and vice versa

We note that a zero order term of this form requires interpolation,
to obtain an unbiased update rule.
We only interpolate over the spatial axes here, and temporally,
we simply 'bias towards the past', or use the last available values.

No obvious divergences are observable

Unlike the xt dual mass case, left versus right dualization with I, gives the same pattern of signs
"""
from common import *

quad = quadratic((64, 64))
m = quad
(edx, idx), (edy, idy), (edt, idt) = partials(1, 1, 1/3)

def leapfrog(phi):
	s, xy, xt, yt = phi
	edt(s, +idx(xt) + idy(yt) - m * interpolate(xy, +1, +1))  # t
	edt(xy, +edx(yt) - edy(xt) + m * interpolate(s, -1, -1))  # xyt
	idt(xt, +edx(s) - idy(xy) + m * interpolate(yt, -1, +1))  # x
	idt(yt, +idx(xy) + edy(s) - m * interpolate(xt, +1, -1))  # y

phi = np.random.normal(size=(4, 1, 1)) * np.exp(-quad * 32)
color = lambda phi: np.abs(phi[1:])
show_animation(leapfrog, color, phi)
