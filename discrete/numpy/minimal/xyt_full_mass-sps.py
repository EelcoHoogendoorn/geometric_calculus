"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = m * phi * I * t = m * phi * xy,
in the full algebra of x+y+t-

We note that a zero order term of this form requires interpolation,
for a stencil without a spatial bias.

We have been unable to find a configuration with nonzero mass term, that does not diverge.
Note that for opposing space-time signature, stable configurations do exist.
"""
from common import *

quad = quadratic((64, 64))
# m = quad / 4
m = quad

edx, idx = ds(0)
edy, idy = ds(1)
edt, idt = dt(1/4)

def leapfrog(phi):
	s, x, y, xy, t, xt, yt, xyt = phi
	edt(s, -(+idx(xt)+idy(yt) -m*interpolate(xyt, +1,+1)))	 # t
	edt(x, +(+edx(t)-idy(xyt) -m*interpolate(yt, -1,+1)))	 # xt
	edt(y, +(+idx(xyt)+edy(t) +m*interpolate(xt, +1,-1)))	 # yt
	edt(xy, -(+edx(yt)-edy(xt) +m*interpolate(t, -1,-1)))	 # xyt
	idt(t, +(+idx(x)+idy(y) -m*interpolate(xy, +1,+1)))	 # s
	idt(xt, -(+edx(s)-idy(xy) -m*interpolate(y, -1,+1)))	 # x
	idt(yt, -(+idx(xy)+edy(s) +m*interpolate(x, +1,-1)))	 # y
	idt(xyt, +(+edx(y)-edy(x) +m*interpolate(s, -1,-1)))	 # xy


phi = (np.random.normal(size=(8, 1, 1)) * np.exp(-quad * 16)).astype(np.float32)
# filter non-propagating modes
filter_stationary(leapfrog, phi)

color = lambda phi: np.abs(phi[[1, 2, 4]])
animate(leapfrog, color, phi)
