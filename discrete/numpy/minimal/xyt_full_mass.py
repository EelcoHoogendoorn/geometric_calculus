"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = m * phi,
in the full algebra of x+y+t-
"""
from common import *

quad = quadratic((64, 64))
# m = 1 - np.exp(-quad / 4)
m = quad

edx, idx = ds(0)
edy, idy = ds(1)
edt, idt = dt(1/4)


def leapfrog(phi):
	s, x, y, xy, t, xt, yt, xyt = phi
	edt(s, -(+idx(xt) + idy(yt) + m * t))  # t
	edt(x, +(+edx(t) - idy(xyt) + m * xt))  # xt
	edt(y, +(+idx(xyt) + edy(t) + m * yt))  # yt
	edt(xy, -(+edx(yt) - edy(xt) + m * xyt))  # xyt
	idt(t, +(+idx(x) + idy(y) + m * s))  # s
	idt(xt, -(+edx(s) - idy(xy) + m * x))  # x
	idt(yt, -(+idx(xy) + edy(s) + m * y))  # y
	idt(xyt, +(+edx(y) - edy(x) + m * xy))  # xy


phi = (np.random.normal(size=(8, 1, 1)) * np.exp(-quad * 16)).astype(np.float32)
# filter non-propagating modes
filter_stationary(leapfrog, phi)

color = lambda phi: np.abs(phi[[1, 2, 4]])
animate(leapfrog, color, phi)


