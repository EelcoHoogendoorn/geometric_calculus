"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = m * phi,
in the full algebra of x-y-t+

We have been unable to find a configuration with nonzero mass term, that does not diverge.
Note that for opposing space-time signature, stable configurations do exist.
"""
from common import *

quad = quadratic((64, 64))
m = quad
(edx, idx), (edy, idy), (edt, idt) = partials(1, 1, 1/3)

def leapfrog(phi):
	s, x, y, xy, t, xt, yt, xyt = phi
	edt(s, -(-idx(xt)-idy(yt) +m*t))	 # t
	edt(x, +(+edx(t)+idy(xyt) +m*xt))	 # xt
	edt(y, +(-idx(xyt)+edy(t) +m*yt))	 # yt
	edt(xy, -(+edx(yt)-edy(xt) +m*xyt))	 # xyt
	idt(t, -(-idx(x)-idy(y) +m*s))	 # s
	idt(xt, +(+edx(s)+idy(xy) +m*x))	 # x
	idt(yt, +(-idx(xy)+edy(s) +m*y))	 # y
	idt(xyt, -(+edx(y)-edy(x) +m*xy))	 # xy

phi = (np.random.normal(size=(8, 1, 1)) * np.exp(-quad * 16)).astype(np.float32)
filter_stationary(leapfrog, phi)
color = lambda phi: np.abs(phi[[1, 2, 4]])
show_animation(leapfrog, color, phi)