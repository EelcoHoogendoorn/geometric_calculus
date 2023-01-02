"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = m * phi * It = m * phi * wxy,
in the even subalgebra of w+x+y+t-

We note that a zero order term of this form requires interpolation,
for a stencil without a spatial bias.
"""
from common import *

quad = quadratic((64, 64))
# m = 0.5
m = quad
mw = quad
(edw, idw), (edx, idx), (edy, idy), (edt, idt) = partials(mw, 1, 1, 1/3)

def leapfrog(phi):
	s, wx, wy, xy, wt, xt, yt, wxyt = phi
	edt(s, -(+idw(wt) + idx(xt) + idy(yt) - m * interpolate(wxyt, +1, +1, +1)))  # t
	edt(wx, -(+edw(xt) - edx(wt) + idy(wxyt) + m * interpolate(yt, -1, -1, +1)))  # wxt
	edt(wy, -(+edw(yt) - idx(wxyt) - edy(wt) - m * interpolate(xt, -1, +1, -1)))  # wyt
	edt(xy, -(+idw(wxyt) + edx(yt) - edy(xt) + m * interpolate(wt, +1, -1, -1)))  # xyt
	idt(wt, -(+edw(s) - idx(wx) - idy(wy) - m * interpolate(xy, -1, +1, +1)))  # w
	idt(xt, -(+idw(wx) + edx(s) - idy(xy) + m * interpolate(wy, +1, -1, +1)))  # x
	idt(yt, -(+idw(wy) + idx(xy) + edy(s) - m * interpolate(wx, +1, +1, -1)))  # y
	idt(wxyt, -(+edw(xy) - edx(wy) + edy(wx) + m * interpolate(s, -1, -1, -1)))  # wxy

phi = np.random.normal(size=(8, 2, 1, 1)) * np.exp(-quad * 16)
filter_lightlike(phi)
filter_stationary(leapfrog, phi)
color = lambda phi: np.abs(phi[1:4]).mean(1)
animate(leapfrog, color, phi)
