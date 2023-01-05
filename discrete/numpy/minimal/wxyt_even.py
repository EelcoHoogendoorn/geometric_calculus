"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = 0,
where phi is an even grade field of w+x+y+t-, and w is a compact dimension
"""
from common import *

quad = quadratic((64, 64))
mw = quad / 2
(edw, idw), (edx, idx), (edy, idy), (edt, idt) = partials(mw, 1, 1, 1 / 2)

def leapfrog(phi):
	s, wx, wy, wt, xy, xt, yt, wxyt = phi
	edt(s, -(+idw(wt) + idx(xt) + idy(yt)))  # t
	edt(wx, -(+edw(xt) - edx(wt) + idy(wxyt)))  # wxt
	edt(wy, -(+edw(yt) - idx(wxyt) - edy(wt)))  # wyt
	edt(xy, -(+idw(wxyt) + edx(yt) - edy(xt)))  # xyt
	idt(wt, -(+edw(s) - idx(wx) - idy(wy)))  # w
	idt(xt, -(+idw(wx) + edx(s) - idy(xy)))  # x
	idt(yt, -(+idw(wy) + idx(xy) + edy(s)))  # y
	idt(wxyt, -(+edw(xy) - edx(wy) + edy(wx)))  # wxy

phi = (np.random.normal(size=(8, 2, 1, 1)) * np.exp(-quad * 16)).astype(np.float32)
filter_lightlike(phi)
filter_stationary(leapfrog, phi, 3)
color = lambda phi: np.abs(phi[4:7]).mean(1)
show_animation(leapfrog, color, phi)
