"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = 0,
where phi is an even grade field of w+x+y+t-, and w is a compact dimension
"""
from common import *

quad = quadratic((64, 64))
metric_w = quad / 2

edw, idw = ds(0, metric_w)
edx, idx = ds(1)
edy, idy = ds(2)
edt, idt = dt(1 / 2)

def leapfrog(phi):
	s, wx, wy, xy, wt, xt, yt, wxyt = phi
	edt(s, -(+idw(wt) + idx(xt) + idy(yt)))  # t
	edt(wx, -(+edw(xt) - edx(wt) + idy(wxyt)))  # wxt
	edt(wy, -(+edw(yt) - idx(wxyt) - edy(wt)))  # wyt
	edt(xy, -(+idw(wxyt) + edx(yt) - edy(xt)))  # xyt
	idt(wt, -(+edw(s) - idx(wx) - idy(wy)))  # w
	idt(xt, -(+idw(wx) + edx(s) - idy(xy)))  # x
	idt(yt, -(+idw(wy) + idx(xy) + edy(s)))  # y
	idt(wxyt, -(+edw(xy) - edx(wy) + edy(wx)))  # wxy

phi = (np.random.normal(size=(8, 2, 1, 1)) * np.exp(-quad * 16)).astype(np.float32)
phi -= phi.mean(1, keepdims=True)   # filter out lightlike modes
# filter non-propagating modes
filter_stationary(leapfrog, phi)
filter_stationary(leapfrog, phi)
filter_stationary(leapfrog, phi)

color = lambda phi: np.abs(phi[[3, 5, 6]]).mean(1)
animate(leapfrog, color, phi)
