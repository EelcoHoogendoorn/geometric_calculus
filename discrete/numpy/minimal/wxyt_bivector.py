"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = 0,
where phi is a bivector field of w+x+y+t-, and w is a compact dimension
"""
from common import *

quad = quadratic((64, 64))
metric_w = quad / 4
metric_q = 1 - np.exp(-quad * 4) / 4

edw, idw = ds(0, metric_w)
edx, idx = ds(1, metric_q)
edy, idy = ds(2, metric_q)
edt, idt = dt(metric_q / 2)

def leapfrog(phi):
	wx, wy, xy, wt, xt, yt = phi
	edt(wx, -(+edw(xt) - edx(wt)))  # wxt
	edt(wy, -(+edw(yt) - edy(wt)))  # wyt
	edt(xy, -(+edx(yt) - edy(xt)))  # xyt
	idt(wt, -(-idx(wx) - idy(wy)))  # w
	idt(xt, -(+idw(wx) - idy(xy)))  # x
	idt(yt, -(+idw(wy) + idx(xy)))  # y

phi = (np.random.normal(size=(6, 2, 1, 1)) * np.exp(-quad * 16)).astype(np.float32)
phi -= phi.mean(1, keepdims=True)   # filter out lightlike modes
# filter non-propagating modes
filter_stationary(leapfrog, phi)
# filter_stationary(leapfrog, phi)
# filter_stationary(leapfrog, phi)

color = lambda phi: np.abs(phi[[2, 4, 5]]).mean(1)
animate(leapfrog, color, phi)