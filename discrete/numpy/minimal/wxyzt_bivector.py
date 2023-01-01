"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = 0,
where phi is a bivector field of w+x+y+z+t-, and w is a compact dimension
"""
from common import *

quad = quadratic((32, 32, 32))
metric_w = quad / 1.5
metric_q = None#1 - np.exp(-quad * 4) / 4

edw, idw = ds(0, metric_w)
edx, idx = ds(1, metric_q)
edy, idy = ds(2, metric_q)
edz, idz = ds(3, metric_q)
edt, idt = dt(1 / 3)

def leapfrog(phi):
	xy, xz, yz, wx, wy, wz, wt, xt, yt, zt = phi
	edt(xy, -(+edx(yt) - edy(xt)))  # xyt
	edt(xz, -(+edx(zt) - edz(xt)))  # xzt
	edt(yz, -(+edy(zt) - edz(yt)))  # yzt
	edt(wx, -(+edw(xt) - edx(wt)))  # wxt
	edt(wy, -(+edw(yt) - edy(wt)))  # wyt
	edt(wz, -(+edw(zt) - edz(wt)))  # wzt
	idt(wt, -(-idx(wx) - idy(wy) - idz(wz)))  # w
	idt(xt, -(+idw(wx) - idy(xy) - idz(xz)))  # x
	idt(yt, -(+idw(wy) + idx(xy) - idz(yz)))  # y
	idt(zt, -(+idw(wz) + idx(xz) + idy(yz)))  # z

phi = (np.random.normal(size=(10, 2, 1, 1, 1)) * np.exp(-quad * 16)).astype(np.float32)
phi -= phi.mean(1, keepdims=True)   # filter out lightlike modes
# filter non-propagating modes
filter_stationary(leapfrog, phi)
# filter_stationary(leapfrog, phi)
# filter_stationary(leapfrog, phi)

color = lambda phi: np.abs(phi[:3, :, 16]).mean(1)
animate(leapfrog, color, phi)