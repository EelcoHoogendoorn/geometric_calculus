"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = 0,
in the odd subalgebra of w+x+y+z+t-, where w is a compact dimension
"""
from common import *

quad = quadratic((32, 32, 32))
metric_w = quad / 1.5

edw, idw = ds(0, metric_w)
edx, idx = ds(1)
edy, idy = ds(2)
edz, idz = ds(3)
edt, idt = dt(1 / 4)

def leapfrog(phi):
	w, x, y, z, t, wxy, wxz, wyz, xyz, wxt, wyt, wzt, xyt, xzt, yzt, wxyzt = phi
	edt(w, +(+edw(t) - idx(wxt) - idy(wyt) - idz(wzt)))  # wt
	edt(x, +(+idw(wxt) + edx(t) - idy(xyt) - idz(xzt)))  # xt
	edt(y, +(+idw(wyt) + idx(xyt) + edy(t) - idz(yzt)))  # yt
	edt(z, +(+idw(wzt) + idx(xzt) + idy(yzt) + edz(t)))  # zt
	edt(wxy, +(+edw(xyt) - edx(wyt) + edy(wxt) - idz(wxyzt)))  # wxyt
	edt(wxz, +(+edw(xzt) - edx(wzt) + idy(wxyzt) + edz(wxt)))  # wxzt
	edt(wyz, +(+edw(yzt) - idx(wxyzt) - edy(wzt) + edz(wyt)))  # wyzt
	edt(xyz, +(+idw(wxyzt) + edx(yzt) - edy(xzt) + edz(xyt)))  # xyzt
	idt(t, +(+idw(w) + idx(x) + idy(y) + idz(z)))  # s
	idt(wxt, +(+edw(x) - edx(w) + idy(wxy) + idz(wxz)))  # wx
	idt(wyt, +(+edw(y) - idx(wxy) - edy(w) + idz(wyz)))  # wy
	idt(wzt, +(+edw(z) - idx(wxz) - idy(wyz) - edz(w)))  # wz
	idt(xyt, +(+idw(wxy) + edx(y) - edy(x) + idz(xyz)))  # xy
	idt(xzt, +(+idw(wxz) + edx(z) - idy(xyz) - edz(x)))  # xz
	idt(yzt, +(+idw(wyz) + idx(xyz) + edy(z) - edz(y)))  # yz
	idt(wxyzt, +(+edw(xyz) - edx(wyz) + edy(wxz) - edz(wxy)))  # wxyz

phi = (np.random.normal(size=(16, 2, 1, 1, 1)) * np.exp(-quad * 16)).astype(np.float32)
phi -= phi.mean(1, keepdims=True)   # filter out lightlike modes
filter_stationary(leapfrog, phi)    # filter non-propagating modes

color = lambda phi: np.abs(phi[-4:-1, :, 16]).mean(1)
animate(leapfrog, color, phi)
