"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = m * xyz * phi,
in the even subalgebra of w+x+y+z+t-, where w is a compact dimension
"""
from common import *

quad = quadratic((32, 32, 32))
mw = quad / 1.5
m = quad * .1
(edw, idw), (edx, idx), (edy, idy), (edz, idz), (edt, idt) = partials(mw, 1, 1, 1, 1 / 3)

def leapfrog(phi):
	(s, wx, wy, wz, xy, xz, yz, wxyz), (wt, xt, yt, zt, wxyt, wxzt, wyzt, xyzt) = phi
	edt(s, -(+idw(wt) + idx(xt) + idy(yt) + idz(zt) + m * interpolate(xyzt, 0, +1, +1, +1)))  # t
	edt(wx, -(+edw(xt) - edx(wt) + idy(wxyt) + idz(wxzt) - m * interpolate(wyzt, 0, -1, +1, +1)))  # wxt
	edt(wy, -(+edw(yt) - idx(wxyt) - edy(wt) + idz(wyzt) + m * interpolate(wxzt, 0, +1, -1, +1)))  # wyt
	edt(xy, -(+idw(wxyt) + edx(yt) - edy(xt) + idz(xyzt) - m * interpolate(zt, 0, -1, -1, +1)))  # xyt
	edt(wz, -(+edw(zt) - idx(wxzt) - idy(wyzt) - edz(wt) - m * interpolate(wxyt, 0, +1, +1, -1)))  # wzt
	edt(xz, -(+idw(wxzt) + edx(zt) - idy(xyzt) - edz(xt) + m * interpolate(yt, 0, -1, +1, -1)))  # xzt
	edt(yz, -(+idw(wyzt) + idx(xyzt) + edy(zt) - edz(yt) - m * interpolate(xt, 0, +1, -1, -1)))  # yzt
	edt(wxyz, -(+edw(xyzt) - edx(wyzt) + edy(wxzt) - edz(wxyt) + m * interpolate(wt, 0, -1, -1, -1)))  # wxyzt
	idt(wt, -(+edw(s) - idx(wx) - idy(wy) - idz(wz) - m * interpolate(wxyz, 0, +1, +1, +1)))  # w
	idt(xt, -(+idw(wx) + edx(s) - idy(xy) - idz(xz) + m * interpolate(yz, 0, -1, +1, +1)))  # x
	idt(yt, -(+idw(wy) + idx(xy) + edy(s) - idz(yz) - m * interpolate(xz, 0, +1, -1, +1)))  # y
	idt(zt, -(+idw(wz) + idx(xz) + idy(yz) + edz(s) + m * interpolate(xy, 0, +1, +1, -1)))  # z
	idt(wxyt, -(+edw(xy) - edx(wy) + edy(wx) - idz(wxyz) + m * interpolate(wz, 0, -1, -1, +1)))  # wxy
	idt(wxzt, -(+edw(xz) - edx(wz) + idy(wxyz) + edz(wx) - m * interpolate(wy, 0, -1, +1, -1)))  # wxz
	idt(wyzt, -(+edw(yz) - idx(wxyz) - edy(wz) + edz(wy) + m * interpolate(wx, 0, +1, -1, -1)))  # wyz
	idt(xyzt, -(+idw(wxyz) + edx(yz) - edy(xz) + edz(xy) - m * interpolate(s, 0, -1, -1, -1)))  # xyz

phi = (np.random.normal(size=(2, 8, 2, 1, 1, 1)) * np.exp(-quad * 16)).astype(np.float32)
filter_lightlike(phi, 1)
filter_stationary(leapfrog, phi)
color = lambda phi: np.abs(phi[0, 1:4, :, 16]).mean(1)
animate(leapfrog, color, phi)
