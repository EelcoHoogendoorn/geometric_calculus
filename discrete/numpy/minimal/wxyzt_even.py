"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = 0,
in the even subalgebra of w+x+y+z+t-, where w is a compact dimension
"""
from common import *

quad = quadratic((32, 32, 32))
mw = quad / 1.5
(edw, idw), (edx, idx), (edy, idy), (edz, idz), (edt, idt) = partials(mw, 1, 1, 1, 1 / 3)

def leapfrog(phi):
	(s, wx, wy, wz, xy, xz, yz, wxyz), (wt, xt, yt, zt, wxyt, wxzt, wyzt, xyzt) = phi
	edt(s, -(+idw(wt) + idx(xt) + idy(yt) + idz(zt)))  # t
	edt(wx, -(+edw(xt) - edx(wt) + idy(wxyt) + idz(wxzt)))  # wxt
	edt(wy, -(+edw(yt) - idx(wxyt) - edy(wt) + idz(wyzt)))  # wyt
	edt(wz, -(+edw(zt) - idx(wxzt) - idy(wyzt) - edz(wt)))  # wzt
	edt(xy, -(+idw(wxyt) + edx(yt) - edy(xt) + idz(xyzt)))  # xyt
	edt(xz, -(+idw(wxzt) + edx(zt) - idy(xyzt) - edz(xt)))  # xzt
	edt(yz, -(+idw(wyzt) + idx(xyzt) + edy(zt) - edz(yt)))  # yzt
	edt(wxyz, -(+edw(xyzt) - edx(wyzt) + edy(wxzt) - edz(wxyt)))  # wxyzt
	idt(wt, -(+edw(s) - idx(wx) - idy(wy) - idz(wz)))  # w
	idt(xt, -(+idw(wx) + edx(s) - idy(xy) - idz(xz)))  # x
	idt(yt, -(+idw(wy) + idx(xy) + edy(s) - idz(yz)))  # y
	idt(zt, -(+idw(wz) + idx(xz) + idy(yz) + edz(s)))  # z
	idt(wxyt, -(+edw(xy) - edx(wy) + edy(wx) - idz(wxyz)))  # wxy
	idt(wxzt, -(+edw(xz) - edx(wz) + idy(wxyz) + edz(wx)))  # wxz
	idt(wyzt, -(+edw(yz) - idx(wxyz) - edy(wz) + edz(wy)))  # wyz
	idt(xyzt, -(+idw(wxyz) + edx(yz) - edy(xz) + edz(xy)))  # xyz

phi = (np.random.normal(size=(2, 8, 2, 1, 1, 1)) * np.exp(-quad * 16)).astype(np.float32)
filter_lightlike(phi, 1)
filter_stationary(leapfrog, phi)
color = lambda phi: np.abs(phi[0, 1:4, :, 16]).mean(1)
show_animation(leapfrog, color, phi)
