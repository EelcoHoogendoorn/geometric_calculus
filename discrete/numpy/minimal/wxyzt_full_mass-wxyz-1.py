"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = m * phi * wxyz + M * phi,
in the full algebra of w+x+y+z+t-, where w is a compact dimension

Interestingly, the sps-wxyz mass potential term,
does not manage to bind the initial excitation in this space.
"""
from common import *

quad = quadratic((32, 32, 32))
mw = 1#quad / 1.5
m = quad * .2
M = quad * 1
(edw, idw), (edx, idx), (edy, idy), (edz, idz), (edt, idt) = partials(mw, 1, 1, 1, 1/4)

def leapfrog(phi):
	s, w, x, y, z, t, wx, wy, xy, wz, xz, yz, wt, xt, yt, zt, wxy, wxz, wyz, xyz, wxt, wyt, xyt, wzt, xzt, yzt, wxyz, wxyt, wxzt, wyzt, xyzt, wxyzt = phi
	edt(s, -(+idw(wt) + idx(xt) + idy(yt) + idz(zt) + m * interpolate(wxyzt, +1, +1, +1, +1) + M * t))  # t
	edt(w, +(+edw(t) - idx(wxt) - idy(wyt) - idz(wzt) - m * interpolate(xyzt, -1, +1, +1, +1) + M * wt))  # wt
	edt(x, +(+idw(wxt) + edx(t) - idy(xyt) - idz(xzt) + m * interpolate(wyzt, +1, -1, +1, +1) + M * xt))  # xt
	edt(y, +(+idw(wyt) + idx(xyt) + edy(t) - idz(yzt) - m * interpolate(wxzt, +1, +1, -1, +1) + M * yt))  # yt
	edt(z, +(+idw(wzt) + idx(xzt) + idy(yzt) + edz(t) + m * interpolate(wxyt, +1, +1, +1, -1) + M * zt))  # zt
	edt(wx, -(+edw(xt) - edx(wt) + idy(wxyt) + idz(wxzt) - m * interpolate(yzt, -1, -1, +1, +1) + M * wxt))  # wxt
	edt(wy, -(+edw(yt) - idx(wxyt) - edy(wt) + idz(wyzt) + m * interpolate(xzt, -1, +1, -1, +1) + M * wyt))  # wyt
	edt(xy, -(+idw(wxyt) + edx(yt) - edy(xt) + idz(xyzt) - m * interpolate(wzt, +1, -1, -1, +1) + M * xyt))  # xyt
	edt(wz, -(+edw(zt) - idx(wxzt) - idy(wyzt) - edz(wt) - m * interpolate(xyt, -1, +1, +1, -1) + M * wzt))  # wzt
	edt(xz, -(+idw(wxzt) + edx(zt) - idy(xyzt) - edz(xt) + m * interpolate(wyt, +1, -1, +1, -1) + M * xzt))  # xzt
	edt(yz, -(+idw(wyzt) + idx(xyzt) + edy(zt) - edz(yt) - m * interpolate(wxt, +1, +1, -1, -1) + M * yzt))  # yzt
	edt(wxy, +(+edw(xyt) - edx(wyt) + edy(wxt) - idz(wxyzt) + m * interpolate(zt, -1, -1, -1, +1) + M * wxyt))  # wxyt
	edt(wxz, +(+edw(xzt) - edx(wzt) + idy(wxyzt) + edz(wxt) - m * interpolate(yt, -1, -1, +1, -1) + M * wxzt))  # wxzt
	edt(wyz, +(+edw(yzt) - idx(wxyzt) - edy(wzt) + edz(wyt) + m * interpolate(xt, -1, +1, -1, -1) + M * wyzt))  # wyzt
	edt(xyz, +(+idw(wxyzt) + edx(yzt) - edy(xzt) + edz(xyt) - m * interpolate(wt, +1, -1, -1, -1) + M * xyzt))  # xyzt
	edt(wxyz, -(+edw(xyzt) - edx(wyzt) + edy(wxzt) - edz(wxyt) + m * interpolate(t, -1, -1, -1, -1) + M * wxyzt))  # wxyzt
	idt(t, +(+idw(w) + idx(x) + idy(y) + idz(z) + m * interpolate(wxyz, +1, +1, +1, +1) + M * s))  # s
	idt(wt, -(+edw(s) - idx(wx) - idy(wy) - idz(wz) - m * interpolate(xyz, -1, +1, +1, +1) + M * w))  # w
	idt(xt, -(+idw(wx) + edx(s) - idy(xy) - idz(xz) + m * interpolate(wyz, +1, -1, +1, +1) + M * x))  # x
	idt(yt, -(+idw(wy) + idx(xy) + edy(s) - idz(yz) - m * interpolate(wxz, +1, +1, -1, +1) + M * y))  # y
	idt(zt, -(+idw(wz) + idx(xz) + idy(yz) + edz(s) + m * interpolate(wxy, +1, +1, +1, -1) + M * z))  # z
	idt(wxt, +(+edw(x) - edx(w) + idy(wxy) + idz(wxz) - m * interpolate(yz, -1, -1, +1, +1) + M * wx))  # wx
	idt(wyt, +(+edw(y) - idx(wxy) - edy(w) + idz(wyz) + m * interpolate(xz, -1, +1, -1, +1) + M * wy))  # wy
	idt(xyt, +(+idw(wxy) + edx(y) - edy(x) + idz(xyz) - m * interpolate(wz, +1, -1, -1, +1) + M * xy))  # xy
	idt(wzt, +(+edw(z) - idx(wxz) - idy(wyz) - edz(w) - m * interpolate(xy, -1, +1, +1, -1) + M * wz))  # wz
	idt(xzt, +(+idw(wxz) + edx(z) - idy(xyz) - edz(x) + m * interpolate(wy, +1, -1, +1, -1) + M * xz))  # xz
	idt(yzt, +(+idw(wyz) + idx(xyz) + edy(z) - edz(y) - m * interpolate(wx, +1, +1, -1, -1) + M * yz))  # yz
	idt(wxyt, -(+edw(xy) - edx(wy) + edy(wx) - idz(wxyz) + m * interpolate(z, -1, -1, -1, +1) + M * wxy))  # wxy
	idt(wxzt, -(+edw(xz) - edx(wz) + idy(wxyz) + edz(wx) - m * interpolate(y, -1, -1, +1, -1) + M * wxz))  # wxz
	idt(wyzt, -(+edw(yz) - idx(wxyz) - edy(wz) + edz(wy) + m * interpolate(x, -1, +1, -1, -1) + M * wyz))  # wyz
	idt(xyzt, -(+idw(wxyz) + edx(yz) - edy(xz) + edz(xy) - m * interpolate(w, +1, -1, -1, -1) + M * xyz))  # xyz
	idt(wxyzt, +(+edw(xyz) - edx(wyz) + edy(wxz) - edz(wxy) + m * interpolate(s, -1, -1, -1, -1) + M * wxyz))  # wxyz

phi = (np.random.normal(size=(32, 2, 1, 1, 1)) * np.exp(-quad * 16)).astype(np.float32)
filter_lightlike(phi)
filter_stationary(leapfrog, phi)
color = lambda phi: np.abs(phi[2:5, :, 16]).mean(1)
show_animation(leapfrog, color, phi)
