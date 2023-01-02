"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = 0,
over the even subalgebra of x+y+z+t-

This is a model of massless neutrinos
"""
from common import *

quad = quadratic((64, 64, 64))
(edx, idx), (edy, idy), (edz, idz), (edt, idt) = partials(1, 1, 1, 1/3)

def leapfrog(phi):
	s, xy, xz, yz, xt, yt, zt, xyzt = phi
	edt(s, -(+idx(xt) + idy(yt) + idz(zt)))  # t
	edt(xy, -(+edx(yt) - edy(xt) + idz(xyzt)))  # xyt
	edt(xz, -(+edx(zt) - idy(xyzt) - edz(xt)))  # xzt
	edt(yz, -(+idx(xyzt) + edy(zt) - edz(yt)))  # yzt
	idt(xt, -(+edx(s) - idy(xy) - idz(xz)))  # x
	idt(yt, -(+idx(xy) + edy(s) - idz(yz)))  # y
	idt(zt, -(+idx(xz) + idy(yz) + edz(s)))  # z
	idt(xyzt, -(+edx(yz) - edy(xz) + edz(xy)))  # xyz

phi = np.random.normal(size=(8, 1, 1, 1)) * np.exp(-quad * 32)
filter_stationary(leapfrog, phi)
color = lambda phi: np.abs(phi[1:4, 32])
animate(leapfrog, color, phi)
