"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = m * xyz * phi,
over the even subalgebra of x+y+z+t-

This mass term is unusual, since a scalar potential profile on m,
contains both a bound and unbound component
"""
from common import *

quad = quadratic((64, 64, 64))
m = 1 - np.exp(-quad*2)

edx, idx = ds(0)
edy, idy = ds(1)
edz, idz = ds(2)
edt, idt = dt(1/3)

def leapfrog(phi):
	s, xy, xz, yz, xt, yt, zt, xyzt = phi
	edt(s, -(+idx(xt) + idy(yt) + idz(zt) + m * interpolate(xyzt, +1, +1, +1)))  # t
	edt(xy, -(+edx(yt) - edy(xt) + idz(xyzt) - m * interpolate(zt, -1, -1, +1)))  # xyt
	edt(xz, -(+edx(zt) - idy(xyzt) - edz(xt) + m * interpolate(yt, -1, +1, -1)))  # xzt
	edt(yz, -(+idx(xyzt) + edy(zt) - edz(yt) - m * interpolate(xt, +1, -1, -1)))  # yzt
	idt(xt, -(+edx(s) - idy(xy) - idz(xz) + m * interpolate(yz, -1, +1, +1)))  # x
	idt(yt, -(+idx(xy) + edy(s) - idz(yz) - m * interpolate(xz, +1, -1, +1)))  # y
	idt(zt, -(+idx(xz) + idy(yz) + edz(s) + m * interpolate(xy, +1, +1, -1)))  # z
	idt(xyzt, -(+edx(yz) - edy(xz) + edz(xy) - m * interpolate(s, -1, -1, -1)))  # xyz

phi = np.random.normal(size=(8, 1, 1, 1)) * np.exp(-quad * 32)
color = lambda phi: np.abs(phi[1:4, 32])
animate(leapfrog, color, phi)
