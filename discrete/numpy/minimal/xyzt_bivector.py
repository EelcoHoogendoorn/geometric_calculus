"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = 0,
over the bivectors of x+y+z+t-

This is just the classic Yee scheme for the Maxwell equation
"""
from common import *

quad = quadratic((64, 64, 64))
(edx, idx), (edy, idy), (edz, idz), (edt, idt) = partials(1, 1, 1, 1/3)

def leapfrog(phi):
	xy, xz, yz, xt, yt, zt = phi
	edt(xy, -(+edx(yt)-edy(xt)))	 # xyt
	edt(xz, -(+edx(zt)-edz(xt)))	 # xzt
	edt(yz, -(+edy(zt)-edz(yt)))	 # yzt
	idt(xt, -(-idy(xy)-idz(xz)))	 # x
	idt(yt, -(+idx(xy)-idz(yz)))	 # y
	idt(zt, -(+idx(xz)+idy(yz)))	 # z

phi = np.random.normal(size=(6, 1, 1, 1)) * np.exp(-quad * 32)
filter_stationary(leapfrog, phi)
color = lambda phi: np.abs(phi[:3, 32])
show_animation(leapfrog, color, phi)
