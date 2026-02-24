"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = 0,
where phi is a bivector field of w+x+y+z+t-, and w is a compact dimension

This is the most Kaluza-Klein type equation;
whereby we take the maxwell equations, and give them a minimal extension,
with an additional compact dimension, while staying with a pure bivector field
"""
from common import *

quad = quadratic((32, 32, 32))
mw = quad / 1.5
mx = my = mz = mt = 1 - np.exp(-quad * 2) / 3
(edw, idw), (edx, idx), (edy, idy), (edz, idz), (edt, idt) = partials(mw, mx, my, mz, mt / 3)

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
filter_lightlike(phi)
filter_stationary(leapfrog, phi)
color = lambda phi: np.abs(phi[:3, :, 16]).mean(1)
show_animation(leapfrog, color, phi)