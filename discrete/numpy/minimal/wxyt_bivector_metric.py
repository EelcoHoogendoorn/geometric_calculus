"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = 0,
where phi is a bivector field of w+x+y+t-, and w is a compact dimension

A metric distortion is applied here to the noncompact dimensions,
which produces qualitatively different bound states from a scalar potential
"""
from common import *

quad = quadratic((64, 64))
mw = .08#quad / 4
mx = my = mt = 1 - np.exp(-quad * 2) / 1.5
(edw, idw), (edx, idx), (edy, idy), (edt, idt) = partials(mw, mx, my, mt / 2)

def leapfrog(phi):
	wx, wy, xy, wt, xt, yt = phi
	edt(wx, -(+edw(xt) - edx(wt)))  # wxt
	edt(wy, -(+edw(yt) - edy(wt)))  # wyt
	edt(xy, -(+edx(yt) - edy(xt)))  # xyt
	idt(wt, -(-idx(wx) - idy(wy)))  # w
	idt(xt, -(+idw(wx) - idy(xy)))  # x
	idt(yt, -(+idw(wy) + idx(xy)))  # y

phi = (np.random.normal(size=(6, 2, 1, 1)) * np.exp(-quad * 16)).astype(np.float32)
filter_lightlike(phi)
filter_stationary(leapfrog, phi)
color = lambda phi: np.abs(phi[[2, 4, 5]]).mean(1)
animate(leapfrog, color, phi, 3)