"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = m * phi,
where phi is an even grade field of x+y+t-, and w is a compact dimension
"""
from common import *

quad = quadratic((128, 128))
m = 0.005
mx = my = mt = 1 - np.exp(-quad * 3) * (1 - 0.15)
(edx, idx), (edy, idy), (edt, idt) = partials(mx, my, mt / 2)

def leapfrog(phi):
	# print((phi / mt).sum())   # this is a conserved quantity
	s, x, y, t, xy, xt, yt, xyt = phi
	edt(s, -(+idx(xt) + idy(yt) + m * t))  # t
	edt(x, +(+edx(t) - idy(xyt) + m * xt))  # xt
	edt(y, +(+idx(xyt) + edy(t) + m * yt))  # yt
	edt(xy, -(+edx(yt) - edy(xt) + m * xyt))  # xyt
	idt(t, +(+idx(x) + idy(y) + m * s))  # s
	idt(xt, -(+edx(s) - idy(xy) + m * x))  # x
	idt(yt, -(+idx(xy) + edy(s) + m * y))  # y
	idt(xyt, +(+edx(y) - edy(x) + m * xy))  # xy

phi = (np.random.normal(size=(8, 1, 1)) * np.exp(-quadratic((128, 128), [0.0, 0.0]) * 10**2)).astype(np.float32)
filter_stationary(leapfrog, phi, 1)
# show_xt(leapfrog, lambda phi: np.abs((phi)[[3, 5, 6]]).mean(1)[..., 64], phi, 512, 10*2)
color = lambda phi: np.abs(phi[4:7])
show_animation(leapfrog, color, phi, 10, scale=0.3)
