"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = 0,
in the even subalgebra of w+x+y+t-, where w is a compact dimension
"""
from common import *

quad = quadratic((64, 64))

edw, idw = ds(0, quad)
edx, idx = ds(1)
edy, idy = ds(2)
edt, idt = dt(1/3)

def leapfrog(phi):
	s, xy, xw, yw, xt, yt, wt, xywt = phi
	edt(s, +idx(xt) + idy(yt) + idw(wt))  # t
	edt(xy, +edx(yt) - edy(xt) + idw(xywt))  # xyt
	edt(xw, +edx(wt) - idy(xywt) - edw(xt))  # xwt
	edt(yw, +idx(xywt) + edy(wt) - edw(yt))  # ywt
	idt(xt, +edx(s) - idy(xy) - idw(xw))  # x
	idt(yt, +idx(xy) + edy(s) - idw(yw))  # y
	idt(wt, +idx(xw) + idy(yw) + edw(s))  # w
	idt(xywt, +edx(yw) - edy(xw) + edw(xy))  # xyw

phi = np.random.normal(size=(8, 2, 1, 1)) * np.exp(-quad * 16)
phi -= phi.mean(axis=1, keepdims=True)
filter_stationary(leapfrog, phi)
color = lambda phi: np.abs(phi[1:4]).mean(1)
animate(leapfrog, color, phi)
