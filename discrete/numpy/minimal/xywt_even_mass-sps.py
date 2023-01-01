"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = m * phi * It = m * phi * wxy,
in the even subalgebra of w+x+y+t-

We note that a zero order term of this form requires interpolation,
for a stencil without a spatial bias.
"""
from common import *

quad = quadratic((64, 64))
# m = 0.5
m = quad / 2

edw, idw = ds(0, m)
edx, idx = ds(1)
edy, idy = ds(2)
edt, idt = dt(1/3)

def interpolate(phi, axes):
	for i, s in enumerate(axes):
		phi = (phi + np.roll(phi, shift=-s, axis=i)) / 2
	return phi

def leapfrog(phi):
	s, xy, xw, yw, xt, yt, wt, xywt = phi
	edt(s,    +idx(xt)   +idy(yt)   +idw(wt)   +m*interpolate(xywt, (-1, -1, -1)))  # t
	edt(xy,   +edx(yt)   -edy(xt)   +idw(xywt) -m*interpolate(wt, (+1, +1, -1)))  # xyt
	edt(xw,   +edx(wt)   -idy(xywt) -edw(xt)   +m*interpolate(yt, (+1, -1, +1)))  # xzt
	edt(yw,   +idx(xywt) +edy(wt)   -edw(yt)   -m*interpolate(xt, (-1, +1, +1)))  # yzt
	idt(xt,   +edx(s)    -idy(xy)   -idw(xw)   +m*interpolate(yw, (+1, -1, -1)))  # x
	idt(yt,   +idx(xy)   +edy(s)    -idw(yw)   -m*interpolate(xw, (-1, +1, -1)))  # y
	idt(wt,   +idx(xw)   +idy(yw)   +edw(s)    +m*interpolate(xy, (-1, -1, +1)))  # z
	idt(xywt, +edx(yw)   -edy(xw)   +edw(xy)   -m*interpolate(s, (+1, +1, +1)))  # xyz

phi = np.random.normal(size=(8, 2, 1, 1)) * np.exp(-quad * 32)
phi -= phi.mean(axis=1, keepdims=True)
filter_stationary(leapfrog, phi)
color = lambda phi: np.abs(phi[1:4]).mean(1)
animate(leapfrog, color, phi)
