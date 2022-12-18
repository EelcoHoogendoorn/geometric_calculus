"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = m * phi * It = m * phi * xy,
in the even subalgebra of x+y+t-

We note that a zero order term of this form requires interpolation,
for a stencil without a spatial bias.
Note that with the even grade spatial pseudoscalar xy, the leapfrog scheme is broken
"""
import numpy as np
import matplotlib.pyplot as plt

def edt(lhs, rhs, courant=0.33):
	lhs -= rhs * courant
idt = edt

def interpolate(phi, axes):
	for i, s in enumerate(axes):
		phi = (phi + np.roll(phi, shift=s, axis=i)) / 2
	return phi
ed = lambda d: lambda a: np.roll(a, shift=-1, axis=d) - a
id = lambda d: lambda a: a - np.roll(a, shift=+1, axis=d)
edx, edy, edw = [ed(d) for d in range(3)]
idx, idy, idw = [id(d) for d in range(3)]

m = 0.1
def leapfrog(phi):
	s, xy, xt, yt = phi
	edt(s, +idx(xt) + idy(yt) - m * interpolate(xy, (-1, -1)))  # t
	edt(xy, +edx(yt) - edy(xt) + m * interpolate(s, (+1, +1)))  # xyt
	idt(xt, +edx(s) - idy(xy) + m * interpolate(yt, (+1, -1)))  # x
	idt(yt, +idx(xy) + edy(s) - m * interpolate(xt, (-1, +1)))  # y

x2 = np.linspace(-1, 1, 64) ** 2
phi = np.random.normal(size=(4, 1, 1)) * np.exp(-np.add.outer(x2, x2) * 16)
for i in range(4):
	plt.imshow(np.abs(phi[1:4]).T * 8)
	plt.show()
	for t in range(64):
		leapfrog(phi)
