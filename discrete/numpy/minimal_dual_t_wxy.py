"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = m * phi * It = m * phi * xyw,
in the even subalgebra of x+y+w+t-

We note that a zero order term of this form requires interpolation,
for a stencil without a spatial bias.
Note that with an odd grade spatial pseudoscalar xyz, the leapfrog scheme is preserved
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

m = 0.0
def leapfrog(phi):
	s, xy, xw, yw, xt, yt, wt, xywt = phi
	edt(s, +idx(xt) + idy(yt) + idw(wt)     + m*interpolate(xywt, (-1, -1, -1)))  # t
	edt(xy, +edx(yt) - edy(xt) + idw(xywt)  - m*interpolate(wt, (+1, +1, -1)))  # xyt
	edt(xw, +edx(wt) - idy(xywt) - edw(xt)  + m*interpolate(yt, (+1, -1, +1)))  # xzt
	edt(yw, +idx(xywt) + edy(wt) - edw(yt)  - m*interpolate(xt, (-1, +1, +1)))  # yzt
	idt(xt, +edx(s) - idy(xy) - idw(xw)     + m*interpolate(yw, (+1, -1, -1)))  # x
	idt(yt, +idx(xy) + edy(s) - idw(yw)     - m*interpolate(xw, (-1, +1, -1)))  # y
	idt(wt, +idx(xw) + idy(yw) + edw(s)     + m*interpolate(xy, (-1, -1, +1)))  # z
	idt(xywt, +edx(yw) - edy(xw) + edw(xy)  - m*interpolate(s, (+1, +1, +1)))  # xyz

phi = np.zeros((8, 64, 64, 2))
x2 = np.linspace(-1, 1, 64) ** 2
phi[..., 0] = np.random.normal(size=(8, 1, 1)) * np.exp(-np.add.outer(x2, x2) * 16)
amp, norm = [], []
for i in range(256):
	norm.append(np.linalg.norm(phi))
	amp.append(phi.sum())
	# plt.imshow(np.abs(phi[1:4]).mean(axis=-1).T * 8)
	# plt.show()
	for t in range(8):
		leapfrog(phi)
plt.figure()
plt.plot(amp)
plt.figure()
plt.plot(norm)
plt.show()