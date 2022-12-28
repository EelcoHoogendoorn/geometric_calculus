"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = m * phi,
in the full algebra of x+w+t-, where w is a compact dimension
"""
import numpy as np
import matplotlib.pyplot as plt

def edt(lhs, rhs, courant=0.33):
	lhs -= rhs * courant
idt = edt
x2 = np.linspace(-1, 1, 256) ** 2
mv = x2 / 12
mw = x2 / 6
# mw = .2

ed = lambda d: lambda a: np.roll(a, shift=-1, axis=d) - a
id = lambda d: lambda a: a - np.roll(a, shift=+1, axis=d)
edx = ed(2)
idx = id(2)
edv = lambda a: id(0)(a * mv)
idv = lambda a: ed(0)(a * mv)
edw = lambda a: id(1)(a * mw)
idw = lambda a: ed(1)(a * mw)

def leapfrog(phi):
	s, vw, vx, wx, vt, wt, xt, vwxt = phi
	edt(s, -(+idv(vt) + idw(wt) + idx(xt)))  # t
	edt(vw, -(+edv(wt) - edw(vt) + idx(vwxt)))  # vwt
	edt(vx, -(+edv(xt) - idw(vwxt) - edx(vt)))  # vxt
	edt(wx, -(+idv(vwxt) + edw(xt) - edx(wt)))  # wxt
	idt(vt, -(+edv(s) - idw(vw) - idx(vx)))  # v
	idt(wt, -(+idv(vw) + edw(s) - idx(wx)))  # w
	idt(xt, -(+idv(vx) + idw(wx) + edx(s)))  # x
	idt(vwxt, -(+edv(wx) - edw(vx) + edx(vw)))  # vwx

phi = np.zeros((8, 2, 2, 256))
phi = np.random.normal(size=(8, 2, 2, 1)) * np.exp(-x2 * 32)
phi -= phi.mean(axis=(1,2), keepdims=True)
# phi -= phi.mean(axis=2, keepdims=True)
xt = np.empty((512, 8, 2, 2, 256))
for t in range(512):
	xt[t] = phi
	for _ in range(3):
		leapfrog(phi)

plt.imshow((np.transpose(np.abs(xt[::-1, 1:4]).mean(axis=(2, 3)), [0, 2, 1]) * 2))
plt.show()