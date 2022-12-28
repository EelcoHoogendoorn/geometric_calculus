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
mw = x2[:, None] / 12

ed = lambda d: lambda a: np.roll(a, shift=-1, axis=d) - a
id = lambda d: lambda a: a - np.roll(a, shift=+1, axis=d)
edx, edw = ed(0), lambda a: ed(1)(a * mw)
idx, idw = id(0), lambda a: id(1)(a * mw)

def leapfrog(phi):
	s, wx, wt, xt = phi
	edt(s,   -(+idw(wt)+idx(xt))) # t
	edt(wx,  -(+edw(xt)-edx(wt))) # wxt
	idt(wt,  -(+edw(s)-idx(wx))) # w
	idt(xt,  -(+idw(wx)+edx(s))) # x

phi = np.zeros((4, 256, 2))
phi[..., 0] = np.random.normal(size=(4, 1)) * np.exp(-x2 * 32)
# phi -= phi.mean(axis=2, keepdims=True)
xt = np.empty((4, 256, 512, 2))
for t in range(512):
	xt[..., t, :] = phi
	for _ in range(3):
		leapfrog(phi)

plt.imshow((np.abs(xt[1:4, :, ::-1]).mean(axis=-1).T * 2))
plt.show()