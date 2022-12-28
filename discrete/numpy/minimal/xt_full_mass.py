"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = m * phi,
in the full algebra of x+t-
"""
import numpy as np
import matplotlib.pyplot as plt

def edt(lhs, rhs, courant=0.33):
	lhs -= rhs * courant
idt = edt

ed = lambda d: lambda a: np.roll(a, shift=-1, axis=d) - a
id = lambda d: lambda a: a - np.roll(a, shift=+1, axis=d)
edx = ed(0)
idx = id(0)

x2 = np.linspace(-1, 1, 256) ** 2
m = x2[:] / 6

def leapfrog(phi):
	s, x, t, xt = phi
	edt(s,  -(+idx(xt) + m * t))  # t
	edt(x,  +(+edx(t)  + m * xt)) # xt
	idt(t,  +(+idx(x)  + m * s))  # s
	idt(xt, -(+edx(s)  + m * x))  # x

phi = np.zeros((4, 256))
phi[...] = np.random.normal(size=(4, 1)) * np.exp(-x2 * 32)
xt = np.empty((4, 512, 256))
for t in range(512):
	xt[..., t, :] = phi
	for _ in range(3):
		leapfrog(phi)

plt.imshow((np.abs(np.moveaxis(xt[1:4], 0, -1))[::-1]))
plt.show()