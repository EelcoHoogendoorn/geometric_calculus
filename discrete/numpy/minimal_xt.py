"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = m * phi,
in the full algebra of x+w+t-, where w is a compact dimension
"""
import numpy as np
import matplotlib.pyplot as plt

def edt(lhs, rhs, courant=0.33):
	lhs -= rhs * courant
idt = edt

ed = lambda d: lambda a: np.roll(a, shift=-1, axis=d) - a
id = lambda d: lambda a: a - np.roll(a, shift=+1, axis=d)
edx, _edw = [ed(d) for d in range(2)]
idx, _idw = [id(d) for d in range(2)]
edw = lambda a: _edw(a) * 0.1
idw = lambda a: _idw(a) * 0.1

x2 = np.linspace(-1, 1, 64) ** 2
m = x2[:, None] * 0.3

def leapfrog(phi):
	s, w, x, t, wx, wt, xt, wxt = phi
	edt(s,   -(+idw(wt)+idx(xt)	+m*t)) # t
	edt(w,   +(+edw(t)-idx(wxt)	+m*wt)) # wt
	edt(x,   +(+idw(wxt)+edx(t)	+m*xt)) # xt
	edt(wx,  -(+edw(xt)-edx(wt)	+m*wxt)) # wxt
	idt(t,   +(+idw(w)+idx(x)	+m*s)) # s
	idt(wt,  -(+edw(s)-idx(wx)	+m*w)) # w
	idt(xt,  -(+idw(wx)+edx(s)	+m*x)) # x
	idt(wxt, +(+edw(x)-edx(w)	+m*wx)) # wx

phi = np.zeros((8, 64, 2))
phi[..., 0] = np.random.normal(size=(8, 1)) * np.exp(-x2 * 16)
xt = np.empty((8, 64, 64, 2))
for t in range(64):
	xt[:, :, t] = phi
	for _ in range(3):
		leapfrog(phi)

plt.imshow((np.abs(xt[1:4,:, ::-1]).mean(axis=-1).T * 4))
plt.show()
