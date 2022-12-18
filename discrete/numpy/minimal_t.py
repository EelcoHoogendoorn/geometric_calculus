"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = m * phi * t,
in the even subalgebra of x+y+t-
We note that the discrete scheme appears unconditionally unstable,
with or without attempted pointwise implicit solver
"""
import numpy as np
import matplotlib.pyplot as plt

if True:
	def edt(lhs, rhs, m, ts, courant=0.33):
		"""
		Solve implicit timestepping scheme, with mass term involving an average of present and future states
		(phi_n - phi_o) * dt = rhs + m * (phi_n + phi_o)
		phi_n * (dt-m) - phi_o * (dt+m) = rhs
		phi_n = (phi_o * (dt+m) + rhs) / (dt-m)
		"""
		dt = 1/courant * ts
		lhs[...] = (lhs * (dt+m) + rhs) / (dt-m)
else:
	def edt(lhs, rhs, m, _, courant=0.33):
		lhs -= (rhs + (m * lhs)) * courant
idt = edt

ed = lambda d: lambda a: np.roll(a, shift=-1, axis=d) - a
id = lambda d: lambda a: a - np.roll(a, shift=+1, axis=d)
edx, edy, edw = [ed(d) for d in range(3)]
idx, idy, idw = [id(d) for d in range(3)]

m = 0.1
def leapfrog(phi):
	s, xy, xt, yt = phi
	edt(s, +idx(xt) + idy(yt), m, +1)  # t
	edt(xy, +edx(yt) - edy(xt), m, +1)  # xyt
	idt(xt, +edx(s) - idy(xy), -m, -1)  # x
	idt(yt, +idx(xy) + edy(s), -m, -1)  # y

x2 = np.linspace(-1, 1, 64) ** 2
phi = np.random.normal(size=(4, 1, 1)) * np.exp(-np.add.outer(x2, x2) * 16)
for i in range(4):
	plt.imshow(np.abs(phi[1:4]).T * 8)
	plt.show()
	for t in range(64):
		leapfrog(phi)
