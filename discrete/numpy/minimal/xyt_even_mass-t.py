"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = m * phi * t,
in the even subalgebra of x+y+t-
We note that the discrete scheme appears unconditionally unstable,
with or without attempted pointwise implicit solver,
in various formulations we have tried, fully implicit, semi implicit, or otherwise.

We have not tried all spaces and signatures exhaustively,
and we note that this is known to make a difference to some other zero order terms.

If we constrain the norm of the solution every timestep as a hack to stabilize the scheme,
in spite of its intrinsic tendencies, the qualitative impression is that the t-mass term
produces similar dynamics as the direct mass term for constant mass terms;
not the unique dynamics observed with the spatial-speudoscalar mass term.

Given spatial variable mass terms, the result are quite hopeless
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

if False:
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

x2 = np.linspace(-1, 1, 128) ** 2
quad = np.add.outer(x2, x2) / 2
# m = quad
m = 0.5

def leapfrog(phi):
	s, xy, xt, yt = phi
	edt(s, +idx(xt) + idy(yt), -m, +1)  # t
	edt(xy, +edx(yt) - edy(xt), -m, +1)  # xyt
	idt(xt, +edx(s) - idy(xy), -m, -1)  # x
	idt(yt, +idx(xy) + edy(s), -m, -1)  # y

phi = np.random.normal(size=(4, 1, 1)) * np.exp(-quad * 32)
norm = np.linalg.norm(phi)
color = lambda phi: np.clip((np.abs(phi[1:4])).T * 4, 0, 1)
im = plt.imshow(color(phi), animated=True, interpolation='bilinear')
def updatefig(*args):
	leapfrog(phi)
	# pin the squared norm of the solution
	phi[...] /= np.linalg.norm(phi) / norm
	im.set_array(color(phi))
	return im,
ani = animation.FuncAnimation(plt.gcf(), updatefig, interval=10, blit=True)
plt.show()
