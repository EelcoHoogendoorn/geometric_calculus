"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = m * phi * I = m * phi * xyt,
in the even subalgebra of x+y+t-

Note that the leapfrog scheme is broken;
the update rule is no longer one of spacelike updated in terms of timelike, and vice versa

We note that a zero order term of this form requires interpolation,
to obtain an unbiased update rule.
We only interpolate over the spatial axes here, and temporally,
we simply 'bias towards the past', or use the last available values.

No obvious divergences are observable

Unlike the xt dual mass case, left versus right dualization with I, gives the same pattern of signs
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def edt(lhs, rhs, courant=0.33):
	lhs -= rhs * courant
idt = edt

ed = lambda d: lambda a: np.roll(a, shift=-1, axis=d) - a
id = lambda d: lambda a: a - np.roll(a, shift=+1, axis=d)
edx, edy, edw = [ed(d) for d in range(3)]
idx, idy, idw = [id(d) for d in range(3)]

x2 = np.linspace(-1, 1, 64) ** 2
quad = np.add.outer(x2, x2)
m = quad

def interpolate(phi, axes):
	for i, s in enumerate(axes):
		phi = (phi + np.roll(phi, shift=-s, axis=i)) / 2
	return phi


def leapfrog(phi):
	s, xy, xt, yt = phi
	edt(s, +idx(xt) + idy(yt) - m * interpolate(xy, (-1, -1)))  # t
	edt(xy, +edx(yt) - edy(xt) + m * interpolate(s, (+1, +1)))  # xyt
	idt(xt, +edx(s) - idy(xy) + m * interpolate(yt, (+1, -1)))  # x
	idt(yt, +idx(xy) + edy(s) - m * interpolate(xt, (-1, +1)))  # y

phi = np.random.normal(size=(4, 1, 1)) * np.exp(-quad * 32)
color = lambda phi: np.clip((np.abs(phi[1:4])).T * 4, 0, 1)
im = plt.imshow(color(phi), animated=True, interpolation='bilinear')
def updatefig(*args):
	leapfrog(phi)
	# pin the squared norm of the solution
	# phi[...] /= np.linalg.norm(phi) / norm
	im.set_array(color(phi))
	return im,
ani = animation.FuncAnimation(plt.gcf(), updatefig, interval=10, blit=True)
plt.show()
