"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = m * phi,
in the full algebra of x+y+t-
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def edt(lhs, rhs, courant=0.5):
	lhs -= rhs * courant
idt = edt
ed = lambda d: lambda a: np.roll(a, shift=-1, axis=d) - a
id = lambda d: lambda a: a - np.roll(a, shift=+1, axis=d)
edx, edy = [ed(a) for a in range(2)]
idx, idy = [id(a) for a in range(2)]

x2 = np.linspace(-1, 1, 64) ** 2
m = (1 - np.exp(-np.add.outer(x2, x2) * 3)) * 0.6

def leapfrog(phi):
	s, x, y, t, xy, xt, yt, xyt = phi
	edt(s,   -(+idx(xt)  +idy(yt)  +m*t))
	edt(x,   +(+edx(t)   -idy(xyt) +m*xt))
	edt(y,   +(+idx(xyt) +edy(t)   +m*yt))
	edt(xy,  -(+edx(yt)  -edy(xt)  +m*xyt))
	idt(t,   +(+idx(x)   +idy(y)   +m*s))
	idt(xt,  -(+edx(s)   -idy(xy)  +m*x))
	idt(yt,  -(+idx(xy)  +edy(s)   +m*y))
	idt(xyt, +(+edx(y)   -edy(x)   +m*xy))

phi = np.random.normal(size=(8, 1, 1)) * np.exp(-np.add.outer(x2, x2) * 16)

color = lambda phi: np.clip((np.abs(phi[4:7])).T * 4, 0, 1)
im = plt.imshow(color(phi), animated=True, interpolation='bilinear')
def updatefig(*args):
	leapfrog(phi)
	im.set_array(color(phi))
	return im,
ani = animation.FuncAnimation(plt.gcf(), updatefig, interval=10, blit=True)
plt.show()
