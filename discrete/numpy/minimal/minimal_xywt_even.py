"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = 0,
in the even subalgebra of x+y+w+t-, where w is a compact dimension
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

def leapfrog(phi):
	s, xy, xw, yw, xt, yt, wt, xywt = phi
	edt(s, +idx(xt) + idy(yt) + idw(wt))  # t
	edt(xy, +edx(yt) - edy(xt) + idw(xywt))  # xyt
	edt(xw, +edx(wt) - idy(xywt) - edw(xt))  # xwt
	edt(yw, +idx(xywt) + edy(wt) - edw(yt))  # ywt
	idt(xt, +edx(s) - idy(xy) - idw(xw))  # x
	idt(yt, +idx(xy) + edy(s) - idw(yw))  # y
	idt(wt, +idx(xw) + idy(yw) + edw(s))  # w
	idt(xywt, +edx(yw) - edy(xw) + edw(xy))  # xyw

phi = np.zeros((8, 64, 64, 2))
x2 = np.linspace(-1, 1, 64) ** 2
phi[..., 0] = np.random.normal(size=(8, 1, 1)) * np.exp(-np.add.outer(x2, x2) * 16)

color = lambda phi: np.clip(np.abs(phi[1:4]).mean(-1).T * 4, 0, 1)
im = plt.imshow(color(phi), animated=True)
def updatefig(*args):
	leapfrog(phi)
	im.set_array(color(phi))
	return im,
ani = animation.FuncAnimation(plt.gcf(), updatefig, interval=10, blit=True)
plt.show()
