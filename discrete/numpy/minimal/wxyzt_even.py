"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = 0,
in the even subalgebra of w+x+y+z+t-, where w is a compact dimension
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

x2 = np.linspace(-1, 1, 32)**2
quad = np.add.outer(np.add.outer(x2, x2), x2)
metric_w = quad

def edt(lhs, rhs, courant=0.25):
	lhs -= rhs * courant
idt = edt

ed = lambda d: lambda a: np.roll(a, shift=-1, axis=d) - a
id = lambda d: lambda a: a - np.roll(a, shift=+1, axis=d)
edx, edy, edz = [ed(d+1) for d in range(3)]
idx, idy, idz = [id(d+1) for d in range(3)]
edw, idw = lambda a: ed(0)(a * metric_w), lambda a: id(0)(a * metric_w)

def leapfrog(phi):
	s, wx, wy, wz, xy, xz, yz, wt, xt, yt, zt, wxyz, wxyt, wxzt, wyzt, xyzt = phi
	edt(s, -(+idw(wt) + idx(xt) + idy(yt) + idz(zt)))  # t
	edt(wx, -(+edw(xt) - edx(wt) + idy(wxyt) + idz(wxzt)))  # wxt
	edt(wy, -(+edw(yt) - idx(wxyt) - edy(wt) + idz(wyzt)))  # wyt
	edt(wz, -(+edw(zt) - idx(wxzt) - idy(wyzt) - edz(wt)))  # wzt
	edt(xy, -(+idw(wxyt) + edx(yt) - edy(xt) + idz(xyzt)))  # xyt
	edt(xz, -(+idw(wxzt) + edx(zt) - idy(xyzt) - edz(xt)))  # xzt
	edt(yz, -(+idw(wyzt) + idx(xyzt) + edy(zt) - edz(yt)))  # yzt
	edt(wxyz, -(+edw(xyzt) - edx(wyzt) + edy(wxzt) - edz(wxyt)))  # wxyzt
	idt(wt, -(+edw(s) - idx(wx) - idy(wy) - idz(wz)))  # w
	idt(xt, -(+idw(wx) + edx(s) - idy(xy) - idz(xz)))  # x
	idt(yt, -(+idw(wy) + idx(xy) + edy(s) - idz(yz)))  # y
	idt(zt, -(+idw(wz) + idx(xz) + idy(yz) + edz(s)))  # z
	idt(wxyt, -(+edw(xy) - edx(wy) + edy(wx) - idz(wxyz)))  # wxy
	idt(wxzt, -(+edw(xz) - edx(wz) + idy(wxyz) + edz(wx)))  # wxz
	idt(wyzt, -(+edw(yz) - idx(wxyz) - edy(wz) + edz(wy)))  # wyz
	idt(xyzt, -(+idw(wxyz) + edx(yz) - edy(xz) + edz(xy)))  # xyz


phi = (np.random.normal(size=(16, 2, 1, 1, 1)) * np.exp(-quad * 16)).astype(np.float32)
phi -= phi.mean(1, keepdims=True)   # filter out lightlike modes
print(phi.size)
color = lambda phi: np.clip(np.abs(phi[1:4, :, 16]).mean(1).T * 4, 0, 1)
im = plt.imshow(color(phi), animated=True, interpolation='bicubic')
def updatefig(*args):
	leapfrog(phi)
	# print(phi.sum())
	im.set_array(color(phi))
	return im,
ani = animation.FuncAnimation(plt.gcf(), updatefig, interval=10, blit=True)
plt.show()
