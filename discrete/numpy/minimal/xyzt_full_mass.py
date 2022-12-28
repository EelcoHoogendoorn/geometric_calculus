"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = m * phi,
in the full algebra of x+y+z+t-
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def edt(lhs, rhs, courant=0.33):
	lhs -= rhs * courant
idt = edt
ed = lambda d: lambda a: np.roll(a, shift=-1, axis=d) - a
id = lambda d: lambda a: a - np.roll(a, shift=+1, axis=d)
edx, edy, edz = [ed(a) for a in range(3)]
idx, idy, idz = [id(a) for a in range(3)]

x2 = np.linspace(-1, 1, 64)**2
quad = np.add.outer(np.add.outer(x2, x2), x2)
m = quad

def leapfrog(phi):
	s,x,y,z,t,xy,xz,yz,xt,yt,zt,xyz,xyt,xzt,yzt,xyzt = phi
	edt(s, -(+idx(xt)+idy(yt)+idz(zt) + m*t))	 # t
	edt(x, +(+edx(t)-idy(xyt)-idz(xzt) + m*xt))	 # xt
	edt(y, +(+idx(xyt)+edy(t)-idz(yzt) + m*yt))	 # yt
	edt(z, +(+idx(xzt)+idy(yzt)+edz(t) + m*zt))	 # zt
	edt(xy, -(+edx(yt)-edy(xt)+idz(xyzt) + m*xyt))	 # xyt
	edt(xz, -(+edx(zt)-idy(xyzt)-edz(xt) + m*xzt))	 # xzt
	edt(yz, -(+idx(xyzt)+edy(zt)-edz(yt) + m*yzt))	 # yzt
	edt(xyz, +(+edx(yzt)-edy(xzt)+edz(xyt) + m*xyzt))	 # xyzt
	idt(t, +(+idx(x)+idy(y)+idz(z) + m*s))	 # s
	idt(xt, -(+edx(s)-idy(xy)-idz(xz) + m*x))	 # x
	idt(yt, -(+idx(xy)+edy(s)-idz(yz) + m*y))	 # y
	idt(zt, -(+idx(xz)+idy(yz)+edz(s) + m*z))	 # z
	idt(xyt, +(+edx(y)-edy(x)+idz(xyz) + m*xy))	 # xy
	idt(xzt, +(+edx(z)-idy(xyz)-edz(x) + m*xz))	 # xz
	idt(yzt, +(+idx(xyz)+edy(z)-edz(y) + m*yz))	 # yz
	idt(xyzt, -(+edx(yz)-edy(xz)+edz(xy) + m*xyz))	 # xyz

phi = (np.random.normal(size=(16, 1, 1, 1)) * np.exp(-quad * 9)).astype(np.float32)
print(phi.size)
color = lambda phi: np.clip((np.abs(phi[-4:-1, 32])).T * 4, 0, 1)
im = plt.imshow(color(phi), animated=True, interpolation='bilinear')
def updatefig(*args):
	leapfrog(phi)
	im.set_array(color(phi))
	return im,
ani = animation.FuncAnimation(plt.gcf(), updatefig, interval=10, blit=True)
plt.show()
