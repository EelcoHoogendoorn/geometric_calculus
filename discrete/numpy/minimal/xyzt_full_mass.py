"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = m * phi,
in the full algebra of x+y+z+t-
"""
from common import *

quad = quadratic((64, 64, 64))
m = quad

edx, idx = ds(0)
edy, idy = ds(1)
edz, idz = ds(2)
edt, idt = dt(1/3)

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

phi = np.random.normal(size=(16, 1, 1, 1)) * np.exp(-quad * 9)
filter_stationary(leapfrog, phi)
color = lambda phi: np.abs(phi[1:4, 32])
animate(leapfrog, color, phi)
