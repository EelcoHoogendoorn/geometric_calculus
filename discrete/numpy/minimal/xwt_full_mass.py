"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = m * phi,
in the full algebra of x+w+t-, where w is a compact dimension
"""
from common import *

x = np.linspace(-1, 1, 256)
x2 = x ** 2
m = x2 / 6
mw = 0.1

edw, idw = ds(0, mw)
edx, idx = ds(1)
edt, idt = dt(1/3)

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

phi = np.random.normal(size=(8, 2, 1)) * np.exp(-(x-0.5)**2 * 16)
color = lambda phi: np.abs(phi[1:4]).mean(axis=1)
plot_xt(leapfrog, color, phi, 1024, 3)
