"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = m * phi,
in the full algebra of x+w+t-, where w is a compact dimension
"""
from common import *

m = quadratic((256,)) / 6
mw = 0.1
(edw, idw), (edx, idx), (edt, idt) = partials(mw, 1, 1/3)

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

phi = np.random.normal(size=(8, 2, 1)) * np.exp(-quadratic((256,), (0.5,)) * 16)
color = lambda phi: np.abs(phi[1:4]).mean(axis=1)
plot_xt(leapfrog, color, phi, 1024, 3)
