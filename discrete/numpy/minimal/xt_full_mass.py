"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = m * phi,
in the full algebra of x+t-
"""
from common import *

edx, idx = ds(0)
edt, idt = dt(1/2)

quad = quadratic((256,))
m = quad[:] / 6

def leapfrog(phi):
	s, x, t, xt = phi
	edt(s,  -(+idx(xt) + m * t))  # t
	edt(x,  +(+edx(t)  + m * xt)) # xt
	idt(t,  +(+idx(x)  + m * s))  # s
	idt(xt, -(+edx(s)  + m * x))  # x

phi = np.random.normal(size=(4, 1)) * np.exp(-quad * 32)
color = lambda phi: np.abs(phi[1:4])
plot_xt(leapfrog, color, phi, 512, 2)
