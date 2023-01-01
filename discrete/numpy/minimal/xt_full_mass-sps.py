"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = m * phi * It = m * phi * x,
in the full algebra of x+t-

We note that the resulting method does not appear to diverge
However, dualizing from the other side, as x * phi rather than phi * x,
does appear to diverge.
"""
from common import *

edx, idx = ds(0)
edt, idt = dt(1/2)

quad = quadratic((256,))
m = quad[:] / 6

def leapfrog(phi):
	s, x, t, xt = phi
	edt(s, -(+idx(xt) - m * interpolate(xt, +1)))  # t
	edt(x, +(+edx(t) - m * interpolate(t, -1)))  # xt
	idt(t, +(+idx(x) + m * interpolate(x, +1)))  # s
	idt(xt, -(+edx(s) + m * interpolate(s, -1)))  # x

phi = np.random.normal(size=(4, 1)) * np.exp(-quad * 32)
color = lambda phi: np.abs(phi[1:4])
plot_xt(leapfrog, color, phi, 512, 2)
