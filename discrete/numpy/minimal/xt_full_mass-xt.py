"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = m * phi * I,
in the full algebra of x+t-

We note that the resulting method does not appear to diverge
However, dualizing from the other side, as I * phi rather than phi * I,
does appear to diverge.
"""
from common import *

quad = quadratic((256,))
m = quad / 6
(edx, idx), (edt, idt) = partials(1, 1/2)

def leapfrog(phi):
	s, x, t, xt = phi
	edt(s,  -(+idx(xt) + m * interpolate(x, +1)))  # t
	edt(x,  +(+edx(t)  + m * interpolate(s, -1))) # xt
	idt(t,  +(+idx(x)  + m * interpolate(xt, +1)))  # s
	idt(xt, -(+edx(s)  + m * interpolate(t, -1)))  # x

phi = np.random.normal(size=(4, 1)) * np.exp(-quad * 32)
color = lambda phi: np.abs(phi[1:4])
show_xt(leapfrog, color, phi, 512, 2)
