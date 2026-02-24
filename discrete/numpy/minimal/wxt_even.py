"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = m * phi,
in the full algebra of x+w+t-, where w is a compact dimension
"""
from common import *

quad = quadratic((256,))
(edw, idw), (edx, idx), (edt, idt) = partials(1, 1, 1/3)

def leapfrog(phi):
	s, wx, wt, xt = phi
	edt(s,   -(+idw(wt)+idx(xt))) # t
	edt(wx,  -(+edw(xt)-edx(wt))) # wxt
	idt(wt,  -(+edw(s)-idx(wx))) # w
	idt(xt,  -(+idw(wx)+edx(s))) # x

phi = np.random.normal(size=(4, 2, 1)) * np.exp(-quad * 32)
color = lambda phi: np.abs(phi[1:]).mean(1)
show_xt(leapfrog, color, phi, 512, 3)