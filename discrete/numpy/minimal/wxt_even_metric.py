"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = 0,
where phi is an even grade field of w+x+t-, and w is a compact dimension

"""
from common import *

quad = quadratic((256,))
mw = 0.02
mx = mt = 1 - np.exp(-quad * 3) * (1 - 0.15)
(edw, idw), (edx, idx), (edt, idt) = partials(mw, mx, mt / 2)

def leapfrog(phi):
	# print((phi / mt).sum())   # this is a conserved quantity
	s, wx, wt, xt = phi
	edt(s, -(+idw(wt) + idx(xt)))  # t
	edt(wx, -(+edw(xt) - edx(wt)))  # wxt
	idt(wt, -(+edw(s) - idx(wx)))  # w
	idt(xt, -(+idw(wx) + edx(s)))  # x

phi = (np.random.normal(size=(4, 2, 1)) * np.exp(-quadratic((256,), [0.1]) * 10**2)).astype(np.float64)
filter_lightlike(phi)
# filter_massive(phi)
filter_stationary(leapfrog, phi, 1)
color = lambda phi: np.abs(phi[1:]).mean(1)
show_xt(leapfrog, color, phi, 1024, 10)
