"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = m * phi,
in the full algebra of x+w+t-, where w is a compact dimension
"""
from common import *

quad = quadratic((256,))
metric_w = quad / 4
metric_q = 1 - np.exp(-quad * 4) / 4

edv, idv = ds(0, quad / 12)
edw, idw = ds(1, quad / 6)
edx, idx = ds(2)
edt, idt = dt(1 / 3)

def leapfrog(phi):
	s, vw, vx, wx, vt, wt, xt, vwxt = phi
	edt(s, -(+idv(vt) + idw(wt) + idx(xt)))  # t
	edt(vw, -(+edv(wt) - edw(vt) + idx(vwxt)))  # vwt
	edt(vx, -(+edv(xt) - idw(vwxt) - edx(vt)))  # vxt
	edt(wx, -(+idv(vwxt) + edw(xt) - edx(wt)))  # wxt
	idt(vt, -(+edv(s) - idw(vw) - idx(vx)))  # v
	idt(wt, -(+idv(vw) + edw(s) - idx(wx)))  # w
	idt(xt, -(+idv(vx) + idw(wx) + edx(s)))  # x
	idt(vwxt, -(+edv(wx) - edw(vx) + edx(vw)))  # vwx

phi = np.random.normal(size=(8, 2, 2, 1)) * np.exp(-quad * 32)
color = lambda phi: np.abs(phi[1:4]).mean(axis=(1, 2))
plot_xt(leapfrog, color, phi, 512, 3)
