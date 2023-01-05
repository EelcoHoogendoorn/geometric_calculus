"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = 0,
where phi is an even grade field of v+w+x+y+t-, and v and w are compact dimensions

This configuration produces some of its own interesting dynamics
"""
from common import *

quad = quadratic((64, 64))
mv = 0.4
mw = 0.+quad / 2
(edv, idv), (edw, idw), (edx, idx), (edy, idy), (edt, idt) = partials(mv, mw, 1, 1, 1/3)
quad = quadratic((64, 64), [0.3, 0.3])

def leapfrog(phi):
	s, vw, vx, wx, vy, wy, xy, vt, wt, xt, yt, vwxy, vwxt, vwyt, vxyt, wxyt = phi
	edt(s, -(+idv(vt) + idw(wt) + idx(xt) + idy(yt)))  # t
	edt(vw, -(+edv(wt) - edw(vt) + idx(vwxt) + idy(vwyt)))  # vwt
	edt(vx, -(+edv(xt) - idw(vwxt) - edx(vt) + idy(vxyt)))  # vxt
	edt(wx, -(+idv(vwxt) + edw(xt) - edx(wt) + idy(wxyt)))  # wxt
	edt(vy, -(+edv(yt) - idw(vwyt) - idx(vxyt) - edy(vt)))  # vyt
	edt(wy, -(+idv(vwyt) + edw(yt) - idx(wxyt) - edy(wt)))  # wyt
	edt(xy, -(+idv(vxyt) + idw(wxyt) + edx(yt) - edy(xt)))  # xyt
	edt(vwxy, -(+edv(wxyt) - edw(vxyt) + edx(vwyt) - edy(vwxt)))  # vwxyt
	idt(vt, -(+edv(s) - idw(vw) - idx(vx) - idy(vy)))  # v
	idt(wt, -(+idv(vw) + edw(s) - idx(wx) - idy(wy)))  # w
	idt(xt, -(+idv(vx) + idw(wx) + edx(s) - idy(xy)))  # x
	idt(yt, -(+idv(vy) + idw(wy) + idx(xy) + edy(s)))  # y
	idt(vwxt, -(+edv(wx) - edw(vx) + edx(vw) - idy(vwxy)))  # vwx
	idt(vwyt, -(+edv(wy) - edw(vy) + idx(vwxy) + edy(vw)))  # vwy
	idt(vxyt, -(+edv(xy) - idw(vwxy) - edx(vy) + edy(vx)))  # vxy
	idt(wxyt, -(+idv(vwxy) + edw(xy) - edx(wy) + edy(wx)))  # wxy

phi = (np.random.normal(size=(16, 2, 2, 1, 1)) * np.exp(-quad * 16)).astype(np.float32)
filter_lightlike(phi, 0)
filter_lightlike(phi, 1)
filter_stationary(leapfrog, phi, 3)
color = lambda phi: np.abs(phi[[6, 8, 9]]).mean((1, 2))
show_animation(leapfrog, color, phi)
