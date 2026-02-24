"""Minimal self contained numpy example of the leapfrog scheme
for the equation geometric_derivative(phi) = 0,
where phi is an even grade field of w+x+y+t-, and w is a compact dimension

This is a setup where the compact dimension is held constant,
and the non-compact metrics have a strong distortion applied to them,
whereby clocks inside the potential well, run at 20% of the speed of those outside.
Note that one may trap the wave function to an arbitrary degree,
no matter how low the mass term, by further increases to the steepness of the metric distortion.

Note that our 20% compares to about 70% clock speed for the surface of a neutron star;
so this setup is going well beyond those values.
At the neutron star values, lightlike rays are still qualitatively very lightlike,
and easily escape merely with some redshift.
But there appears to be somewhat of a phase transition around the 30% value,
where the majority of the wavefunction remains in a bound state in the potential,
and one might say they are 'within the Schwarzschild radius' of the metric distortion applied here,
despite the fact that its a distortion without a singularity, not a Schwarzschild metric.
"""
from common import *

quad = quadratic((256, 256))
mw = 0.02#quad / 4
mx = my = mt = 1 - np.exp(-quad * 3) * (1 - 0.25)
# mx = my = mt = np.exp(-quad)
(edw, idw), (edx, idx), (edy, idy), (edt, idt) = partials(mw, mx, my, mt / 2 / 2)

def leapfrog(phi):
	# print((phi / mt).sum())   # this is a conserved quantity
	s, wx, wy, xy, wt, xt, yt, wxyt = phi
	edt(s, -(+idw(wt) + idx(xt) + idy(yt)))  # t
	edt(wx, -(+edw(xt) - edx(wt) + idy(wxyt)))  # wxt
	edt(wy, -(+edw(yt) - idx(wxyt) - edy(wt)))  # wyt
	edt(xy, -(+idw(wxyt) + edx(yt) - edy(xt)))  # xyt
	idt(wt, -(+edw(s) - idx(wx) - idy(wy)))  # w
	idt(xt, -(+idw(wx) + edx(s) - idy(xy)))  # x
	idt(yt, -(+idw(wy) + idx(xy) + edy(s)))  # y
	idt(wxyt, -(+edw(xy) - edx(wy) + edy(wx)))  # wxy

phi = (np.random.normal(size=(8, 2, 1, 1)) * np.exp(-quadratic((256, 256), [0.0, 0.0]) * 10**2)).astype(np.float32)
# filter_lightlike(phi)
filter_massive(phi)
filter_stationary(leapfrog, phi, 1)
show_xt(leapfrog, lambda phi: np.abs((phi)[[3, 5, 6]]).mean(1)[..., 128], phi, 512, 10 * 2)
color = lambda phi: np.abs(phi[[3, 5, 6]]).mean(1)
show_animation(leapfrog, color, phi, 10, scale=0.5)
