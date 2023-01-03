"""This goes a little against the mimimal self contained philosophy...
but lets not have the perfect be the enemy of the good"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def dt(metric):
	"""Construct exterior and interior temporal derivative operators"""
	def inner(lhs, rhs):
		lhs += rhs * metric
	return inner, inner
def ds(d, m=None):
	"""Construct exterior and interior spatial derivative operators along an axis d, and optional metric m"""
	ed = lambda a: np.roll(a, shift=-1, axis=d) - a
	id = lambda a: a - np.roll(a, shift=+1, axis=d)
	if m is None or np.allclose(m, 1):
		return ed, id
	else:
		return lambda a: ed(a*m), lambda a: id(a*m)
def partials(*metric):
	"""Construct spatio-temporal partial derivative operators"""
	*ms, mt = metric
	return [ds(i, m) for i, m in enumerate(ms)] + [dt(mt)]


def meshgrid(shape):
	return np.array(np.meshgrid([np.linspace(-1, 1, s) for s in shape], indexing='ij'))
def quadratic(shape, loc=None):
	quad = 0
	loc = [0] * len(shape) if loc is None else loc
	for s, l in zip(shape, loc):
		x2 = (np.linspace(-1, 1, s) - l) ** 2
		quad = np.add.outer(quad, x2)
	return quad


def filter_stationary(leapfrog, phi, repeats=1):
	"""remove field components that are stationary under the action of the leapfrog scheme"""
	for _ in range(repeats):
		old = phi.copy()
		leapfrog(phi)
		phi -= old

def filter_lightlike(phi, axis=0):
	"""filter out lightlike modes"""
	phi -= phi.mean(axis=axis+1, keepdims=True)

def filter_massive(phi, axis=0):
	"""filter out massive modes"""
	old = phi.copy()
	filter_lightlike(old, axis = axis)
	phi -= old


def interpolate(phi, *axes):
	"""spatial interpolation of field values
	when interpolating a field that does not extend along a given axis, like a scalar field,
	'axes' should recieve a negative value, to obtain the correct shift
	"""
	for i, s in enumerate(axes):
		phi = (phi + np.roll(phi, shift=s, axis=i)) / 2
	return phi


def animate(leapfrog, color, phi, unroll=1, scale=1.3):
	"""animate the given leapfrog scheme"""
	c = color(phi)
	cmax = np.percentile(c, 98) / scale
	color_scaled = lambda phi: np.clip(color(phi) / cmax, 0, 1).T

	im = plt.imshow(color_scaled(phi), animated=True, interpolation='bilinear')
	plt.axis('off')
	def updatefig(*args):
		for _ in range(unroll):
			leapfrog(phi)
		im.set_array(color_scaled(phi))
		return im,

	ani = animation.FuncAnimation(plt.gcf(), updatefig, interval=10, blit=True)
	plt.show()


def plot_xt(leapfrog, color, phi, steps, unroll):
	"""Space-time plot of evolving wave equation"""
	r = []
	for t in range(steps):
		r.append(color(phi).T)
		for _ in range(unroll):
			leapfrog(phi)
	arr = np.array(r[::-1])
	plt.imshow(np.clip(arr / np.percentile(arr, 99), 0, 1))
	plt.show()