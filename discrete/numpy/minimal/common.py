"""This goes a little against the self contained philosophy...
but lets not have the perfect be the enemy of the good
The only thing that is not boilerplate, and which needs to be
appreciated to understand the schemes constructed, are the first dozen lines,
constructing the partial derivative operators.
"""
import numpy as np


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


def interpolate(phi, *axes):
	"""spatial interpolation of field values
	when interpolating a field that does not extend along a given axis, like a scalar field,
	'axes' should recieve a negative value, to obtain the correct shift
	"""
	for i, s in enumerate(axes):
		phi = (phi + np.roll(phi, shift=s, axis=i)) / 2
	return phi


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


def rollout(leapfrog, color, phi, steps, unroll):
	def generate(phi):
		for t in range(steps):
			yield color(phi).T
			for _ in range(unroll):
				leapfrog(phi)
	return np.array(list(generate(phi.copy())))

def show_animation(leapfrog, color, phi, unroll=1, scale=1.3):
	"""animate the given leapfrog scheme"""
	c = color(phi)
	cmax = np.percentile(c, 98) / scale
	color_scaled = lambda phi: np.clip(color(phi) / cmax, 0, 1).T

	import matplotlib.pyplot as plt
	import matplotlib.animation as animation
	im = plt.imshow(color_scaled(phi), animated=True)#, interpolation='bilinear')
	plt.axis('off')
	def updatefig(*args):
		for _ in range(unroll):
			leapfrog(phi)
		im.set_array(color_scaled(phi))
		return im,

	ani = animation.FuncAnimation(plt.gcf(), updatefig, interval=10, blit=True)
	plt.show()

def show_xt(leapfrog, color, phi, steps, unroll):
	"""Space-time plot of evolving wave equation"""
	img = rollout(leapfrog, color, phi, steps, unroll)[::-1]
	import matplotlib.pyplot as plt
	plt.imshow(np.clip(img / np.percentile(img, 99), 0, 1))
	plt.show()

def save_animation(path, leapfrog, color, phi, steps, unroll=1, scale=1.0):
	img = rollout(leapfrog, color, phi, steps, unroll)
	cmax = np.percentile(img, 98) / scale
	img = np.clip((img / cmax) * 255, 0, 255).astype(np.uint8)
	import imageio.v3 as iio
	iio.imwrite(path, img, loop=0)

def save_xt(path, leapfrog, color, phi, steps, unroll=1, scale=1.0):
	img = rollout(leapfrog, color, phi, steps, unroll)[::-1]
	cmax = np.percentile(img, 98) / scale
	img = np.clip((img / cmax) * 255, 0, 255).astype(np.uint8)
	import imageio.v3 as iio
	iio.imwrite(path, img)