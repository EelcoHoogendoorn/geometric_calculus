

import jax.numpy as jnp
import jax
from optax import huber_loss

from continuous.models import make_field_model
from continuous.geometry import Geometry
from continuous.domain import *
from continuous.optimize import optimize


def test_image_fit():
	"""test that we can reconstruct an rgb image; example without gradients"""
	import imageio.v3 as iio

	im = iio.imread('imageio:chelsea.png')
	gray = im.mean(axis=-1, keepdims=True)
	gray = jnp.array(gray) / 128 - 1
	print(gray.shape)

	geometry = Geometry(2)
	domain = UnitCube(geometry)
	model, params = make_field_model(
		geometry,
		inputs=geometry.domain,
		outputs=geometry.algebra.subspace.scalar(),
		n_frequencies=128,
		n_hidden=[128]*2,
		scale=3e1,
	)
	plot_2d_0form(domain, model(params), n=512)


	shape = jnp.array(gray.shape[:-1]) / 2 - 1
	def image_loss(image, x):
		i, j = ((x + 1) * shape).astype(jnp.int_)
		return huber_loss(image(x) - gray[i, j], delta=1e-2)

	objectives = [
		(image_loss, domain.sample_interior, 1024, 1e+1),
	]
	params = optimize(model, params, objectives, n_steps=301)

	# visualize solution
	plot_2d_0form(domain, model(params), 512)


def test_multi_output():
	"""test that we can reconstruct an rgb image; example without gradients"""

	geometry = Geometry(2)
	domain = UnitCube(geometry)
	model, params = make_field_model(
		geometry,
		inputs=geometry.domain,
		outputs={
			'f0': geometry.algebra.subspace.scalar(),
			'f1': geometry.algebra.subspace.vector(),
		},
		n_frequencies=128,
		n_hidden=[128]*2,
		scale=3e1,
	)
	import matplotlib.pyplot as plt
	plt.figure()
	print(model(params)['f0'](jnp.zeros(2)).shape)
	plot_2d_0form(domain, model(params)['f0'], n=512)
	plt.figure()
	print(model(params)['f1'](jnp.zeros(2)).shape)
	plot_2d_1form_grid(domain, model(params)['f1'], n=64)
	plt.show()
