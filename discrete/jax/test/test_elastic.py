import jax
import jax.numpy as jnp
import numpy as np

from discrete.jax.field import Field
from numga.algebra.algebra import Algebra


def test_scalar_wave_1d():
	# FIXME: set up variable material property fields
	#  would that help with funny linear propagation? why so lnear now?
	algebra = Algebra.from_str('x+')
	shape = (64,)
	displacement = Field.from_subspace(algebra.subspace.scalar(), shape)
	velocity = displacement * 0
	x = displacement.meshgrid()
	x2 = (x ** 2).sum(axis=0, keepdims=False)
	gauss = jnp.exp(-x2*10*2)
	displacement.arr = displacement.arr.at[0].set(gauss)
	lame = Field.from_subspace(algebra.subspace.vector(), shape)
	window = jnp.all(jnp.abs(x) < 0.5, axis=0) + 1e-2
	lame.arr = lame.arr.at[0].set(window)
	# density = Field.from_subspace(algebra.subspace.vector(), shape)
	# density = density.arr.at[0].set(window)
	# density = density.arr.at[1].set(window)
	# print(field.arr.shape)
	# return
	@jax.jit
	def step(displacement, velocity):
		grad = displacement.geometric_derivative()
		force = (grad*lame).geometric_derivative(displacement.subspace)
		velocity = velocity + force * 0.33
		return displacement + velocity, velocity


	frames = []
	for i in range(90):
		displacement, velocity = step(displacement, velocity)
		frames.append(displacement.arr.T)
		print(jnp.linalg.norm(velocity.arr))

	import matplotlib.pyplot as plt

	path = r'../../output/wave_scalar_1d_0'
	import imageio.v3 as iio
	import os
	os.makedirs(path, exist_ok=True)
	frames = jnp.array(frames)
	frames = jnp.abs(frames)
	print(frames.shape)
	# frames = frames / 0.15  # arr.max()
	frames = frames / jnp.percentile(frames.flatten(), 95)
	frames = jnp.clip(frames * 255, 0, 255).astype(jnp.uint8)
	frames = np.array(frames)
	print(frames.shape, type(frames))
	iio.imwrite(os.path.join(path, 'anim.gif'), frames[..., [0, 0, 0]])
	return
	fig, ax = plt.subplots()
	ax.imshow(animation)
	if False:
		fig, ax = plt.subplots(2, 1, squeeze=False)
		for i in range(1):
			ax[0, i].plot(grad.arr[i])
			ax[1, i].plot(force.arr[i])
			# ax[i].colorbar()
	plt.show()


def test_scalar_wave_2d():
	path = r'../../output/wave_scalar_2d'
	import imageio.v3 as iio

	from numga.algebra.algebra import Algebra
	algebra = Algebra.from_str('x+y+')
	shape = (128, 128)
	displacement = Field.from_subspace(algebra.subspace.scalar(), shape)
	velocity = displacement * 0
	x = displacement.meshgrid() * np.array([1,2])[:, None, None]
	x2 = (x ** 2).sum(axis=0, keepdims=False)
	gauss = jnp.exp(-x2*10*2)
	displacement.arr = displacement.arr.at[0].set(gauss)
	lame = Field.from_subspace(algebra.subspace.vector(), shape)
	window = jnp.all(jnp.abs(x) < 0.7, axis=0) + 1e-2
	lame.arr = lame.arr.at[0].set(window)
	lame.arr = lame.arr.at[1].set(window)
	# density = Field.from_subspace(algebra.subspace.vector(), shape)
	# density = density.arr.at[0].set(window)
	# density = density.arr.at[1].set(window)
	# print(field.arr.shape)
	# return
	@jax.jit
	def step(displacement, velocity, unroll=3):
		def inner(_, state):
			displacement, velocity = state
			grad = displacement.geometric_derivative()
			force = (grad * lame).geometric_derivative(displacement.subspace)
			velocity = velocity + force * 0.33
			return displacement + velocity, velocity
		return jax.lax.fori_loop(0, unroll, inner, (displacement, velocity))

	frames = []
	for i in range(200):
		displacement, velocity = step(displacement, velocity, unroll=3)
		frames.append((displacement.arr * window).T)


	import os
	os.makedirs(path, exist_ok=True)
	frames = jnp.array(frames)
	frames = jnp.abs(frames)
	# frames = frames / 0.15  # arr.max()
	frames = frames / jnp.percentile(frames.flatten(), 95)
	frames = jnp.clip(frames * 255, 0, 255).astype(jnp.uint8)
	frames = np.array(frames)
	print(frames.shape, type(frames))
	iio.imwrite(os.path.join(path, 'anim.gif'), frames[..., [0, 0, 0]])

	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(2, 2)
	if False:
		fig, ax = plt.subplots(2, 2)
		for i in range(2):
			ax[0, i].imshow(grad.arr[i], origin='lower')
			ax[1, i].imshow(force.arr[i], origin='lower')
			# ax[i].colorbar()
	# plt.show()


def test_elastic_wave():
	from numga.algebra.algebra import Algebra
	algebra = Algebra.from_str('x+y+')
	shape = (128,128)
	displacement = Field.from_subspace(algebra.subspace.vector(), shape)
	velocity = displacement * 0
	x = displacement.meshgrid()
	x2 = (x ** 2).sum(axis=0, keepdims=False)
	gauss = jnp.exp(-x2*10)
	gauss = jnp.roll(gauss, 50, 0)
	displacement.arr = displacement.arr.at[0].set(gauss)
	lame = Field.from_subspace(algebra.subspace.even_grade(), shape)
	window = jnp.all(jnp.abs(x) < 0.8, axis=0) + 1e-2
	lame.arr = lame.arr.at[0].set(window)
	lame.arr = lame.arr.at[1].set(window / 2)
	# density = Field.from_subspace(algebra.subspace.vector(), shape)
	# density = density.arr.at[0].set(window)
	# density = density.arr.at[1].set(window)
	# print(field.arr.shape)
	# return
	@jax.jit
	def step(displacement, velocity, unroll):
		def inner(_, state):
			d, v = state
			curl_and_div = d.geometric_derivative()
			force = (curl_and_div * lame).geometric_derivative(d.subspace)
			v = v + force * 0.33
			return d + v, v
		return jax.lax.fori_loop(0, unroll, inner, (displacement, velocity))


	image = window

	coords = np.array(np.meshgrid(*[np.arange(s) for s in image.shape], indexing='ij'))
	image = image * np.bitwise_xor(*(coords // 4) % 2)

	import scipy.ndimage

	def warp(image, field):
		c = coords + field
		cols = scipy.ndimage.map_coordinates(image, c.reshape(2, -1))
		return cols.reshape(image.shape)

	frames = []
	for i in range(200):
		displacement, velocity = step(displacement, velocity, unroll=13)
		# frames.append((displacement.arr * window).T)
		q = warp(image, displacement.arr * window * 20)
		print(q.shape)
		frames.append(q[..., None])

	path = r'../../output/wave_elastic_2d'
	import imageio.v3 as iio
	import os
	os.makedirs(path, exist_ok=True)
	frames = jnp.array(frames)
	frames = jnp.abs(frames)
	# frames = frames / 0.15  # arr.max()
	# frames = frames / jnp.percentile(frames.flatten(), 95)
	frames = jnp.clip(frames * 255, 0, 255).astype(jnp.uint8)
	frames = np.array(frames)
	iio.imwrite(os.path.join(path, 'anim_warp.gif'), frames[..., [0, 0, 0]])

	# import matplotlib.pyplot as plt
	# fig, ax = plt.subplots(2, 2)
	# for i in range(2):
	# 	ax[0, i].imshow(curl_and_div.arr[i], origin='lower')
	# 	ax[1, i].imshow(force.arr[i], origin='lower')
	# 	# ax[i].colorbar()
	# plt.show()