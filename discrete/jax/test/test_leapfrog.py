"""
ok... we have something working but its unconditionally unstable.
what gives?
"""
# from jax.config import config
# config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

from discrete.jax.field import SpaceTimeField


# def step_raw(field):
# 	"""raw gp step for full gp on x+t-"""
# 	id = lambda x, a: x - jnp.roll(x, shift=+1, axis=a)
# 	ed = lambda x, a: jnp.roll(x, shift=-1, axis=a) - x
#
# 	arr = field.arr.copy()
# 	speed = 0.5
#
# 	arr = arr.at[0].add( id(arr[3], 0) * +speed)        # et
# 	arr = arr.at[1].add( ed(arr[2], 0) * -speed)        # et
#
# 	arr = arr.at[2].add( id(arr[1], 0) * -speed)        # it
# 	arr = arr.at[3].add( ed(arr[0], 0) * +speed)        # it
#
# 	return field.copy(arr=arr)


def test_1d():
	# FIXME: x-t+ sig does not support mass terms!? not sure i understand why?
	print()
	from numga.algebra.algebra import Algebra
	algebra = Algebra.from_str('x+t-')
	shape = (256,)
	field = SpaceTimeField.from_subspace(algebra.subspace.multivector(), shape)
	# field = SpaceTimeField.from_subspace(algebra.subspace.even_grade(), shape)
	# field = SpaceTimeField.from_subspace(algebra.subspace.vector(), shape)
	x = field.meshgrid()
	x2 = (x ** 2).sum(axis=0, keepdims=False)
	gauss = jnp.exp(-x2*50)
	# gauss = jnp.roll(gauss, 30, axis=-1)
	key = jax.random.PRNGKey(4)
	vec = jax.random.normal(key, (field.components,))
	vec = vec / jnp.linalg.norm(vec)
	q = jnp.outer(vec, gauss)
	field.arr = q

	speed = 0.5 #* (1-jnp.exp(-x2 * 3)*0.5)
	mass = (x2 *0 + 0.1) * 0
	@jax.jit
	def step(state, unroll):
		def inner(_, field):
			return field.geometric_derivative_leapfrog(speed=speed, mass=mass)
		return jax.lax.fori_loop(0, unroll, inner, state)


	frames = []
	for i in range(256):
		field = step(field, unroll=2)
		frames.append(field.x_t_xt.T)
		# frames.append(field.arr[jnp.array([0, 0, 1])] .T)
		# frames.append(field.arr[jnp.array([0, 0, 1])] .T)


	path = r'../../output/leapfrog_1d'
	import imageio.v3 as iio
	import os
	os.makedirs(path, exist_ok=True)
	frames = jnp.array(frames)[::-1]
	frames = jnp.abs(frames)
	# frames = frames / 0.15  # arr.max()
	# frames = frames / jnp.percentile(frames.flatten(), 95)
	frames = jnp.clip(frames * 255, 0, 255).astype(jnp.uint8)
	frames = jnp.array(frames)
	iio.imwrite(os.path.join(path, 'anim.gif'), frames)


def kink(field):
	"""
	apply a rotor to a 4d field
	how to construct rotor? exp(k.dot(X).dual)?
	"""

	return field


def test_1d_mass():
	print()
	from numga.algebra.algebra import Algebra
	algebra = Algebra.from_str('x+t-')
	shape = (256,)
	field = SpaceTimeField.from_subspace(algebra.subspace.multivector(), shape)
	# field = SpaceTimeField.from_subspace(algebra.subspace.even_grade(), shape)
	# field = SpaceTimeField.from_subspace(algebra.subspace.vector(), shape)
	x = field.meshgrid()
	x2 = (x ** 2).sum(axis=0, keepdims=False)
	gauss = jnp.exp(-x2*30)
	gauss = jnp.roll(gauss, 30, axis=-1)
	key = jax.random.PRNGKey(11)
	vec = jax.random.normal(key, (field.components,))
	vec = vec / jnp.linalg.norm(vec)
	q = jnp.outer(vec, gauss)
	field.arr = q
	# mass = 0.3 + x2 /6# + 0.1
	mass = 0.2 #+ (1 - jnp.exp(-x2 * 3)) / 13
	# mass = 0
	# print(mass.max(), mass.shape)
	speed = 1/2 * (1-jnp.exp(-x2 * 3)*0.5)


	@jax.jit
	def step(state, unroll):
		def inner(_, field):
			return field.geometric_derivative_leapfrog(mass=mass, speed=speed)
		return jax.lax.fori_loop(0, unroll, inner, state)


	frames = []
	for i in range(256*4):
		field = step(field, unroll=2)
		# frames.append(field.x_t_xt.T)
		# frames.append(field.arr[jnp.array([0, 0, 1])] .T)
		frames.append(field.arr[jnp.array([0, 1, 2])] .T)


	path = r'../../output/leapfrog_1d_massl_g'
	import imageio.v3 as iio
	import os
	os.makedirs(path, exist_ok=True)
	frames = jnp.array(frames)
	frames = jnp.abs(frames) / 2
	# frames = frames / 0.15  # arr.max()
	# frames = frames / jnp.percentile(frames.flatten(), 99)
	frames = jnp.clip(frames * 255, 0, 255).astype(jnp.uint8)
	frames = jnp.array(frames)
	iio.imwrite(os.path.join(path, 'xt.gif'), frames[::-1])


def test_1d_massI():
	print()
	from numga.algebra.algebra import Algebra
	algebra = Algebra.from_str('x-t+')
	shape = (256,)
	field = SpaceTimeField.from_subspace(algebra.subspace.multivector(), shape)
	# field = SpaceTimeField.from_subspace(algebra.subspace.even_grade(), shape)
	# field = SpaceTimeField.from_subspace(algebra.subspace.vector(), shape)
	x = field.meshgrid()
	x2 = (x ** 2).sum(axis=0, keepdims=False)
	gauss = jnp.exp(-x2*30)
	gauss = jnp.roll(gauss, 30, axis=-1)
	key = jax.random.PRNGKey(11)
	vec = jax.random.normal(key, (field.components,))
	vec = vec / jnp.linalg.norm(vec)
	q = jnp.outer(vec, gauss)
	field.arr = q
	# mass = 0.3 + x2 /6# + 0.1
	mass = 0.0 #+ (1 - jnp.exp(-x2 * 3)) / 13
	# mass = 0
	# print(mass.max(), mass.shape)
	speed = 1/2 * (1-jnp.exp(-x2 * 3)*0.5)
	mass_I = 0.2


	@jax.jit
	def step(state, unroll):
		def inner(_, field):
			return field.geometric_derivative_leapfrog(mass=mass, mass_I=mass_I, speed=speed)
		return jax.lax.fori_loop(0, unroll, inner, state)


	frames = []
	for i in range(256*4):
		field = step(field, unroll=2)
		# frames.append(field.x_t_xt.T)
		# frames.append(field.arr[jnp.array([0, 0, 1])] .T)
		frames.append(field.arr[jnp.array([0, 1, 2])] .T)


	path = r'../../output/leapfrog_1d_mass_I'
	import imageio.v3 as iio
	import os
	os.makedirs(path, exist_ok=True)
	frames = jnp.array(frames)
	frames = jnp.abs(frames) / 2
	# frames = frames / 0.15  # arr.max()
	# frames = frames / jnp.percentile(frames.flatten(), 99)
	frames = jnp.clip(frames * 255, 0, 255).astype(jnp.uint8)
	frames = jnp.array(frames)
	iio.imwrite(os.path.join(path, 'xt.gif'), frames[::-1])


def test_1d_mass_sig():
	"""attempted deep dive into flipped sigs
	mass term appears broken?
	"""
	print()
	from numga.algebra.algebra import Algebra
	algebra = Algebra.from_str('x-t+')
	shape = (256,)
	field = SpaceTimeField.from_subspace(algebra.subspace.multivector(), shape)
	# field = SpaceTimeField.from_subspace(algebra.subspace.even_grade(), shape)
	# field = SpaceTimeField.from_subspace(algebra.subspace.vector(), shape)
	x = field.meshgrid()
	x2 = (x ** 2).sum(axis=0, keepdims=False)
	gauss = jnp.exp(-x2*20)
	gauss = jnp.roll(gauss, 30, axis=-1)
	key = jax.random.PRNGKey(11)
	vec = jax.random.normal(key, (field.components,))
	vec = vec / jnp.linalg.norm(vec)
	q = jnp.outer(vec, gauss)
	field.arr = q
	# mass = 0.3 + x2 /6# + 0.1
	mass = -0.3 #+ (1 - jnp.exp(-x2 * 3)) / 13
	# mass = 0
	# print(mass.max(), mass.shape)
	speed = 1/2 * (1-jnp.exp(-x2 * 3)*0.5)


	@jax.jit
	def step(state, unroll):
		def inner(_, field):
			return field.geometric_derivative_leapfrog(mass=mass, speed=speed)
		return jax.lax.fori_loop(0, unroll, inner, state)


	frames = []
	for i in range(256*4):
		field = step(field, unroll=2)
		# frames.append(field.x_t_xt.T)
		# frames.append(field.arr[jnp.array([0, 0, 1])] .T)
		frames.append(field.arr[jnp.array([0, 1, 2])] .T)


	path = r'../../output/leapfrog_1d_sig_mass_g'
	import imageio.v3 as iio
	import os
	os.makedirs(path, exist_ok=True)
	frames = jnp.array(frames)
	frames = jnp.abs(frames) / 2
	# frames = frames / 0.15  # arr.max()
	# frames = frames / jnp.percentile(frames.flatten(), 99)
	frames = jnp.clip(frames * 255, 0, 255).astype(jnp.uint8)
	frames = jnp.array(frames)
	iio.imwrite(os.path.join(path, 'xt.gif'), frames[::-1])


def test_2d():
	print()
	from numga.algebra.algebra import Algebra
	algebra = Algebra.from_str('x+y+t-')
	shape = (128, 128)
	field = SpaceTimeField.from_subspace(algebra.subspace.multivector(), shape)
	# field = SpaceTimeField.from_subspace(algebra.subspace.odd_grade(), shape)
	# field = SpaceTimeField.from_subspace(algebra.subspace.vector(), shape)
	x = field.meshgrid()
	x2 = (x ** 2).sum(axis=0, keepdims=False)
	gauss = jnp.exp(-x2*50)
	# gauss = jnp.roll(gauss, 5, axis=-1)

	key = jax.random.PRNGKey(1)
	vec = jax.random.normal(key, (field.components,))
	vec = vec / jnp.linalg.norm(vec)
	# vec = vec.at[0].add(2)
	q = jnp.einsum('c, ...->c...', vec, gauss)
	field.arr = q
	mass = 0.4 + x2 / 2# + 0.1
	speed = 1/3 #* (1-jnp.exp(-x2 * 3) *0.5)
	@jax.jit
	def step(state, unroll):
		def inner(_, field):
			return field.geometric_derivative_leapfrog(speed=speed)
		return jax.lax.fori_loop(0, unroll, inner, state)


	frames = []
	for i in range(256):
		field = step(field, unroll=3)
		frames.append(field.xt_yt_xy.T)
		# frames.append(field.x_y_t.T)


	path = r'../../output/leapfrog_2d_mv'
	import imageio.v3 as iio
	import os
	os.makedirs(path, exist_ok=True)
	frames = jnp.array(frames)#[::-1]
	frames = jnp.abs(frames)
	# frames = frames / 0.15  # arr.max()
	frames = frames / jnp.percentile(frames.flatten(), 99)
	frames = jnp.clip(frames * 255, 0, 255).astype(jnp.uint8)
	frames = jnp.array(frames)
	iio.imwrite(os.path.join(path, 'anim.gif'), frames)
	iio.imwrite(os.path.join(path, 'xy.gif'), frames[50])
	iio.imwrite(os.path.join(path, 'xt.gif'), frames[:128][::-1, 64])


def test_2d_1vec():
	print()
	from numga.algebra.algebra import Algebra
	algebra = Algebra.from_str('x+y+t-')
	shape = (128, 128)
	field = SpaceTimeField.from_subspace(algebra.subspace.vector(), shape)
	x = field.meshgrid()
	x2 = (x ** 2).sum(axis=0, keepdims=False)
	gauss = jnp.exp(-x2*50)
	gauss = jnp.roll(gauss, 20, axis=-1)

	key = jax.random.PRNGKey(12)
	vec = jax.random.normal(key, (field.components,))
	vec = vec / jnp.linalg.norm(vec)
	vec = vec.at[0].add(2)
	q = jnp.einsum('c, ...->c...', vec, gauss)
	field.arr = q
	mass = 0.4 + x2 / 2# + 0.1

	@jax.jit
	def step(state, unroll):
		def inner(_, field):
			return field.geometric_derivative_leapfrog()
		return jax.lax.fori_loop(0, unroll, inner, state)


	frames = []
	for i in range(256):
		field = step(field, unroll=3)
		# frames.append(field.xt_yt_xy.T)
		frames.append(field.x_y_t.T)


	path = r'../../output/leapfrog_2d_1vec'
	import imageio.v3 as iio
	import os
	os.makedirs(path, exist_ok=True)
	frames = jnp.array(frames)#[::-1]
	frames = jnp.abs(frames)
	# frames = frames / 0.15  # arr.max()
	frames = frames / jnp.percentile(frames.flatten(), 99)
	frames = jnp.clip(frames * 255, 0, 255).astype(jnp.uint8)
	frames = jnp.array(frames)
	iio.imwrite(os.path.join(path, 'anim.gif'), frames)


def test_2d_compact():
	print()
	from numga.algebra.algebra import Algebra
	algebra = Algebra.from_str('x+y+t-')
	nx = 2
	shape = (nx, 256)
	field = SpaceTimeField.from_subspace(algebra.subspace.multivector(), shape)
	x = field.meshgrid()
	x2 = (x[1:] ** 2).sum(axis=0, keepdims=False)
	gauss = jnp.exp(-x2*30)
	gauss = jnp.roll(gauss, 30, axis=-1)

	key = jax.random.PRNGKey(2)
	vec = jax.random.normal(key, (field.components, nx))
	vec = vec + vec[:, 1:] * 1
	vec = vec / jnp.linalg.norm(vec)
	# vec = vec.at[0].add(2)
	q = jnp.einsum('cx, x...->cx...', vec, gauss)
	field.arr = q
	mass = None #+ x2 / 2# + 0.1
	speed = 1/3 #* (1-jnp.exp(-x2 * 3)*0.5)
	dy = (1-jnp.exp(-x2 * 3)*0.5)
	metric = [0.2*dy, 1]


	@jax.jit
	def step(state, unroll):
		def inner(_, field):
			return field.geometric_derivative_leapfrog(speed=speed, mass=mass, metric=metric)
		return jax.lax.fori_loop(0, unroll, inner, state)


	frames = []
	for i in range(256*4):
		field = step(field, unroll=3)
		# frames.append(field.xt_yt_xy.T)
		# frames.append(field.x_y_t.T[:, 0])
		frames.append(jnp.abs(field.xt_yt_xy).T.mean(axis=1))


	path = r'../../output/leapfrog_2d_compact_slow'
	import imageio.v3 as iio
	import os
	os.makedirs(path, exist_ok=True)
	frames = jnp.array(frames)[::-1]
	frames = jnp.abs(frames)
	# frames = frames / 0.15  # arr.max()
	frames = frames / jnp.percentile(frames.flatten(), 99)
	frames = jnp.clip(frames * 255, 0, 255).astype(jnp.uint8)
	frames = jnp.array(frames)
	# iio.imwrite(os.path.join(path, 'anim.gif'), frames)
	iio.imwrite(os.path.join(path, 'xt.gif'), frames)


def test_2d_compact_sig():
	"""in 2d, direct mass term with flipped sig also seems broken"""
	print()
	from numga.algebra.algebra import Algebra
	algebra = Algebra.from_str('x-y-t+')
	nx = 2
	shape = (nx, 256)
	field = SpaceTimeField.from_subspace(algebra.subspace.full(), shape)
	x = field.meshgrid()
	x2 = (x[1:] ** 2).sum(axis=0, keepdims=False)
	gauss = jnp.exp(-x2*50)
	gauss = jnp.roll(gauss, 20, axis=-1)

	key = jax.random.PRNGKey(5)
	vec = jax.random.normal(key, (field.components, nx))
	vec = vec + vec[:, 1:] * 1
	vec = vec / jnp.linalg.norm(vec)
	# vec = vec.at[0].add(2)
	q = jnp.einsum('cx, x...->cx...', vec, gauss)
	field.arr = q
	mass = 0.0 #+ x2 / 2# + 0.1
	speed = 1/3 * (1-jnp.exp(-x2 * 3)*0.5)


	@jax.jit
	def step(state, unroll):
		def inner(_, field):
			return field.geometric_derivative_leapfrog(speed=speed, mass=mass)
		return jax.lax.fori_loop(0, unroll, inner, state)


	frames = []
	for i in range(256*4):
		field = step(field, unroll=3)
		# frames.append(field.xt_yt_xy.T)
		# frames.append(field.x_y_t.T[:, 0])
		frames.append(jnp.abs(field.x_y_t).T.mean(axis=1))


	path = r'../../output/leapfrog_2d_compact_sig'
	import imageio.v3 as iio
	import os
	os.makedirs(path, exist_ok=True)
	frames = jnp.array(frames)[::-1]
	frames = jnp.abs(frames)
	# frames = frames / 0.15  # arr.max()
	frames = frames / jnp.percentile(frames.flatten(), 99)
	frames = jnp.clip(frames * 255, 0, 255).astype(jnp.uint8)
	frames = jnp.array(frames)
	# iio.imwrite(os.path.join(path, 'anim.gif'), frames)
	iio.imwrite(os.path.join(path, 'xt.gif'), frames)


def test_3d():
	print()
	from numga.algebra.algebra import Algebra
	algebra = Algebra.from_str('x+y+z+t-')
	shape = (64, 64, 64)
	field = SpaceTimeField.from_subspace(algebra.subspace.multivector(), shape)
	# field = SpaceTimeField.from_subspace(algebra.subspace.even_grade(), shape)
	# field = SpaceTimeField.from_subspace(algebra.subspace.vector(), shape)
	x = field.meshgrid()
	x2 = (x ** 2).sum(axis=0, keepdims=False)
	gauss = jnp.exp(-x2*30)
	gauss = jnp.roll(gauss, 10, axis=-1)

	key = jax.random.PRNGKey(10)
	vec = jax.random.normal(key, (field.components,))
	vec = vec / jnp.linalg.norm(vec)
	# vec = vec.at[0].add(2)
	q = jnp.einsum('c, ...->c...', vec, gauss)
	field.arr = q
	mass = 0.2 + x2 / 4# + 0.1

	@jax.jit
	def step(state, unroll):
		def inner(_, field):
			return field.geometric_derivative_leapfrog()
		return jax.lax.fori_loop(0, unroll, inner, state)


	frames = []
	for i in range(400):
		field = step(field, unroll=1)
		# frames.append(field.xt_yt_xy.T[:, 32])
		frames.append(field.xt_yt_xy.T[:, 32])


	path = r'../../output/leapfrog_3d_1'
	import imageio.v3 as iio
	import os
	os.makedirs(path, exist_ok=True)
	frames = jnp.array(frames)#[::-1]
	frames = jnp.abs(frames)
	# frames = frames / 0.15  # arr.max()
	frames = frames / jnp.percentile(frames.flatten(), 99)
	frames = jnp.clip(frames * 255, 0, 255).astype(jnp.uint8)
	frames = jnp.array(frames)
	iio.imwrite(os.path.join(path, 'anim.gif'), frames)
	iio.imwrite(os.path.join(path, 'static.gif'), frames[-1])


def test_3d_bivector():
	print()
	from numga.algebra.algebra import Algebra
	algebra = Algebra.from_str('x+y+z+t-')
	shape = (128, 128, 128)
	field = SpaceTimeField.from_subspace(algebra.subspace.bivector(), shape)
	x = field.meshgrid()
	x2 = (x ** 2).sum(axis=0, keepdims=False)
	gauss = jnp.exp(-x2*10)
	# gauss = jnp.roll(gauss, 10, axis=-1)

	key = jax.random.PRNGKey(10)
	vec = jax.random.normal(key, (field.components,))
	vec = vec / jnp.linalg.norm(vec)
	# vec = vec.at[0].add(2)
	q = jnp.einsum('c, ...->c...', vec, gauss)
	field.arr = q
	mass = 0.2 +0* x2 / 4# + 0.1

	@jax.jit
	def step(state, unroll):
		def inner(_, field):
			return field.geometric_derivative_leapfrog(speed=1/4)
		return jax.lax.fori_loop(0, unroll, inner, state)


	frames = []
	for i in range(128):
		field = step(field, unroll=4)
		# frames.append(field.xy_yz_xz.T[:, 32])
		frames.append(field.xt_yt_zt.T[:, 64])


	path = r'../../output/leapfrog_3d_bivector'
	import imageio.v3 as iio
	import os
	os.makedirs(path, exist_ok=True)
	frames = jnp.array(frames)#[::-1]
	frames = jnp.abs(frames)
	# frames = frames / 0.15  # arr.max()
	frames = frames / jnp.percentile(frames.flatten(), 99)
	frames = jnp.clip(frames * 255, 0, 255).astype(jnp.uint8)
	frames = jnp.array(frames)
	iio.imwrite(os.path.join(path, 'xyt.gif'), frames)
	iio.imwrite(os.path.join(path, 'xy.gif'), frames[50])
	iio.imwrite(os.path.join(path, 'xt.gif'), frames[:, 64])


def test_3d_even_compact():
	print()
	from numga.algebra.algebra import Algebra
	algebra = Algebra.from_str('x+y+z+t-')
	nx = 2
	shape = (nx, 128, 128)
	field = SpaceTimeField.from_subspace(algebra.subspace.even_grade(), shape)
	# field = SpaceTimeField.from_subspace(algebra.subspace.from_grades([1]), shape)
	x = field.meshgrid()
	x = jnp.einsum('cxyz, c-> cxyz', x, jnp.array([0.1, 1, 1]))
	x2 = (x ** 2).sum(axis=0, keepdims=False)
	gauss = jnp.exp(-x2*40)
	# gauss = jnp.roll(gauss, 10, axis=-1)

	key = jax.random.PRNGKey(0)
	vec = jax.random.normal(key, (field.components, nx))
	vec = vec / jnp.linalg.norm(vec)
	# vec = vec.at[0].add(2)
	q = jnp.einsum('cx, x...->cx...', vec, gauss)
	field.arr = q
	mass = 0.2 +0* x2 / 4# + 0.1
	speed = 1/4
	mass_I = -0.2
	metric = [mass, 1, 1]

	@jax.jit
	def step(state, unroll):
		def inner(_, field):
			return field.geometric_derivative_leapfrog(speed=speed, mass_I=mass_I, metric=metric)
		return jax.lax.fori_loop(0, unroll, inner, state)


	frames = []
	for i in range(128):
		field = step(field, unroll=4)
		# frames.append(field.xt_yt_xy.T[:, 32])
		# frames.append(field.xt_yt_xy.T[:, 32])
		frames.append(field.xt_yt_zt.T[:, :, 0])
		# frames.append(field.y_z_t.T[:, :, 0])
		# frames.append(field.arr[jnp.array([0, 0, 1])].T[:, :, 0])


	path = r'../../output/leapfrog_3d_2_even_massI'
	import imageio.v3 as iio
	import os
	os.makedirs(path, exist_ok=True)
	frames = jnp.array(frames)#[::-1]
	frames = jnp.abs(frames)
	# frames = frames / 0.15  # arr.max()
	frames = frames / jnp.percentile(frames.flatten(), 99)
	frames = jnp.clip(frames * 255, 0, 255).astype(jnp.uint8)
	frames = jnp.array(frames)
	iio.imwrite(os.path.join(path, 'anim.gif'), frames, loop=0)
	iio.imwrite(os.path.join(path, 'static.gif'), frames[50])
	iio.imwrite(os.path.join(path, 'xt.gif'), frames[::-1, 64])
