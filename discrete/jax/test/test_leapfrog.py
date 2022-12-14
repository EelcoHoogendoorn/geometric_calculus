"""Jax based tests
"""

import jax
import jax.numpy as jnp
import numpy as np

from discrete.jax.field import SpaceTimeField

import imageio.v3 as iio
import os


# FIXME: make writer part of abstract base class?
def tonemap(field, components, norm, gamma):
	frames = getattr(field, components)
	# print(frames.shape)
	frames = np.abs(frames) * 1.5
	if isinstance(norm, int):
		frames = frames / jnp.percentile(frames.flatten(), norm)
	if isinstance(norm, float):
		frames = frames / norm
	if gamma:
		frames = np.sqrt(frames)
	frames = jnp.clip(frames * 255, 0, 255).astype(np.uint8)

	frames = np.moveaxis(frames, 0, -1) # color components last
	frames = np.moveaxis(frames, -2, 0) # t first

	return frames


def write_gif_1d(basepath, field, components, pre='', post='', norm=None, gamma=True):
	basename = '_'.join([pre, str(field.shape), field.algebra.description.description_str, field.subspace.pretty_str, components, post])
	os.makedirs(basepath, exist_ok=True)

	frames = tonemap(field, components, norm, gamma)

	iio.imwrite(os.path.join(basepath, basename+'_xt.gif'), frames[::-1])


def write_gif_2d(basepath, field, components, pre='', post='', norm=None, gamma=True):
	basename = '_'.join([pre, str(field.shape), field.algebra.description.description_str, field.subspace.pretty_str, components, post])
	os.makedirs(basepath, exist_ok=True)
	frames = tonemap(field, components, norm, gamma)

	iio.imwrite(os.path.join(basepath, basename+'_anim.gif'), frames)
	iio.imwrite(os.path.join(basepath, basename+'_xt.gif'), frames[::-1, field.shape[0]//2])


def write_gif_2d_compact(basepath, field, components, pre='', post='', norm=None, gamma=True):
	basename = '_'.join([pre, str(field.shape), field.algebra.description.description_str, field.subspace.pretty_str, components, post])
	os.makedirs(basepath, exist_ok=True)
	frames = tonemap(field, components, norm, gamma)
	iio.imwrite(os.path.join(basepath, basename+'_xt.gif'), frames[::-1].mean(axis=1).astype(np.uint8))


def write_gif_3d(basepath, field, components, pre='', post='', norm=None, gamma=True):
	basename = '_'.join([pre, str(field.shape), field.algebra.description.description_str, field.subspace.pretty_str, components, post])
	os.makedirs(basepath, exist_ok=True)
	frames = tonemap(field, components, norm, gamma)

	iio.imwrite(os.path.join(basepath, basename+'_anim.gif'), frames[:, field.shape[0]//2])
	iio.imwrite(os.path.join(basepath, basename+'_xt.gif'), frames[::-1, field.shape[0]//2, field.shape[1]//2])


def random_gaussian(field, sigma, location=0, seed=11):
	"""initialize a field with a random gaussian"""
	gauss = field.gauss(jnp.array(sigma), jnp.array(location))
	key = jax.random.PRNGKey(seed)
	vec = jax.random.normal(key, (field.components,))
	vec = vec / jnp.linalg.norm(vec)
	q = jnp.einsum('c, ...->c...', vec, gauss)
	return field.copy(arr=q)


def test_1d():
	print()
	from numga.algebra.algebra import Algebra
	algebra = Algebra.from_str('x+t-')
	shape = (256,)
	steps = 256
	field = SpaceTimeField.from_subspace(algebra.subspace.multivector(), shape)
	field = random_gaussian(field, 0.1)

	mass = (field.quadratic() *0 + 0.1) * 0

	full_field = field.rollout(steps, mass=mass)

	write_gif_1d('../../output', full_field, 'x_t_xt', post='mass')


def test_1d_mass():
	print()
	from numga.algebra.algebra import Algebra
	algebra = Algebra.from_str('x+t-')
	shape = (256,)
	steps = 256
	field = SpaceTimeField.from_subspace(algebra.subspace.multivector(), shape)

	field = random_gaussian(field, 0.1)

	mass = (field.quadratic() *0 + 0.1)

	metric = {'t': (1-field.gauss(0.3)*0.5)}

	full_field = field.rollout(steps, mass=mass, metric=metric)

	write_gif_1d('../../output', full_field, 'x_t_xt', post='mass_metric')


def test_1d_massI():
	from numga.algebra.algebra import Algebra
	algebra = Algebra.from_str('x-t+')
	shape = (256,)
	steps = 256*4
	field = SpaceTimeField.from_subspace(algebra.subspace.multivector(), shape)
	field = random_gaussian(field, 0.1, 0.1)
	mass = 0.0 #+ (1-field.gauss(0.3)) / 13
	metric = {'t': (1-field.gauss(0.3)*0.5)}
	mass_I = 0.2

	full_field = field.rollout(steps, mass_I=mass_I, metric=metric)

	write_gif_1d('../../output', full_field, 'x_t_xt', post='mass_I')


def test_1d_mass_sig():
	"""attempted deep dive into flipped sigs
	mass term appears broken? or is this sig just to blame?
	"""
	print()
	from numga.algebra.algebra import Algebra
	algebra = Algebra.from_str('x-t+')
	shape = (256,)
	steps = 256

	field = SpaceTimeField.from_subspace(algebra.subspace.multivector(), shape)
	field = random_gaussian(field, 0.1)
	mass = -0.3 #+ (1 - jnp.exp(-x2 * 3)) / 13

	metric = {'t': (1-field.gauss(0.3)) * 0.5}

	full_field = field.rollout(steps, mass=mass, metric=metric)

	write_gif_1d('../../output', full_field, 'x_t_xt', post='sig_mass')



def test_2d():
	print()
	from numga.algebra.algebra import Algebra
	algebra = Algebra.from_str('x+y+t-')
	shape = (128, 128)
	steps = 256
	field = SpaceTimeField.from_subspace(algebra.subspace.multivector(), shape)
	field = random_gaussian(field, 0.1)
	mass = 0.4 + field.quadratic() / 2# + 0.1

	# metric = {'t': (1-field.gauss(0.3)*0.5)}

	full_field = field.rollout(steps, mass=mass, metric={})

	write_gif_2d('../../output', full_field, 'xt_yt_xy', post='mass', norm=99)


def test_2d_1vec():
	"""note: like all equations over a non-closed subspace, this thing has non-propagating residual,
	because we are too lazy to implement a compatible initalization yet"""
	print()
	from numga.algebra.algebra import Algebra
	algebra = Algebra.from_str('x+y+t-')
	shape = (128, 128)
	steps = 128
	field = SpaceTimeField.from_subspace(algebra.subspace.vector(), shape)

	field = random_gaussian(field, 0.1)

	full_field = field.rollout(steps, metric={})

	write_gif_2d('../../output', full_field, 'x_y_t', post='mass')


def test_2d_compact():
	print()
	from numga.algebra.algebra import Algebra
	algebra = Algebra.from_str('x+y+t-')
	nx = 2
	shape = (nx, 256)
	steps = 256*2
	field = SpaceTimeField.from_subspace(algebra.subspace.multivector(), shape)
	field = random_gaussian(field, 0.1, 0.1)
	dimple = (1-field.gauss(jnp.array([1e16, 0.3]))*0.3)
	metric = {'x': dimple * 0.2}

	full_field = field.rollout(steps, metric=metric, mass=0)
	write_gif_2d_compact('../../output', full_field, 'xy_xt_yt', post='compact_slow', norm=99)




def test_2d_compact_sig():
	"""in 2d, direct mass term with flipped sig also seems broken"""
	print()
	from numga.algebra.algebra import Algebra
	algebra = Algebra.from_str('x-y-t+')
	nx = 2
	shape = (nx, 256)
	steps = 256
	field = SpaceTimeField.from_subspace(algebra.subspace.full(), shape)
	field = random_gaussian(field, 0.1, 0.1)
	dimple = (1-field.gauss(jnp.array([1e16, 0.3]))*0.3)
	metric = {'x': dimple * 0.2}
	mass = 0.2

	full_field = field.rollout(steps, metric=metric, mass=mass)

	write_gif_2d_compact('../../output', full_field, 'xy_xt_yt', post='compact_sig')



def test_3d():
	print()
	from numga.algebra.algebra import Algebra
	algebra = Algebra.from_str('x+y+z+t-')
	steps = 64
	shape = (64, 64, 64)
	field = SpaceTimeField.from_subspace(algebra.subspace.multivector(), shape)

	field = random_gaussian(field, 0.1, 0.1)
	mass = 0.2 + field.quadratic() / 4
	metric = {}

	full_field = field.rollout(steps, metric=metric, mass=mass)
	write_gif_3d('../../output', full_field, 'xy_xt_yt', post='', norm=99)


def test_3d_bivector():
	print()
	from numga.algebra.algebra import Algebra
	algebra = Algebra.from_str('x+y+z+t-')
	shape = (64, 64, 64)
	steps = 64
	field = SpaceTimeField.from_subspace(algebra.subspace.bivector(), shape)
	field = random_gaussian(field, 0.1)
	metric = {}

	full_field = field.rollout(steps, metric=metric)
	write_gif_3d('../../output', full_field, 'xt_yt_zt', post='', norm=99)


def test_3d_even_compact():
	print()
	from numga.algebra.algebra import Algebra
	algebra = Algebra.from_str('x+y+z+t-')
	nx = 2
	shape = (nx, 128, 128)
	steps = 128
	field = SpaceTimeField.from_subspace(algebra.subspace.even_grade(), shape)

	field = random_gaussian(field, 0.1)
	mass = 0.2 +0* field.quadratic() / 4# + 0.1
	mass_I = -0.2

	metric = {'x': mass}

	full_field = field.rollout(steps, metric=metric, mass_I=mass_I)
	write_gif_3d('../../output', full_field, 'xt_yt_zt', post='mass_I', norm=99)
