"""Jax based tests
"""

import jax.numpy as jnp
import numpy as np

from numga.algebra.algebra import Algebra
from discrete.jax.field_slice import FieldSlice


def filter_stationary(field, n=1, **kwargs):
	"""Subtract a timestepped field from itself, to bring out higher-frequency components"""
	for i in range(n):
		correction = field.rollout(2, **kwargs).slice(-1).arr
		field.arr = field.arr - correction
		# field = field.geometric_derivative_leapfrog(metric=metric)
		# field.arr = field.arr - field.arr.mean(axis=1, keepdims=True)
	return field
def filter_lightlike(field):
	"""erase average of compact dimension, suppressing lightlike modes"""
	return field.copy(field.arr - field.arr.mean(axis=1, keepdims=True))
def filter_massive(field, axis=1):
	"""erase motion in compact dimension, supressing massive modes"""
	return field.copy(field.arr.at[...].set(field.arr.mean(axis=axis, keepdims=True)))


def test_wxyt_even_conservation():
	"""test amplitude conservation"""
	print()
	algebra = Algebra.from_str('w+x+y+t-')
	shape, steps = (16, 16, 16), 32
	field = FieldSlice.from_subspace(algebra.subspace.even_grade(), shape)

	field = field.random_gaussian([0.3, 0.3, 0.3], [0, 0, 0.1])
	dimple = (1-field.gauss([0.6, 0.6, 0.6])*0.5)
	metric = {'w': dimple / 4, 't': 0.1}

	ff = field.rollout(steps, metric=metric)
	# test amplitude conservation over time
	print(ff.arr.sum((0, 1, 2, 3)))
	metric = {'w': dimple[..., None] / 4, 't': 0.1}   # broadcast over time dims
	res = ff.geometric_derivative(metric=metric)
	print(np.unravel_index(res.arr.argmax(), res.arr.shape))
	print (res.arr.max())


def test_xt_full_mass():
	print()
	algebra = Algebra.from_str('x+t-')
	shape, steps = (256,), 512
	field = FieldSlice.from_subspace(algebra.subspace.multivector(), shape)

	field = field.random_gaussian([0.2])
	# mass = 0.1
	field.arr = field.arr.at[1:3].set(0)

	dimple = (1-field.gauss([0.6])*0.5) / 4
	mass = dimple
	# metric = {'t': dimple / 4}
	# metric = {'t': 0.33}

	field.write_gif_1d_generator(
		field.rollout_generator(steps, mass=mass),
		basepath='../../output', components='x_t_xt', pre='jax',
	)



def test_1d():
	print()
	algebra = Algebra.from_str('x+t-')
	shape = (256,)
	steps = 256
	field = FieldSlice.from_subspace(algebra.subspace.multivector(), shape)
	field = field.random_gaussian(0.1)

	mass = (field.quadratic() *0 + 0.1) * 0

	full_field = field.rollout(steps, mass=mass)

	full_field.write_gif_1d(basepath='../../output', components='x_t_xt', post='mass')


def test_1d_mass():
	print()
	algebra = Algebra.from_str('x+t-')
	shape = (256,)
	steps = 256
	field = FieldSlice.from_subspace(algebra.subspace.multivector(), shape)

	field = field.random_gaussian(0.1)

	mass = (field.quadratic() *0 + 0.1)

	metric = {'t': (1-field.gauss(0.3)*0.5)}

	full_field = field.rollout(steps, mass=mass)#, metric=metric)

	full_field.write_gif_1d(basepath='../../output', components='x_t_xt', post='mass_metric')


def test_1d_massI():
	algebra = Algebra.from_str('x-t+')
	shape = (256,)
	steps = 256*4
	field = FieldSlice.from_subspace(algebra.subspace.multivector(), shape)
	field = field.random_gaussian(0.1, 0.1)
	mass = 0.0 #+ (1-field.gauss(0.3)) / 13
	metric = {'t': (1-field.gauss(0.3)*0.5)}
	mass_I = 0.2

	full_field = field.rollout(steps, mass_I=mass_I, metric=metric)

	full_field.write_gif_1d(basepath='../../output', components='x_t_xt', post='mass_I')


def test_1d_mass_sig():
	"""attempted deep dive into flipped sigs
	mass term appears broken? or is this sig just to blame?
	"""
	print()
	algebra = Algebra.from_str('x-t+')
	shape = (256,)
	steps = 256

	field = FieldSlice.from_subspace(algebra.subspace.multivector(), shape)
	field = field.random_gaussian(0.1)
	mass = -0.3 #+ (1 - jnp.exp(-x2 * 3)) / 13

	metric = {'t': (1-field.gauss(0.3)) * 0.5}

	full_field = field.rollout(steps, mass=mass, metric=metric)

	full_field.write_gif_1d(basepath='../../output', components='x_t_xt', post='sig_mass')



def test_2d():
	print()
	algebra = Algebra.from_str('x+y+t-')
	shape = (128, 128)
	steps = 256
	field = FieldSlice.from_subspace(algebra.subspace.multivector(), shape)
	field = field.random_gaussian(0.1)
	mass = 0.4 + field.quadratic() / 2# + 0.1

	# metric = {'t': (1-field.gauss(0.3)*0.5)}

	# full_field = field.rollout(steps, mass=mass, metric={})
	# full_field.write_gif_2d('../../output', 'xt_yt_xy', post='mass', norm=99)
	field.write_gif_2d_generator(
		field.rollout_generator(steps, mass=mass),
		basepath='../../output', components='xt_yt_xy', post='mass',
	)


def test_2d_perf():
	"""performance profiling test, to compare vs numpy and opencl"""
	print()
	algebra = Algebra.from_str('x+y+t-')
	shape = (512, 512)
	steps = 128
	field = FieldSlice.from_subspace(algebra.subspace.multivector(), shape)
	field = field.random_gaussian(0.1)

	full_field = field.rollout(steps)

	full_field.write_gif_2d(basepath='../../output', components='xt_yt_xy', pre='jax_perf', anim=False)


def test_2d_1vec():
	"""note: like all equations over a non-closed subspace, this thing has non-propagating residual,
	because we are too lazy to implement a compatible initalization yet"""
	print()
	algebra = Algebra.from_str('x+y+t-')
	shape = (128, 128)
	steps = 128
	field = FieldSlice.from_subspace(algebra.subspace.vector(), shape)

	field = field.random_gaussian(0.1)

	full_field = field.rollout(steps, metric={})

	full_field.write_gif_2d(basepath='../../output', components='x_y_t', post='mass')


def test_2d_compact():
	print()
	algebra = Algebra.from_str('w+y+t-')
	shape = (2, 256)
	steps = 256*2
	field = FieldSlice.from_subspace(algebra.subspace.multivector(), shape)
	field = field.random_gaussian(0.1, 0.1)
	# field = field.smooth_noise([1, 8])
	dimple = (1-field.gauss(jnp.array([1e16, 0.3]))*0.3)
	metric = {'w': dimple * 0.3}

	full_field = field.rollout(steps, metric=metric, mass=0.2)
	full_field.write_gif_2d_compact(basepath='../../output', components='wy_wt_yt', pre='')


def test_2d_compact_sig():
	"""in 2d, direct mass term with flipped sig also seems broken"""
	print()
	algebra = Algebra.from_str('x-y-t+')
	shape = (2, 256)
	steps = 256
	field = FieldSlice.from_subspace(algebra.subspace.full(), shape)
	field = field.random_gaussian(0.1, 0.1)
	dimple = (1-field.gauss(jnp.array([1e16, 0.3]))*0.3)
	metric = {'x': dimple * 0.2}
	mass = 0.2

	full_field = field.rollout(steps, metric=metric, mass=mass)

	full_field.write_gif_2d_compact(basepath='../../output', components='xy_xt_yt', post='compact_sig')


def test_3d():
	print()
	algebra = Algebra.from_str('x+y+z+t-')
	steps = 128
	shape = (64, 64, 64)
	field = FieldSlice.from_subspace(algebra.subspace.even_grade(), shape)

	field = field.random_gaussian(0.1, 0.0)
	mass = 0.2 + field.quadratic() / 4
	mass = None
	dimple = (1-field.gauss(0.3)*0.5)
	metric = {'t': dimple}
	# metric = {'t': dimple / 2}

	full_field = field.rollout(steps, metric=metric, mass=mass)
	full_field.write_gif_3d(basepath='../../output', components='xy_xt_yt', post='')


def test_3d_bivector():
	print()
	algebra = Algebra.from_str('x+y+z+t-')
	shape = (64, 64, 64)
	steps = 128
	field = FieldSlice.from_subspace(algebra.subspace.bivector(), shape)
	field = field.random_gaussian(0.1)

	for n in range(3):
		field = filter_stationary(field, n=n)
		full_field = field.rollout(steps)
		full_field.write_gif_3d(basepath='../../output', components='xt_yt_zt', pre=f'filtered{n}')


def test_3d_bivector_compact():
	print()
	algebra = Algebra.from_str('w+x+y+t-')
	shape = (2, 128, 128)
	steps = 128
	field = FieldSlice.from_subspace(algebra.subspace.bivector(), shape)
	field = field.random_gaussian(0.1, [0, 0, 0.1])
	dimple = (1-field.gauss(jnp.array([1, 0.3, 0.3]))*0.9)
	# metric = {'t': dimple}
	metric = {'w': dimple / 2}

	for n in range(3):
		full_field = filter_stationary(field, n=n, metric=metric).rollout(steps, metric=metric)
		full_field.write_gif_3d(basepath='../../output', components='xt_yt_wt', pre=f'filtered{n}')


def test_3d_bivector_potential():
	"""test potential based construction of compatible maxwell field
	another splendid display of my ignorance...
	why does this not work? is the lorentz gauge a lie?
	or are non-propagating field components an underrated aspect of the maxwell equation?
	"""
	print()
	algebra = Algebra.from_str('x+y+z+t-')
	shape = (64, 64, 64, 3)
	steps = 128
	from discrete.jax.field import Field
	# A_ = Field.from_subspace(algebra.subspace.vector(), shape)
	A2 = Field.from_subspace(algebra.subspace.bivector(), shape)
	A2 = A2.random_gaussian([0.1, 0.1, 0.1, 0.1])
	# higher order constructions of the same form seem to make no difference
	# A2 = A2.interior_derivative().exterior_derivative()
	# A2 = A2.geometric_derivative().geometric_derivative(output=A2.subspace)
	print(np.linalg.norm(A2.arr))
	# 1-vector EM potential A
	A1 = A2.interior_derivative()
	print(np.linalg.norm(A1.arr))
	# check that we indeed satisfy the lorentz gauge; all good
	print(np.linalg.norm(A1.interior_derivative().arr))
	# now this should be a valid EM field, amirite?
	F2 = A1.exterior_derivative()
	print(np.linalg.norm(F2.arr))
	# lets take one temporal slice and roll it out
	F2 = F2.slice(1).rollout(steps)
	print(np.linalg.norm(F2.arr))
	F2.write_gif_3d(basepath='../../output', componnts='xt_yt_zt', pre=f'potential', anim=True)


def test_3d_even_compact():
	"""test some compact spaces with more DOFs
	indeed it appears we can observe components with variable reactivity to w potential
	"""
	print()
	algebra = Algebra.from_str('w+x+y+t-')
	shape = (2, 256, 256)
	steps = 512
	field = FieldSlice.from_subspace(algebra.subspace.even_grade(), shape)

	field = field.random_gaussian([0.3], [0, 0, 0.1])
	field = filter_lightlike(field)
	# grid = field.meshgrid()
	# for i in range(2):
	# 	field.arr = field.arr * grid[1+i]
	# field.arr = field.arr * grid[2]
	dimple = (1-field.gauss(np.array([1e16, 0.6, 0.6]))*0.5)
	# metric = {'t': dimple}
	metric = {'w': dimple / 4}

	# field = filter_lightlike(field)
	bivecs = ['wt_xt_yt', 'wx_wy_wt']
	for bv in bivecs:
		field.write_gif_3d_generator(
			field.rollout_generator(steps, metric=metric),
			basepath='../../output', components=bv, pre='',
		)


def test_3d_odd():
	"""does not seem like odd has non-propagating parts"""
	print()
	algebra = Algebra.from_str('w+x+y+t-')
	shape = (2, 128, 128)
	steps = 256
	field = FieldSlice.from_subspace(algebra.subspace.odd_grade(), shape)

	# field = field.random_gaussian([1e16, 0.1, 0.1], [0, 0, 0.0])
	grid = field.meshgrid()
	# for i in range(1)
	# field.arr = field.arr * grid[2]
	dimple = (1-field.gauss(np.array([1e16, 0.3, 0.3]))*0.5)
	# metric = {'t': dimple}
	metric = {'w': dimple / 2}
	# metric = {}

	for n in range(2):
		field = field.random_gaussian(0.1, seed=0)
		field.arr = field.arr * grid[1] * grid[2]
		field = filter_stationary(field, n, metric=metric)
		# field = filter_lightlike(field)
		full_field = field.rollout(steps, metric=metric)
		full_field.write_gif_3d(basepath='../../output', components='x_y_w', pre=f'power_{n}')


def test_3d_compact_generations():
	"""
	"""
	print()
	algebra = Algebra.from_str('w+x+y+t-')
	shape = (2, 128, 128)
	steps = 256
	field = FieldSlice.from_subspace(algebra.subspace.even_grade(), shape)
	dimple = (1-field.gauss(jnp.array([1, 0.3, 0.3]))*0.7)
	metric = {'w': dimple*0.5}

	for gen in range(4):
		field = field.random_gaussian(0.1, 0.1)
		field = filter_stationary(field, gen, metric=metric)
		# suppress lightlike components from initial field conditions
		# for i in range(gen):
		# 	field = field.rollout(2, metric=metric).slice(-1)
		# 	field = field.geometric_derivative_leapfrog(metric=metric)
			# field.arr = field.arr - field.arr#.mean(axis=1, keepdims=True)
		# field = field.rollout(3, metric=metric).slice(-1)
		# field.arr = field.arr - field.arr.mean(axis=1, keepdims=True)

		# metric['t'] = gen
		full_field = field.rollout(steps, metric=metric)
		full_field.write_gif_3d(basepath='../../output', components='xw_yw_xy', pre=f'gen{gen}')


def test_3d_compact_filtered():
	"""
	note that a t-potential does not radiate
	but a w potential does radiate
	"""
	print()
	algebra = Algebra.from_str('w+x+y+t-')
	shape = (2, 128, 128)
	steps = 256
	field = FieldSlice.from_subspace(algebra.subspace.even_grade(), shape)
	dimple = (1-field.gauss(jnp.array([1, 0.3, 0.3]))*0.9)
	metric = {'t': dimple}
	# metric = {'w': dimple / 2}

	for n in range(3):
		field = field.random_gaussian(0.1, 0.1)

		field = filter_stationary(field, n, metric=metric)

		full_field = field.rollout(steps, metric=metric)
		full_field.write_gif_3d(basepath='../../output', components='xw_yw_xy', pre=f'filtered{n}_t')



def test_4d_even_compact():
	print()
	algebra = Algebra.from_str('w+x+y+z+t-')
	shape = (2, 64, 64, 64)
	steps = 128
	field = FieldSlice.from_subspace(algebra.subspace.even_grade(), shape)

	field = field.random_gaussian(0.3, [0, 0, 0, 0.1])
	field = filter_lightlike(field)
	grid = field.meshgrid()
	# for i in range(3):  # change to gauss * x
	# 	field.arr = field.arr * grid[i+1]
	dimple = (1-field.gauss(jnp.array([1e16, 0.5, 0.5, 0.5]))*0.5)
	# metric = {'t': dimple}
	metric = {'w': dimple / 2}

	bivecs = ['xw_yw_zw', 'xt_yt_zt', 'xy_yz_zx']   # bunch to choose from
	for bivec in bivecs:
		field.write_gif_4d_generator_compact(
			field.rollout_generator(steps, metric=metric),
			basepath='../../output', components=bivec, pre='masslike4',
		)

