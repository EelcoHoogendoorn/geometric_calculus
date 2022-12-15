"""Jax based tests
"""

import jax
import jax.numpy as jnp
import numpy as np

from numga.algebra.algebra import Algebra
from discrete.jax.field import SpaceTimeField


def test_1d():
	print()
	algebra = Algebra.from_str('x+t-')
	shape = (256,)
	steps = 256
	field = SpaceTimeField.from_subspace(algebra.subspace.multivector(), shape)
	field = field.random_gaussian(0.1)

	mass = (field.quadratic() *0 + 0.1) * 0

	full_field = field.rollout(steps, mass=mass)

	full_field.write_gif_1d('../../output', 'x_t_xt', post='mass')


def test_1d_mass():
	print()
	algebra = Algebra.from_str('x+t-')
	shape = (256,)
	steps = 256
	field = SpaceTimeField.from_subspace(algebra.subspace.multivector(), shape)

	field = field.random_gaussian(0.1)

	mass = (field.quadratic() *0 + 0.1)

	metric = {'t': (1-field.gauss(0.3)*0.5)}

	full_field = field.rollout(steps, mass=mass, metric=metric)

	full_field.write_gif_1d('../../output', 'x_t_xt', post='mass_metric')


def test_1d_massI():
	algebra = Algebra.from_str('x-t+')
	shape = (256,)
	steps = 256*4
	field = SpaceTimeField.from_subspace(algebra.subspace.multivector(), shape)
	field = field.random_gaussian(0.1, 0.1)
	mass = 0.0 #+ (1-field.gauss(0.3)) / 13
	metric = {'t': (1-field.gauss(0.3)*0.5)}
	mass_I = 0.2

	full_field = field.rollout(steps, mass_I=mass_I, metric=metric)

	full_field.write_gif_1d('../../output', 'x_t_xt', post='mass_I')


def test_1d_mass_sig():
	"""attempted deep dive into flipped sigs
	mass term appears broken? or is this sig just to blame?
	"""
	print()
	algebra = Algebra.from_str('x-t+')
	shape = (256,)
	steps = 256

	field = SpaceTimeField.from_subspace(algebra.subspace.multivector(), shape)
	field = field.random_gaussian(0.1)
	mass = -0.3 #+ (1 - jnp.exp(-x2 * 3)) / 13

	metric = {'t': (1-field.gauss(0.3)) * 0.5}

	full_field = field.rollout(steps, mass=mass, metric=metric)

	full_field.write_gif_1d('../../output', 'x_t_xt', post='sig_mass')



def test_2d():
	print()
	algebra = Algebra.from_str('x+y+t-')
	shape = (128, 128)
	steps = 256
	field = SpaceTimeField.from_subspace(algebra.subspace.multivector(), shape)
	field = field.random_gaussian(0.1)
	mass = 0.4 + field.quadratic() / 2# + 0.1

	# metric = {'t': (1-field.gauss(0.3)*0.5)}

	full_field = field.rollout(steps, mass=mass, metric={})

	full_field.write_gif_2d('../../output', 'xt_yt_xy', post='mass', norm=99)


def test_2d_1vec():
	"""note: like all equations over a non-closed subspace, this thing has non-propagating residual,
	because we are too lazy to implement a compatible initalization yet"""
	print()
	algebra = Algebra.from_str('x+y+t-')
	shape = (128, 128)
	steps = 128
	field = SpaceTimeField.from_subspace(algebra.subspace.vector(), shape)

	field = field.random_gaussian(0.1)

	full_field = field.rollout(steps, metric={})

	full_field.write_gif_2d('../../output', 'x_y_t', post='mass')


def test_2d_compact():
	print()
	algebra = Algebra.from_str('w+y+t-')
	shape = (2, 256)
	steps = 256*2
	field = SpaceTimeField.from_subspace(algebra.subspace.multivector(), shape)
	field = field.random_gaussian(0.1, 0.1)
	# field = field.smooth_noise([1, 8])
	dimple = (1-field.gauss(jnp.array([1e16, 0.3]))*0.3)
	metric = {'w': dimple * 0.3}

	full_field = field.rollout(steps, metric=metric, mass=0.2)
	full_field.write_gif_2d_compact('../../output', 'wy_wt_yt', pre='', norm=99)




def test_2d_compact_sig():
	"""in 2d, direct mass term with flipped sig also seems broken"""
	print()
	algebra = Algebra.from_str('x-y-t+')
	shape = (2, 256)
	steps = 256
	field = SpaceTimeField.from_subspace(algebra.subspace.full(), shape)
	field = field.random_gaussian(0.1, 0.1)
	dimple = (1-field.gauss(jnp.array([1e16, 0.3]))*0.3)
	metric = {'x': dimple * 0.2}
	mass = 0.2

	full_field = field.rollout(steps, metric=metric, mass=mass)

	full_field.write_gif_2d_compact('../../output', 'xy_xt_yt', post='compact_sig')



def test_3d():
	print()
	algebra = Algebra.from_str('x+y+z+t-')
	steps = 64
	shape = (64, 64, 64)
	field = SpaceTimeField.from_subspace(algebra.subspace.multivector(), shape)

	field = field.random_gaussian(0.1, 0.1)
	mass = 0.2 + field.quadratic() / 4
	metric = {}

	full_field = field.rollout(steps, metric=metric, mass=mass)
	full_field.write_gif_3d('../../output', 'xy_xt_yt', post='', norm=99)


def filter_stationary(field, n=1, mean_axis=None, **kwargs):
	# filter field to a state without non-propagating components
	for i in range(n):
		correction = field.rollout(1, **kwargs).slice(-1).arr
		if not mean_axis is None:
			correction = correction.mean(mean_axis, keepdims=True)
		field.arr = field.arr - correction
		# field = field.geometric_derivative_leapfrog(metric=metric)
		# field.arr = field.arr - field.arr.mean(axis=1, keepdims=True)
	return field


def test_3d_bivector():
	print()
	algebra = Algebra.from_str('x+y+z+t-')
	shape = (64, 64, 64)
	steps = 64
	field = SpaceTimeField.from_subspace(algebra.subspace.bivector(), shape)
	field = field.random_gaussian(0.1)

	field = filter_stationary(field)

	full_field = field.rollout(steps)
	full_field.write_gif_3d('../../output', 'xt_yt_zt', pre='filtered', norm=99)


def test_3d_even_compact():
	print()
	algebra = Algebra.from_str('x+y+z+t-')
	shape = (2, 128, 128)
	steps = 128
	field = SpaceTimeField.from_subspace(algebra.subspace.even_grade(), shape)

	field = field.random_gaussian(0.1)
	mass = 0.2 +0* field.quadratic() / 4# + 0.1
	mass_I = -0.2

	metric = {'x': mass}

	full_field = field.rollout(steps, metric=metric, mass_I=mass_I)
	full_field.write_gif_3d('../../output', 'xt_yt_zt', post='mass_I', norm=99)


def test_3d_compact_generations():
	"""wow; we can make subluminal excitations,
	unaffected by compact metric dimple
	"""
	print()
	algebra = Algebra.from_str('w+x+y+t-')
	shape = (2, 128, 128)
	steps = 256
	field = SpaceTimeField.from_subspace(algebra.subspace.even_grade(), shape)
	dimple = (1-field.gauss(jnp.array([1, 0.3, 0.3]))*0.7)
	metric = {'w': dimple*0.5}

	for gen in range(4):
		field = field.random_gaussian(0.1, 0.1)
		# suppress lightlike components from initial field conditions
		for i in range(gen):
			field = field.rollout(1, metric=metric).slice(-1)
			# field = field.geometric_derivative_leapfrog(metric=metric)
			field.arr = field.arr - field.arr.mean(axis=1, keepdims=True)
		# field = field.rollout(3, metric=metric).slice(-1)
		# field.arr = field.arr - field.arr.mean(axis=1, keepdims=True)

		# metric['t'] = gen
		full_field = field.rollout(steps, metric=metric, unroll=8**gen)
		full_field.write_gif_3d('../../output', 'xw_yw_xy', pre=f'gen{gen}', norm=99)


def test_3d_compact_filtered():
	"""wow; we can make subluminal excitations,
	unaffected by compact metric dimple
	can we view this as a model for Z-bosons?
	succesive generations of excitations constructed in this manner travel more slowly;
	both its radiating component unbound by the potential,
	as well as its massive component

	note that a t-potential does not radiate
	but a w potential does radiate
	"""
	print()
	algebra = Algebra.from_str('w+x+y+t-')
	shape = (2, 128, 128)
	steps = 256
	field = SpaceTimeField.from_subspace(algebra.subspace.even_grade(), shape)
	dimple = (1-field.gauss(jnp.array([1, 0.3, 0.3]))*0.9)
	metric = {'t': dimple}
	# metric = {'w': dimple / 2}

	for n in range(3):
		field = field.random_gaussian(0.1, 0.1)

		field = filter_stationary(field, n, metric=metric)

		full_field = field.rollout(steps, metric=metric,unroll=8**n)
		full_field.write_gif_3d('../../output', 'xw_yw_xy', pre=f'filtered{n}_t', norm=99)

