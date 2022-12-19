from numga.algebra.algebra import Algebra
from discrete.loopy.context import SpaceTimeContext
import numpy as np


def test_2d():
	print()
	algebra = Algebra.from_str('x+y+t-')
	shape = (64, 64)
	steps = 128
	context = SpaceTimeContext(algebra)
	field = context.make_field(algebra.subspace.even_grade(), shape)

	field = field.random_gaussian(0.1)
	# mass = 0.4 + field.quadratic() / 2# + 0.1

	field.write_gif_2d_generator(
		field.rollout_generator(steps),
		basepath='../../output', components='xt_yt_xy', pre='opencl',
	)


def test_3d_compact():
	print()
	algebra = Algebra.from_str('w+x+y+t-')
	shape = (2, 128, 128)
	steps = 256
	context = SpaceTimeContext(algebra)
	field = context.make_field(algebra.subspace.even_grade(), shape)

	field = field.random_gaussian(0.1)
	# mass = 0.4 + field.quadratic() / 2# + 0.1

	field.write_gif_3d_generator_compact(
		field.rollout_generator(steps),
		basepath='../../output', components='xt_yt_xy', pre='opencl'
	)


def test_4d_even_compact():
	print()
	algebra = Algebra.from_str('w+x+y+z+t-')
	shape = (2, 64, 64, 64)
	steps = 128
	context = SpaceTimeContext(algebra)
	field = context.make_field(algebra.subspace.even_grade(), shape)

	field = field.random_gaussian(0.1)
	dimple = (1-field.gauss(np.array([1e16, 0.3, 0.3, 0.3]))*0.9)
	# metric = {'t': dimple}
	metric = {'w': dimple / 2}

	field.write_gif_4d_generator_compact(
		field.rollout_generator(steps),
		basepath='../../output', components='xt_yt_xy', pre='opencl',
	)
