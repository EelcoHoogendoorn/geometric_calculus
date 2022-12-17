from numga.algebra.algebra import Algebra
from discrete.loopy.context import SpaceTimeContext
import numpy as np

def test_sta_2d():
	# path = r'../../output/loopy_wave_scalar_2d'
	# import imageio.v3 as iio

	algebra = Algebra.from_str('x+y+t-')
	shape = (32, 32) * (algebra.n_dimensions - 1)
	context = SpaceTimeContext(algebra)
	field = context.make_field(algebra.subspace.full(), shape)
	foo = field.geometric_derivative()
	print()
	for k in foo.build_kernels:
		print(k)

	print(foo.apply())


def test_2d():
	print()
	algebra = Algebra.from_str('x+y+t-')
	shape = (512, 512)
	steps = 128
	context = SpaceTimeContext(algebra)
	field = context.make_field(algebra.subspace.full(), shape)

	field = field.random_gaussian(0.1)
	# mass = 0.4 + field.quadratic() / 2# + 0.1

	full_field = field.rollout(steps)
	full_field.arr = full_field.arr.map_to_host(queue=context.queue)
	full_field.write_gif_2d('../../output', 'xt_yt_xy', pre='opencl', norm=99, anim=False)


def test_4d_even_compact():
	print()
	algebra = Algebra.from_str('w+x+y+z+t-')
	shape = (2, 64, 64, 64)
	steps = 128
	context = SpaceTimeContext(algebra)
	field = context.make_field(algebra.subspace.even_grade(), shape)


	field = field.random_gaussian(0.1)
	grid = field.meshgrid()
	# field.arr = field.arr * grid[1]
	# field.arr = field.arr * grid[2]
	# mass = 0.2 +0* field.quadratic() / 4# + 0.1
	# mass_I = -0.2
	dimple = (1-field.gauss(np.array([1e16, 0.3, 0.3, 0.3]))*0.9)
	# metric = {'t': dimple}
	metric = {'w': dimple / 2}

	full_field = field.rollout(steps)
	full_field.arr = full_field.arr.map_to_host(queue=context.queue)

	full_field.write_gif_4d_compact('../../output', 'xt_yt_zt', pre='opencl', norm=99)

