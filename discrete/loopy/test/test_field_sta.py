from numga.algebra.algebra import Algebra
from discrete.loopy.context import SpaceTimeContext


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
	shape = (128, 128)
	steps = 256
	context = SpaceTimeContext(algebra)
	field = context.make_field(algebra.subspace.full(), shape)

	field = field.random_gaussian(0.1)
	# mass = 0.4 + field.quadratic() / 2# + 0.1

	# metric = {'t': (1-field.gauss(0.3)*0.5)}
	op = field.geometric_derivative()
	# full_field = field.rollout(steps, metric={})
	for t in range(1000):
		op.apply()
	import numpy as np
	field.arr = field.arr.map_to_host(queue=context.queue)

	field.write_gif_2d('../../output', 'xt_yt_xy', pre='opencl', norm=99)
