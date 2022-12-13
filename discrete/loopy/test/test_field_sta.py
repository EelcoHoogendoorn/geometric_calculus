from discrete.loopy.context import SpaceTimeContext


def test_sta_2d():
	# path = r'../../output/loopy_wave_scalar_2d'
	# import imageio.v3 as iio

	from numga.algebra.algebra import Algebra
	algebra = Algebra.from_str('x+t-')
	shape = (32, ) * (algebra.n_dimensions -1)
	context = SpaceTimeContext(algebra)
	field = context.make_field(algebra.subspace.full(), shape)
	foo = field.geometric_derivative()
	print()
	for k in foo.build_kernels:
		print(k)

	print(foo.apply())