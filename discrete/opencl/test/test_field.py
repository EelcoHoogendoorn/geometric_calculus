from discrete.opencl.context import Context


def test_vector_wave_2d():
	"""second order vector wave equation"""
	# path = r'../../output/loopy_wave_scalar_2d'
	# import imageio.v3 as iio

	from numga.algebra.algebra import Algebra
	algebra = Algebra.from_str('x+y+')
	# algebra = Algebra.from_str('x+')
	shape = (32, ) * algebra.n_dimensions
	context = Context(algebra)
	field = context.make_field(algebra.subspace.vector(), shape)
	gd = field.geometric_derivative()
	print()

	r = gd.apply()
	print(r.arr.shape)
