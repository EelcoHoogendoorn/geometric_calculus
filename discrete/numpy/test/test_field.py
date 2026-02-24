from numga.algebra.algebra import Algebra
from discrete.numpy.field import Field


def test_xt():
	print()
	algebra = Algebra.from_str('x+t-')
	shape = (256, 256)
	field = Field.from_subspace(algebra.subspace.multivector(), shape)

	print(field.geometric_to_str())