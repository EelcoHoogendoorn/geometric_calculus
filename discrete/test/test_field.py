"""These test can be ran and adapted, to check a textual human readable output,
of the generated geometric operators"""
from numga.algebra.algebra import Algebra

from discrete.field import AbstractField
from discrete.field_slice import AbstractFieldSlice


def test_generate_11():
	algebra = Algebra.from_str('x+t-')

	field = AbstractField(algebra.subspace.full())
	print()
	print(field.generate_geometric())


def test_generate_30():
	algebra = Algebra.from_str('x+y+z+')
	field = AbstractField(algebra.subspace.bivector())
	print()
	print(field.generate_interior())


def test_generate_sta_11():
	algebra = Algebra.from_str('x+t-')
	field = AbstractFieldSlice(algebra.subspace.full())
	print()
	print(field.generate_geometric())


def test_generate_sta_31():
	algebra = Algebra.from_str('x+y+z+t-')
	field = AbstractFieldSlice(algebra.subspace.even_grade())
	print()
	print(field.generate_geometric())

def test_generate_sta_21():
	algebra = Algebra.from_str('w+x+t-')
	field = AbstractFieldSlice(algebra.subspace.full())
	print()
	print(field.generate_geometric())
	print(field.subspace)


def test_generate_TE():
	algebra = Algebra.from_str('x+y+z+t-')
	field = AbstractField(
		algebra.subspace.xt_yt_xy,
		domain=algebra.subspace.x_y_t)
	print()
	print(field.generate_geometric())

def test_generate_TM():
	algebra = Algebra.from_str('x+y+z+t-')
	field = AbstractField(
		algebra.subspace.xz_yz_zt,
		domain=algebra.subspace.x_y_t)
	print()
	print(field.generate_geometric())


def test_generate_odd():
	algebra = Algebra.from_str('x+y+z+t-')
	field = AbstractFieldSlice(
		algebra.subspace.odd_grade())
	print()
	print(field.generate_geometric())



def test_generate_bivector():
	"""is there a clean discrete version of the bivector commutator?"""
	algebra = Algebra.from_str('w+x+y+z+t-')
	algebra = Algebra.from_str('x+y+z+t-')
	vec = algebra.subspace.vector()
	bi = algebra.subspace.bivector()
	full = algebra.subspace.full()
	even = algebra.subspace.even_grade()
	op = algebra.operator.product(vec, even)
	print()
	print(op)
	op = algebra.operator.product(bi, even)
	print()
	print(op)
	# op = algebra.operator.commutator(bi, even)
	# print()
	# print(op)
