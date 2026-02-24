from numga.algebra.algebra import Algebra

from continuous.field import Field


class Geometry:
	def __init__(self, algebra, constant=None):
		if isinstance(algebra, int):
			algebra = Algebra.from_pqr(algebra, 0, 0)
		if isinstance(algebra, tuple):
			algebra = Algebra.from_pqr(*algebra)
		if isinstance(algebra, str):
			algebra = Algebra.from_str(*algebra)
		self.algebra = algebra
		self.domain = self.algebra.subspace.vector()
		if constant:
			self.domain = self.domain.difference(constant)

	def field(self, f, subspace):
		return Field(f=f, subspace=subspace, geometry=self)
	def k_field(self, f, k):
		return self.field(f=f, subspace=self.algebra.subspace.k_vector(k))
	def rotor_field(self, f):
		return self.field(f=f, subspace=self.algebra.subspace.even_grade())

