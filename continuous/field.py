import jax
from jax import numpy as jnp


class Field:
	"""Differentiable multi-vector field

	f is a mapping from points in the domain to a multivector of the given subspace

	NOTE: all geometric derivatives implemented thus far are vector-geometric derivatives
	The addition of more general multi-vectorial derivatives should be trivial,
	but until we actually find a use case for those,
	this class will restrict itself to vector-geometric derivatives only.
	"""
	f: "Callable"
	subspace: "Subspace"
	geometry: "Geometry"

	def __init__(self, f, subspace, geometry):
		self.f = f
		self.subspace = subspace
		self.geometry = geometry

	@property
	def operator(self):
		return self.geometry.algebra.operator
	@property
	def domain(self):
		return self.geometry.domain

	def copy(self, f, subspace=None):
		return Field(
			f=f,
			subspace=self.subspace if subspace is None else subspace,
			geometry=self.geometry
		)
	def __add__(self, other):
		assert self.subspace == other.subspace
		return self.copy(lambda x: self.f(x) + other.f(x))
	def __sub__(self, other):
		assert self.subspace == other.subspace
		return self.copy(lambda x: self.f(x) - other.f(x))
	def __mul__(self, other):
		# FIXME: only handles scalar case now
		return self.copy(lambda x: self.f(x) * other)
	def __rmul__(self, other):
		# FIXME: only handles scalar case now
		return self.copy(lambda x: self.f(x) * other)

	def __call__(self, x):
		return self.f(x)

	def make_derivative(self, op: "Operator") -> "Field":
		def inner(x):
			# jaxrev seems faster than jaxfwd in limited testing so far
			jac = jax.jacrev(self.f)(x)
			return jnp.einsum('...ic,cio->...o', jac, op.kernel)
		return self.copy(f=inner, subspace=op.output)

	def exterior_derivative(self) -> "Field":
		op = self.operator.outer_product(self.domain, self.subspace)
		return self.make_derivative(op)
	def interior_derivative(self) -> "Field":
		op = self.operator.inner_product(self.domain, self.subspace)
		return self.make_derivative(op)
	def geometric_derivative(self, subspace=None) -> "Field":
		op = self.operator.geometric_product(self.domain, self.subspace)
		return self.make_derivative(op)
	def anti_geometric_derivative(self, subspace=None) -> "Field":
		op = self.operator.anti_geometric_product(self.domain, self.subspace)
		if not subspace is None:
			op = op.select_subspace(subspace)
		return self.make_derivative(op)
	def regressive_derivative(self) -> "Field":
		# grade-raising
		op = self.operator.regressive_product(self.domain, self.subspace)
		return self.make_derivative(op)
	def anti_exterior_derivative(self) -> "Field":
		op = self.operator.anti_outer_product(self.domain, self.subspace)
		return self.make_derivative(op)
	def anti_interior_derivative(self) -> "Field":
		op = self.operator.anti_inner_product(self.domain, self.subspace)
		return self.make_derivative(op)
	def left_contraction_derivative(self) -> "Field":
		op = self.operator.left_contraction_product(self.domain, self.subspace)
		return self.make_derivative(op)


	def make_product(self, other, op: "Operator") -> "Field":
		def inner(x):
			return jnp.einsum('...i,...j,ijo->...o', self(x), other(x), op.kernel)
		return self.copy(f=inner, subspace=op.output)
	def geometric_product(self, other) -> "Field":
		op = self.operator.geometric_product(self.subspace, other.subspace)
		return self.make_product(other, op)
	def exterior_product(self, other) -> "Field":
		op = self.operator.outer_product(self.subspace, other.subspace)
		return self.make_product(other, op)
	def interior_product(self, other) -> "Field":
		op = self.operator.inner_product(self.subspace, other.subspace)
		return self.make_product(other, op)
	def left_contraction_product(self, other) -> "Field":
		op = self.operator.left_contraction_product(self.subspace, other.subspace)
		return self.make_product(other, op)
	def right_contraction_product(self, other) -> "Field":
		op = self.operator.right_contraction_product(self.subspace, other.subspace)
		return self.make_product(other, op)

	def directional_derivative(self, other) -> "Field":
		"""Directional derivative; cartan magic formula"""
		assert self.subspace.inside.vector()    # FIXME: it 'works' for other grades; but likely unintended use?
		a: Field = self.left_contraction_product(other.exterior_derivative())
		b: Field = self.left_contraction_product(other).exterior_derivative()
		if b.subspace.equals.empty():
			return a
		if a.subspace.equals.empty():
			return b
		if a.subspace == b.subspace:
			return a + b
	# def directional_derivative(self, other):
	# 	"""https://en.wikipedia.org/wiki/Geometric_calculus"""
	# 	# FIXME: looks like an elegant definition; but why doesnt it work? should be grade preserving no?
	# 	return self.right_contraction_product(other.geometric_derivative())
	def geometric_bracket(self, other) -> "Field":
		"""7.3.2 of
		Geometric Algebra for Special Relativity and Manifold Geometry
		by Joseph Wilson"""
		return self.left_contraction_derivative().exterior_product(other) - \
		       other.left_contraction_derivative().exterior_product(self)

	def laplacian(self):
		# we can drop additional grade terms since they are zero by construction
		return self.geometric_derivative().anti_geometric_derivative(subspace=self.subspace)

	def make_unary(self, op: "Operator") -> "Field":
		def inner(x):
			return jnp.einsum('...i,io->...o', self(x), op.kernel)
		return self.copy(f=inner, subspace=op.output)
	def dual(self) -> "Field":
		op = self.operator.dual(self.subspace)
		return self.make_unary(op)
	def dual_inverse(self) -> "Field":
		op = self.operator.dual_inverse(self.subspace)
		return self.make_unary(op)

	gd = geometric_derivative
	id = interior_derivative
	ed = exterior_derivative
	dd = directional_derivative