"""JAX based discrete calculus implementation
"""

import jax.numpy as jnp
from numga.backend.jax.pytree import register

from discrete.field import AbstractField


@register
class Field(AbstractField):
	__pytree_ignore__ = ('subspace', 'domain', 'shape', 'algebra')

	def __init__(self, subspace, domain, arr):
		super(Field, self).__init__(subspace, domain)
		self.arr = arr
		assert len(arr) == len(subspace)
		assert len(self.shape) == len(self.domain)

	@classmethod
	def from_subspace(cls, subspace, shape, domain=None):
		return cls(
			subspace=subspace,
			domain=subspace.algebra.subspace.vector() if domain is None else domain,
			arr=jnp.zeros((len(subspace),)+shape)
		)

	def __getattr__(self, item):
		subspace = getattr(self.algebra.subspace, item)
		idx = self.subspace.to_relative_indices(subspace.blades)
		return self.arr[idx]
	def set(self, key, value):
		subspace = self.algebra.subspace.from_str(key)
		idx = self.subspace.relative_indices(subspace)
		self.arr = self.arr.at[idx].set(value)

	def dual(self):
		# FIXME: create unary op for this?
		op = self.algebra.operator.dual(self.subspace)
		arr = jnp.einsum('oc,c...->o...', op.kernel, self.arr)
		return self.copy(subspace=op.subspace, arr=arr)
	def reverse(self):
		# FIXME: create unary op for this?
		op = self.algebra.operator.reverse(self.subspace)
		arr = jnp.einsum('oc,c...->o...', op.kernel, self.arr)
		return self.copy(subspace=op.subspace, arr=arr)

	def copy(self, arr=None, subspace=None):
		return type(self)(
			subspace=self.subspace if subspace is None else subspace,
			domain=self.domain,
			arr=arr.copy() if arr is None else jnp.array(arr)
		)
	def __add__(self, other):
		if isinstance(other, type(self)):
			assert self.subspace == other.subspace
			return self.copy(self.arr + other.arr)
		return self.copy(self.arr + other)
	def __sub__(self, other):
		if isinstance(other, type(self)):
			assert self.subspace == other.subspace
			return self.copy(self.arr - other.arr)
		return self.copy(self.arr - other)
	def __mul__(self, other):
		if isinstance(other, type(self)):
			# user discretion is advised!
			# assert self.subspace == other.subspace
			return self.copy(self.arr * other.arr)
		return self.copy(self.arr * other)
	def __truediv__(self, other):
		if isinstance(other, type(self)):
			assert other.subspace.equals.scalar()
			return self.copy(self.arr / other.arr)
		return self.copy(self.arr / other)

	def restrict(self, subspace):
		op = self.subspace.algebra.operator.restrict(self.subspace, subspace)
		return self.copy(subspace=subspace, arr=jnp.einsum('io,i...->o...', op.kernel, self.arr))

	def make_partial_derivatives(self, metric):
		"""Construct upwind and downwind (non-contracting and contracting)
		derivative operators along each dimension

		NOTE: these are derivatives, not including +- signs obtained from the product,
		which enter elsewhere

		NOTE: this construction assumes the t axis is last
		"""
		domain = self.domain.named_str.split(',')
		def ed(x, a):
			f = x * metric.get(domain[a], 1)
			return jnp.roll(f, shift=-1, axis=a) - f
		def id(x, a):
			f = x * metric.get(domain[a], 1)
			return f - jnp.roll(f, shift=+1, axis=a)
		return [ed, id]
		# return [
		# 	lambda x, a: (jnp.roll(x, shift=-1, axis=a) - x) * metric.get(domain[a], 1),
		# 	lambda x, a: (x - jnp.roll(x, shift=+1, axis=a)) * metric.get(domain[a], 1),
		# ]

	def partial_term(self, metric):
		"""Construct partial derivative for a term in a geometric-algebraic product"""
		partial = self.make_partial_derivatives(metric)
		return lambda f, t: partial[t.contraction](f[t.f_idx], t.d_idx) * t.sign

	def make_derivative(self, op, metric):
		"""an arbitrary vector-derivative operation, no leapfrog"""
		partial = self.partial_term(metric)
		arr = jnp.zeros(shape=(len(op.subspace),) + self.shape)
		for eq_idx, eq in self.process_op(op):
			total = sum(partial(self.arr, term) for term in eq)
			arr = arr.at[eq_idx].set(total)
		return self.copy(arr=arr, subspace=op.subspace)

	def geometric_derivative(self, output=None, metric={}):
		"""vector-geometric derivative"""
		op = self.algebra.operator.geometric_product(self.domain, self.subspace)
		if output:
			op = op.select_subspace(output)
		return self.make_derivative(op, metric)
	def interior_derivative(self, output=None, metric={}):
		"""vector-interior derivative"""
		op = self.algebra.operator.inner_product(self.domain, self.subspace)
		if output:
			op = op.select_subspace(output)
		return self.make_derivative(op, metric)
	def exterior_derivative(self, output=None, metric={}):
		"""vector-exterior derivative"""
		op = self.algebra.operator.outer_product(self.domain, self.subspace)
		if output:
			op = op.select_subspace(output)
		return self.make_derivative(op, metric)

	def slice(self, idx):
		"""Take a field slice over the last axis"""
		from discrete.jax.field_slice import FieldSlice # defer import to break cirularity
		return FieldSlice(self.subspace, self.domain, self.arr[..., idx])
