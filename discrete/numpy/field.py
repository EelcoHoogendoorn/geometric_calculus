"""numpy based discrete calculus implementation
"""
import numpy as np

from discrete.field import AbstractField


class Field(AbstractField):

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
			arr=np.zeros((len(subspace),)+shape)
		)

	def __getattr__(self, item):
		subspace = getattr(self.algebra.subspace, item)
		idx = self.subspace.to_relative_indices(subspace.blades)
		return self.arr[idx]
	def set(self, key, value):
		subspace = self.algebra.subspace.from_str(key)
		idx = self.subspace.relative_indices(subspace)
		self.arr[idx] = value

	def dual(self):
		# FIXME: create unary op for this?
		op = self.algebra.operator.dual(self.subspace)
		arr = np.einsum('oc,c...->o...', op.kernel, self.arr, out=self.arr)
		return self.copy(subspace=op.subspace, arr=arr)
	def reverse(self):
		# FIXME: create unary op for this?
		op = self.algebra.operator.reverse(self.subspace)
		arr = np.einsum('oc,c...->o...', op.kernel, self.arr, out=self.arr)
		return self.copy(subspace=op.subspace, arr=arr)

	def copy(self, arr=None, subspace=None) -> "Field":
		return type(self)(
			subspace=self.subspace if subspace is None else subspace,
			domain=self.domain,
			arr=self.arr.copy() if arr is None else np.array(arr)
		)
	def __add__(self, other) -> "Field":
		if isinstance(other, type(self)):
			assert self.subspace == other.subspace
			return self.copy(self.arr + other.arr)
		return self.copy(self.arr + other)
	def __mul__(self, other) -> "Field":
		if isinstance(other, type(self)):
			assert self.subspace == other.subspace
			return self.copy(self.arr * other.arr)
		return self.copy(self.arr * other)

	def restrict(self, subspace) -> "Field":
		op = self.subspace.algebra.operator.restrict(self.subspace, subspace)
		return self.copy(subspace=subspace, arr=np.einsum('io,i...->o...', op.kernel, self.arr))

	def generate(self, op) -> str:
		"""textual version of a derivative operator in numpy syntax"""
		output = op.subspace.named_str.replace('1', 's').split(',')
		term_to_str = self.term_to_str()
		return '\n'.join([
			f'{output[eq_idx]} = ' + ''.join([term_to_str(term) for term in eq])
			for eq_idx, eq in self.process_op(op)
		])

	def make_partial_derivatives(self, metric):
		"""Construct upwind and downwind (non-contracting and contracting)
		derivative operators along each dimension

		NOTE: these are derivatives, not including +- signs obtained from the product,
		which enter elsewhere

		NOTE: this construction assumes the t axis is last
		"""
		domain = self.domain.named_str.split(',')
		return [
			lambda x, a: (np.roll(x, shift=-1, axis=a) - x) * metric.get(domain[a], 1),
			lambda x, a: (x - np.roll(x, shift=+1, axis=a)) * metric.get(domain[a], 1),
		]

	def partial_term(self, metric):
		"""Construct partial derivative for a term in a geometric-algebraic product"""
		partial = self.make_partial_derivatives(metric)
		return lambda f, t: partial[t.contraction](f[t.f_idx], t.d_idx) * t.sign

	def make_derivative(self, op, metric) -> "Field":
		"""an arbitrary vector-derivative operation, no leapfrog"""
		partial = self.partial_term(metric)
		arr = np.zeros(shape=(len(op.subspace),) + self.shape)
		for eq_idx, eq in self.process_op(op):
			total = sum(partial(self.arr, term) for term in eq)
			arr[eq_idx] = total
		return self.copy(arr=arr, subspace=op.subspace)

	def geometric_derivative(self, output=None, metric={}) -> "Field":
		"""vector-geometric derivative"""
		op = self.algebra.operator.geometric_product(self.domain, self.subspace)
		if output:
			op = op.select_subspace(output)
		return self.make_derivative(op, metric)
	def interior_derivative(self, output=None, metric={}) -> "Field":
		"""vector-interior derivative"""
		op = self.algebra.operator.inner_product(self.domain, self.subspace)
		if output:
			op = op.select_subspace(output)
		return self.make_derivative(op, metric)
	def exterior_derivative(self, output=None, metric={}) -> "Field":
		"""vector-exterior derivative"""
		op = self.algebra.operator.outer_product(self.domain, self.subspace)
		if output:
			op = op.select_subspace(output)
		return self.make_derivative(op, metric)

	def slice(self, idx) -> "FieldSlice":
		"""Take a field slice over the last axis"""
		from discrete.numpy.field_slice import FieldSlice # defer import to break cirularity
		return FieldSlice(self.subspace, self.domain, self.arr[..., idx])
