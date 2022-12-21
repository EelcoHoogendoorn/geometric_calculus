"""JAX based discrete calculus implementation
"""
import jax
import jax.numpy as jnp
import numpy as np
from numga.backend.jax.pytree import register
from discrete.field import AbstractField, AbstractSpaceTimeField


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
		arr = np.einsum('oc,c...->o...', op.kernel, self.arr, out=self.arr)
		return self.copy(subspace=op.subspace, arr=arr)
	def reverse(self):
		# FIXME: create unary op for this?
		op = self.algebra.operator.reverse(self.subspace)
		arr = np.einsum('oc,c...->o...', op.kernel, self.arr, out=self.arr)
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
	def __mul__(self, other):
		if isinstance(other, type(self)):
			assert self.subspace == other.subspace
			return self.copy(self.arr * other.arr)
		return self.copy(self.arr * other)

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
		return [
			lambda x, a: (jnp.roll(x, shift=-1, axis=a) - x) * metric.get(domain[a], 1),
			lambda x, a: (x - jnp.roll(x, shift=+1, axis=a)) * metric.get(domain[a], 1),
		]

	def make_derivative(self, op, metric):
		"""plain direct derivative of an arbitrary GC-derivative operation, no leapfrog"""
		partial = self.make_partial_derivatives(metric)
		arr = jnp.zeros(shape=(len(op.subspace),) + self.shape)
		for eq_idx, eq in self.process_op(op):
			total = sum(
				partial[term.contraction](self.arr[term.f_idx], term.d_idx) * term.sign
				for term in eq
			)
			arr = arr.at[eq_idx, ...].set(total)
		return self.copy(arr=arr, subspace=op.subspace)

	def geometric_derivative(self, output=None, metric={}):
		op = self.algebra.operator.geometric_product(self.domain, self.subspace)
		if output:
			op = op.select_subspace(output)
		return self.make_derivative(op, metric)
	def interior_derivative(self, output=None, metric={}):
		op = self.algebra.operator.inner_product(self.domain, self.subspace)
		if output:
			op = op.select_subspace(output)
		return self.make_derivative(op, metric)
	def exterior_derivative(self, output=None, metric={}):
		op = self.algebra.operator.outer_product(self.domain, self.subspace)
		if output:
			op = op.select_subspace(output)
		return self.make_derivative(op, metric)

	def slice(self, idx):
		return SpaceTimeField(self.subspace, self.domain, self.arr[..., idx])


@register
class SpaceTimeField(Field, AbstractSpaceTimeField):
	"""Field with one axis that is stepped over, rather than allocated"""
	def __init__(self, subspace, domain, arr):
		super(AbstractSpaceTimeField, self).__init__(subspace, domain)
		self.arr = arr
		assert len(arr) == len(subspace)
		assert len(self.shape) == len(domain) - 1
		# FIXME: currently only works over last axis named t
		assert self.algebra.description.basis_names[-1] == 't'

	@property
	def dimensions(self):
		return int((np.array(self.shape) > 1).sum())

	@property
	def courant(self):
		return float(self.dimensions) ** (-0.5)

	def geometric_derivative_leapfrog(self, mass=None, metric={}):
		arr = self.arr.copy()
		op = self.algebra.operator.geometric_product(self.domain, self.subspace)

		partial = self.make_partial_derivatives(metric)

		T, S = self.process_op_leapfrog(op)
		for eq_idx, (t_term, s_terms) in T + S:
			total = sum(
				partial[term.contraction](arr[term.f_idx], term.d_idx) * term.sign
				for term in s_terms
			)
			if not mass is None:
				# 'direct' mass term; proportional to element being stepped over, or eq
				# FIXME: assure ourselves this element exists in self.subspace
				#  if implicit zero just skip?
				assert self.subspace == op.subspace
				total += arr[eq_idx] * mass

			arr = arr.at[t_term.f_idx].add(total * metric.get('t', 1) * t_term.sign)

		return self.copy(arr=arr)

	def rollout(self, steps, **kwargs) -> Field:
		"""perform a rollout of a field slice into a whole field"""
		output = Field.from_subspace(self.subspace, self.shape + (steps,))
		arr = np.asfortranarray(output.arr) # more contiguous if t is last

		for t, field in enumerate(self.rollout_generator(steps, **kwargs)):
			arr[..., t] = field.arr

		output.arr = jnp.array(arr)
		return output

	def rollout_generator(self, steps, unroll=1, metric={}, **kwargs):
		"""perform a rollout of a field slice as a generator"""
		# work safe CFL condition into metric scaling
		cfl_unroll = self.dimensions
		cfl_metric = {**metric, 't': metric.get('t', 1) / cfl_unroll}

		@jax.jit
		def step(state):
			def inner(_, field):
				return field.geometric_derivative_leapfrog(metric=cfl_metric, **kwargs)
			return jax.lax.fori_loop(0, unroll*cfl_unroll, inner, state)

		step(self)  # timing warmup
		import time
		tt = time.time()

		field = self
		for t in range(steps):
			yield field
			field = step(field)

		print('time: ', time.time() - tt)
