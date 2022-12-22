from typing import Iterator

import jax
import numpy as np
from jax import numpy as jnp
from numga.backend.jax.pytree import register

from discrete.field_slice import AbstractFieldSlice
from discrete.jax.field import Field


@register
class FieldSlice(Field, AbstractFieldSlice):
	"""Field with one axis that is stepped over, rather than allocated"""
	def __init__(self, subspace, domain, arr):
		super(AbstractFieldSlice, self).__init__(subspace, domain)
		self.arr = arr
		assert len(arr) == len(subspace)
		assert len(self.shape) == len(domain) - 1
		# FIXME: currently only works over last axis named t
		assert self.algebra.description.basis_names[-1] == 't'

	def geometric_derivative_leapfrog(self, mass=None, metric={}) -> "FieldSlice":
		op = self.algebra.operator.geometric_product(self.domain, self.subspace)
		partial = self.partial_term(metric)
		T, S = self.process_op_leapfrog(op)
		arr = self.arr.copy()

		for eq_idx, (t_term, s_terms) in T + S:
			total = sum(partial(arr, term) for term in s_terms)

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

	def rollout_generator(self, steps, unroll=1, metric={}, **kwargs) -> Iterator["FieldSlice"]:
		"""perform a rollout of a field slice as a generator"""
		unroll, metric = self.cfl(unroll, metric, kwargs)

		@jax.jit
		def step(state):
			def inner(_, field):
				return field.geometric_derivative_leapfrog(metric=metric, **kwargs)
			return jax.lax.fori_loop(0, unroll, inner, state)

		step(self)  # timing warmup
		import time
		tt = time.time()

		field = self
		for t in range(steps):
			yield field
			field = step(field)

		print('time: ', time.time() - tt)
