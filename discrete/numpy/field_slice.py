from typing import Iterator

import numpy as np

from discrete.field_slice import AbstractFieldSlice
from discrete.numpy.field import Field


class FieldSlice(Field, AbstractFieldSlice):
	"""Field with one axis that is stepped over, rather than allocated"""
	def __init__(self, subspace, domain, arr):
		super(AbstractFieldSlice, self).__init__(subspace, domain)
		self.arr = arr
		assert len(arr) == len(subspace)
		assert len(self.shape) == len(domain) - 1
		# FIXME: currently only works over last axis named t
		assert self.algebra.description.basis_names[-1] == 't'

	def geometric_derivative_leapfrog_inplace(self, mass=None, metric={}) -> "FieldSlice":
		op = self.algebra.operator.geometric_product(self.domain, self.subspace)
		partial = self.partial_term(metric)
		T, S = self.process_op_leapfrog(op)

		for eq_idx, (t_term, s_terms) in T + S:
			total = sum(partial(self.arr, term) for term in s_terms)

			if not mass is None:
				# 'direct' mass term; proportional to element being stepped over, or eq
				# FIXME: assure ourselves this element exists in self.subspace
				#  if implicit zero just skip?
				assert self.subspace == op.subspace
				total += self.arr[eq_idx] * mass

			self.arr[t_term.f_idx] += (total * metric.get('t', 1) * t_term.sign)

		return self

	def rollout(self, steps, **kwargs) -> Field:
		"""perform a rollout of a field slice into a whole field"""
		output = Field.from_subspace(self.subspace, self.shape + (steps,))
		arr = np.asfortranarray(output.arr) # more contiguous if t is last

		for t, field in enumerate(self.rollout_generator(steps, **kwargs)):
			arr[..., t] = field.arr

		output.arr = np.array(arr)
		return output

	def rollout_generator(self, steps, unroll=1, metric={}, **kwargs) -> Iterator["FieldSlice"]:
		"""perform a rollout of a field slice as a generator"""
		unroll, metric = self.cfl(unroll, metric, kwargs)

		import time
		tt = time.time()

		field = self.copy()    # lets not mutate self so our interface isnt more mutable than it needs to be
		for t in range(steps):
			yield field
			for i in range(unroll):
				field.geometric_derivative_leapfrog_inplace(**kwargs, metric=metric)

		print('time: ', time.time() - tt)


	def operator_to_str(self, op) -> str:
		"""Text representation of a leapfrog geometric derivative operator in numpy syntax"""
		term_to_str = self.term_to_str()
		equation = op.subspace.named_str.replace('1', 's').split(',')
		def line(eq):
			eq_idx, (tt, ts) = eq
			s, *r, _ = term_to_str(tt)
			r = ''.join(r)
			rhs = ''.join([term_to_str(t) for t in ts])
			return f'{r}, {s}({rhs}))' + f'\t # {equation[eq_idx]}'
		T, S = self.process_op_leapfrog(op)
		return '\n'.join(line(l) for l in T + S)
