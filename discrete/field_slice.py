from typing import Iterator

import numpy as np

import discrete.util
from discrete.field import AbstractField
from discrete.util import split


class AbstractFieldSlice(AbstractField):
	"""Field with special axis t, where we assume a full field over t is never allocated,
	but rather traversed by timestepping"""

	def process_op_leapfrog(self, op):
		"""Preprocess the terms in the derivative operator,
		to be consumed by a leapfrog timestepping scheme"""
		output = op.subspace.named_str.split(',')
		domain = self.domain.named_str.split(',')
		is_spacelike_gradient = lambda term: not 't' is domain[term.d_idx]
		is_spacelike_equation = lambda eq: not 't' in output[eq[0]]

		# split equations into terms of timelike and spacelike partial derivatives
		eqs = [(i, split(eq, is_spacelike_gradient)) for i, eq in self.process_op(op)]

		eqs = [
			# pull t term to the other side of equality
			(i, (tt[0]._replace(sign=-tt[0].sign), ts))
			for i, (tt, ts) in eqs
			# drop equations that do not contribute a leapfrog update
			if len(tt) == 1
		]
		# split into equations defined on spacelike and timelike elements
		return split(eqs, is_spacelike_equation)

	def generate(self, op) -> str:
		"""Text representation of a leapfrog geometric derivative operator"""
		equation = op.subspace.named_str.replace('1', 's').split(',')
		term_to_str = self.term_to_str()
		T, S = self.process_op_leapfrog(op)
		return '\n'.join([
			f'{term_to_str(tt)} = ' + ''.join([term_to_str(t) for t in ts]) + f'\t # {equation[eq_idx]}'
			for eq_idx, (tt, ts) in T + S
		])

	def generate_geometric(self) -> str:
		return self.generate(self.algebra.operator.geometric_product(self.domain, self.subspace))



	def rollout(self) -> AbstractField:
		raise NotImplemented

	def rollout_generator(self) -> Iterator["AbstractFieldSlice"]:
		raise NotImplemented



	def write_animation(self, selector, generator, basepath, components, **kwargs):
		def get_components(f, components):
			"""sample field components as a numpy array, with field components last, for rendering"""
			return np.moveaxis(getattr(f, components), 0, -1)
		def to_animation(generator, components):
			return np.array([selector(get_components(f, components)) for f in generator])
		image = to_animation(generator, components)
		self.write_animation_base(image, basepath, components, **kwargs)

	def write_gif_2d_generator(self, *args, **kwargs):
		def selector(arr):
			return np.abs(arr)
		self.write_animation(selector, *args, **kwargs)

	def write_gif_3d_generator(self, *args, **kwargs):
		def selector(arr):
			mid = lambda a: a[a.shape[0] // 2]
			return mid(np.abs(arr))
		self.write_animation(selector, *args, **kwargs)

	def write_gif_3d_generator_compact(self, *args, **kwargs):
		def selector(arr):
			mean = lambda a: a.mean(0)
			return mean(np.abs(arr))
		self.write_animation(selector, *args, **kwargs)

	def write_gif_4d_generator_compact(self, *args, **kwargs):
		def selector(arr):
			mean = lambda a: a.mean(0)
			mid = lambda a: a[a.shape[0] // 2]
			return mid(mean(np.abs(arr)))
		self.write_animation(selector, *args, **kwargs)
