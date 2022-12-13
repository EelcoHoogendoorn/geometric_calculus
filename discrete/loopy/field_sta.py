from functools import cached_property

from discrete.field import AbstractSpaceTimeField
from discrete.loopy.field import Field, Operator

import loopy as lp


class STAOperator(Operator):

	def build_kernel(self, halfstep):
		"""build loopy kernel from string syntax. prob cleaner to use the API, but too lazy to figure it out"""
		# strip t from indexing axes
		*domain, t = self.field.domain.named_str.split(',')
		assert t is 't'

		axes = ','.join(domain)
		limits = ' and '.join(f'0 <= {d} < N{d}' for d in domain)
		sign = {-1: '-', +1: '+'}

		loopy_term = self.loopy_term(domain)
		# FIXME: allow dynamic extension by fusing other stuff onto statements

		line = '{lhs} = {lhs} {s} ({rhs})'
		statements = [
			line.format(
				lhs=f'R[{tt.f_idx}, {axes}]',
				s=sign[tt.sign],
				rhs='  '.join(loopy_term(t) for t in ts))
			for eq_idx, (tt, ts) in halfstep
		]
		print("{[" + axes + "]:" + limits + "}")
		print('\n'.join(statements))

		knl = lp.make_kernel(
			"{[" + axes + "]:" + limits + "}",
			'\n'.join(statements)
		)

		# FIXME: make this stuff configurable
		for i, d in zip(range(1), domain[::-1]):
			knl = lp.split_iname(knl, d, 16, outer_tag=f"g.{i}", inner_tag=f"l.{i}")
		# assume grid size and thread block size match
		# for d in domain:
		# 	knl = lp.assume(knl, f'N{d} mod 16 = 0')

		knl = lp.set_options(knl, write_cl=True)

		return knl

	@cached_property
	def build_kernels(self):
		"""build a pair of leapfrog kernels"""
		return [self.build_kernel(hs) for hs in self.field.process_op_leapfrog(self.op)]

	def apply(self):
		"""Note: we use a pair of kernels placed sequentially in queue now;
		probably its superior to have a single kernel with a global sync in the middle,
		but I suppose that only starts paying off when we prefetch into local memory anyway"""
		for halfstep in self.build_kernels:
			halfstep(R=self.field.arr, queue=self.context.queue)
		return self


class SpaceTimeField(Field, AbstractSpaceTimeField):

	""""""
	def __init__(self, context, subspace, domain, arr):
		super(AbstractSpaceTimeField, self).__init__(subspace, domain)
		self.context = context
		self.arr = arr
		self.shape = arr.shape[1:]
		assert len(arr) == len(subspace)
		assert len(self.shape) == len(context.spatial_domain)

	def geometric_derivative(self, mass=None, metric=None):
		op = self.algebra.operator.geometric_product(self.domain, self.subspace)
		return STAOperator(self.context, self, op)
