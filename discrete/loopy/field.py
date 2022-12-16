from functools import cached_property

import loopy as lp

import numpy as np
from discrete.field import AbstractField


class Operator:
	def __init__(self, context, field, op):
		self.context = context
		self.field = field
		self.op = op

	def loopy_term(self, domain):
		"""return callable to transform abstract term into loopy term"""
		sign = {-1: '-', +1: '+'}
		axes = ','.join(domain)

		def idx(d_idx, dir):
			q = [f'({d}{dir})&(N{d}-1)' if i == d_idx else d for i, d in enumerate(domain)]
			return ','.join(q)

		def id(v, c, d) -> str:
			return f'{v}(R[{c}, {axes}] - R[{c}, {idx(d, "-1")}])'

		def ed(v, c, d) -> str:
			return f'{v}(R[{c}, {idx(d, "+1")}] - R[{c}, {axes}])'

		d = [ed, id]

		def make_term(t) -> str:
			return d[t.contraction](sign[t.sign], t.f_idx, t.d_idx)
		return make_term


	@cached_property
	def build_kernel(self):
		"""build loopy kernel from string syntax. prob cleaner to use the API, but too lazy to figure it out"""
		domain = self.field.domain.named_str.split(',')

		axes = ','.join(domain)
		limits = ' and '.join(f'0 <= {d} < N{d}' for d in domain)

		loopy_term = self.loopy_term(domain)
		# FIXME: allow dynamic extension by fusing other stuff onto statements
		statements = [
			f'W[{eq_idx}, {axes}] = ' + '  '.join(loopy_term(t) for t in eq)
			for eq_idx, eq in self.field.process_op(self.op)
		]
		# print("{[" + axes + "]:" + limits + "}")
		# print('\n'.join(statements))

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

	def apply(self):
		output = self.context.make_field(self.op.subspace, self.field.shape)
		self.build_kernel(W=output.arr, R=self.field.arr, queue=self.context.queue)
		return output


class Field(AbstractField):
	def __init__(self, context, subspace, domain, arr):
		super(Field, self).__init__(subspace, domain)
		self.context = context
		self.arr = self.context.coerce_array(arr)
		assert len(arr) == len(subspace)
		assert len(self.shape) == len(self.domain)

	def __hash__(self):
		return self.subspace, self.domain, self.shape

	# @classmethod
	# def from_subspace(cls, subspace, shape, domain=None):
	# 	return cls(
	# 		subspace=subspace,
	# 		domain=subspace.algebra.subspace.vector() if domain is None else domain,
	# 		arr=cla.zeros((len(subspace),)+shape)
	# 	)

	def __getattr__(self, item):
		subspace = getattr(self.algebra.subspace, item)
		idx = self.subspace.to_relative_indices(subspace.blades)
		return self.arr[idx]
	def set(self, key, value):
		subspace = self.algebra.subspace.from_str(key)
		idx = self.subspace.relative_indices(subspace)
		self.arr = self.arr[idx].set(value)

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
			context=self.context,
			subspace=self.subspace if subspace is None else subspace,
			domain=self.domain,
			arr=arr.copy() if arr is None else arr
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
		return self.copy(subspace=subspace, arr=np.einsum('io,i...->o...', op.kernel, self.arr))

	def geometric_derivative(self, output=None):
		op = self.algebra.operator.geometric_product(self.domain, self.subspace)
		if output:
			op = op.select_subspace(output)
		return Operator(self.context, self, op)

