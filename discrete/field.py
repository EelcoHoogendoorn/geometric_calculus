import numpy as np

def split(seq, f):
	seqs = [], []
	for item in seq:
		seqs[f(item)].append(item)
	return seqs

class AbstractField:
	"""Field without specific array backing storage, just doing symbolic manipulation"""
	def __init__(self, subspace, domain=None):
		self.subspace = subspace
		self.algebra = subspace.algebra
		self.domain = self.algebra.subspace.vector() if domain is None else domain

	@property
	def components(self):
		return len(self.subspace)
	@property
	def dimensions(self):
		return len(self.domain)
	@property
	def shape(self):
		return self.arr.shape[1:]

	def meshgrid(self):
		xs = [np.linspace(-1, 1, s, endpoint=False) for s in self.shape]
		c = np.array(np.meshgrid(*xs, indexing='ij'))
		# d = deltas(self.subspace.algebra.subspace.scalar(), self.subspace)[:, 0, :]
		from discrete.util import deltas
		d = deltas(self.subspace, self.algebra.subspace.scalar())
		return c    # FIXME: add version with per element offset

	def quadratic(self, sigma=1, location=0):
		x = (self.meshgrid().T - location)
		return ((x ** 2) / sigma**2).sum(axis=-1).T

	def gauss(self, sigma=0.1, location=0):
		return np.exp(-self.quadratic(sigma, location))


	def process_op(self, op):
		"""preprocess operator into easily consumable terms"""
		from collections import namedtuple
		Term = namedtuple('Term', ['contraction', 'd_idx', 'f_idx', 'sign'])

		is_id = lambda di, fi: 1 if np.bitwise_and(self.domain.blades[di], self.subspace.blades[fi]) else 0
		return tuple([
			(eqi, tuple([
				Term(contraction=is_id(di, fi), d_idx=int(di), f_idx=int(fi), sign=int(op.kernel[di, fi, eqi]))
				for di, fi in zip(*np.nonzero(op.kernel[..., eqi]))
			]))
			for eqi in range(len(op.subspace))
		])

	def term_to_str(self):
		domain = self.domain.named_str.split(',')
		field = self.subspace.named_str.replace('1', 's').split(',')
		sign = {-1: '-', +1: '+'}
		ei = {0: 'e', 1: 'i'}
		def inner(t):
			return f'{sign[t.sign]}{ei[t.contraction]}d{domain[t.d_idx]}({field[t.f_idx]})'
		return inner

	def generate(self, op) -> str:
		"""textual version of a derivative operator"""
		output = op.subspace.named_str.replace('1', 's').split(',')
		term_to_str = self.term_to_str()
		return '\n'.join([
			f'{output[eq_idx]} = ' + ''.join([term_to_str(term) for term in eq])
			for eq_idx, eq in self.process_op(op)
		])

	def generate_geometric(self):
		return self.generate(self.algebra.operator.geometric_product(self.domain, self.subspace))

	def generate_exterior(self):
		return self.generate(self.algebra.operator.outer_product(self.domain, self.subspace))

	def generate_interior(self):
		return self.generate(self.algebra.operator.inner_product(self.domain, self.subspace))


class AbstractSpaceTimeField(AbstractField):
	"""Spacetime field with domain axis t, where we assume a full field over t is never allocated,
	but rather traversed by timestepping"""

	def process_op_leapfrog(self, op):
		"""Preprocess the terms in the derivative operator,
		to be consumed by a leapfrog timestepping scheme"""
		output = op.subspace.named_str.split(',')
		domain = self.domain.named_str.split(',')
		is_spacelike_gradient = lambda term: not 't' is domain[term.d_idx]
		is_spacelike_equation = lambda eq: not 't' in output[eq[0]]

		# split equations into terms of timelike and spacelike gradients
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
