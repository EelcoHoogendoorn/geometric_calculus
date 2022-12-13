import numpy as np


class AbstractField:
	"""Field without specific array backing storage, just doing symbolic manipulation"""
	def __init__(self, subspace, domain=None):
		self.subspace = subspace
		self.algebra = subspace.algebra
		self.domain = self.algebra.subspace.vector() if domain is None else domain

	@property
	def components(self):
		return len(self.subspace)

	def process_op(self, op):
		"""preprocess operator into easily consumable terms"""
		from collections import namedtuple
		Term = namedtuple('Term', ['contraction', 'd_idx', 'f_idx', 'sign'])

		is_id = lambda di, fi: 1 if np.bitwise_and(self.domain.blades[di], self.subspace.blades[fi]) else 0
		return tuple([
			(eqi, tuple([
				# (is_id(di, fi), int(di), int(fi), int(op.kernel[di, fi, eqi]))
				Term(contraction=is_id(di, fi), d_idx=int(di), f_idx=int(fi), sign=int(op.kernel[di, fi, eqi]))
				for di, fi in zip(*np.nonzero(op.kernel[..., eqi]))
			]))
			for eqi in range(len(op.subspace))
		])

	def generate(self, op) -> str:
		"""textual version of a derivative operator"""
		domain = self.domain.named_str.split(',')
		field = self.subspace.named_str.replace('1', 's').split(',')
		output = op.subspace.named_str.replace('1', 's').split(',')
		sign = {-1: '-', +1: '+'}
		ei = {0: 'e', 1: 'i'}

		def term_to_str(term):
			return f'{sign[term.sign]}{ei[term.contraction]}d{domain[term.d_idx]}({field[term.f_idx]})'

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
		is_t = lambda term: 't' is domain[term.d_idx]

		def split_eq(eq):
			T, *S = sorted(eq, key=is_t, reverse=True)
			return T._replace(sign=-T.sign), tuple(S)
		# pull dt to the left
		equations = [
			(eq_idx, split_eq(eq))
			for eq_idx, eq in self.process_op(op)
			# drop equations that trivialize
			if any(is_t(term) for term in eq) and len(eq) > 1
		]
		# split by spacelike and timelike equations
		timelike, spacelike = [], []
		for eq_idx, eq in equations:
			(spacelike, timelike)['t' in output[eq_idx]].append((eq_idx, eq))
		return tuple(timelike), tuple(spacelike)

	def generate(self, op) -> str:
		"""Text representation of a leapfrog geometric derivative operator"""
		domain = self.domain.named_str.split(',')
		field = self.subspace.named_str.replace('1', 's').split(',')
		equation = op.subspace.named_str.replace('1', 's').split(',')
		sign = {-1: '-', +1: '+'}
		ei = {0: 'e', 1: 'i'}

		def term_to_str(term):
			return f'{sign[term.sign]}{ei[term.contraction]}d{domain[term.d_idx]}({field[term.f_idx]})'

		T, S = self.process_op_leapfrog(op)
		return '\n'.join([
			f'{term_to_str(tt)} = ' + ''.join([term_to_str(t) for t in ts]) + f'\t # {equation[eq_idx]}'
			for eq_idx, (tt, ts) in T + S
		])

	def generate_geometric(self) -> str:
		return self.generate(self.algebra.operator.geometric_product(self.domain, self.subspace))
