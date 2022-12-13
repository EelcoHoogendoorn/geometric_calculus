"""JAX based discrete calculus implementation
"""
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
		self.shape = arr.shape[1:]
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

	def meshgrid(self):
		xs = [jnp.linspace(-1, 1, s) for s in self.shape]
		c = jnp.array(jnp.meshgrid(*xs, indexing='ij'))
		# d = deltas(self.subspace.algebra.subspace.scalar(), self.subspace)[:, 0, :]
		from discrete.util import deltas
		d = deltas(self.subspace, self.algebra.subspace.scalar())
		return c    # FIXME: add version with per element offset

	def restrict(self, subspace):
		op = self.subspace.algebra.operator.restrict(self.subspace, subspace)
		return self.copy(subspace=subspace, arr=jnp.einsum('io,i...->o...', op.kernel, self.arr))

	def geometric_derivative(self, output=None):
		op = self.algebra.operator.geometric_product(self.domain, self.subspace)
		if output:
			op = op.select_subspace(output)
		return self.make_derivative(op)

	def make_derivative(self, op):
		"""plain direct derivative, no leapfrog"""
		arr = jnp.zeros(shape=(len(op.subspace),) + self.shape)
		d = {   # NOTE: these are derivatives, not including metric signs; those enter below
			1: lambda x, a: x - jnp.roll(x, shift=+1, axis=a),
			0: lambda x, a: jnp.roll(x, shift=-1, axis=a) - x
		}
		for eq_idx, eq in self.process_op(op):
			total = sum(
				d[term.contraction](self.arr[term.f_idx], term.d_idx) * term.sign
				for term in eq
			)
			arr = arr.at[eq_idx, ...].set(total)
		return self.copy(arr=arr, subspace=op.subspace)


@register
class SpaceTimeField(Field, AbstractSpaceTimeField):
	# __pytree_ignore__ = ('subspace', 'domain', 'shape', 'algebra')

	""""""
	def __init__(self, subspace, domain, arr):
		super(AbstractSpaceTimeField, self).__init__(subspace, domain)
		self.arr = arr
		self.shape = arr.shape[1:]
		assert len(arr) == len(subspace)
		assert len(self.shape) == len(domain) - 1
		assert 't' in str(self.algebra)

	@property
	def dimensions(self):
		return int((np.array(self.shape) > 1).sum())

	@property
	def courant(self):
		return float(self.dimensions) ** (-0.5)

	def geometric_derivative_leapfrog(self, mass=None, mass_I=None, cubic=None, speed=None, metric=None):
		arr = self.arr.copy()
		if metric is None:
			metric = [1] * self.dimensions
		op = self.algebra.operator.geometric_product(self.domain, self.subspace)
		# FIXME: comp just xyz based dual? prod from other side? not sure if working relative to eq is correct
		dual = self.algebra.operator.geometric_product(self.algebra.subspace.pseudoscalar(), self.subspace)
		kernel = dual.kernel[0]
		dual_map = {k:(v, kernel[k, v]) for k,v in zip(*np.nonzero(kernel))}

		# FIXME: this only now works with t axis last
		derivative = {   # NOTE: these are plain derivatives, not including metric signs
			1: lambda x, a: (x - jnp.roll(x, shift=+1, axis=a)) * metric[a],
			0: lambda x, a: (jnp.roll(x, shift=-1, axis=a) - x) * metric[a]
		}

		speed = self.courant * 0.95 if speed is None else speed

		T, S = self.process_op_leapfrog(op)
		for eq_idx, (t_term, s_terms) in T+S:
			total = sum(
				derivative[term.contraction](arr[term.f_idx], term.d_idx) * term.sign
				for term in s_terms
			)
			if not mass is None:
				# 'direct' mass term; porportional to element being stepped over, or eq
				# FIXME: assure ourselves this element exists in self.subspace
				#  if implicit zero just skip
				assert self.subspace == op.subspace
				total += arr[eq_idx] * mass
			# if not cubic is None:
			# 	term += arr[eqi]**2 * cubic
			if not mass_I is None:
				# dual-t or xyz mass term; direct dual of varialbe fi_ being added to
				#  note; depending on even or odd grade, this will toggle grades
				#  need to check if this term can exists in the given space.
				assert self.subspace == dual.subspace
				fdi, v = dual_map[t_term.f_idx]
				total += arr[fdi] * mass_I * v

			arr = arr.at[t_term.f_idx].add(total * speed * t_term.sign)

		return self.copy(arr=arr)

