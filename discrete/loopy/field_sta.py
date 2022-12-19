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

		# FIXME: integrate metric properly
		metric_t = 0.33

		loopy_term = self.loopy_term(domain)
		# FIXME: allow dynamic extension by fusing other stuff onto statements
		line = '{lhs} = {lhs} {s} ({rhs}) * {metric_t}'
		statements = [
			line.format(
				lhs=f'R[{tt.f_idx}, {axes}]',
				s=sign[tt.sign],
				rhs='  '.join(loopy_term(t) for t in ts),
				metric_t=metric_t,
			)
			for eq_idx, (tt, ts) in halfstep
		]
		print("{[" + axes + "]:" + limits + "}")
		print('\n'.join(statements))

		knl = lp.make_kernel(
			"{[" + axes + "]:" + limits + "}",
			'\n'.join(statements)
		)

		# FIXME: make this stuff configurable. 16 beats no split on cpu.
		#  16-33s, None->52s, 128->55s, 4->27s, 8->29, 2->11.5s; 1->11s.
		#  whoa; no clue what this is all about
		#  on-device a single block of 16 seems near optimal. rollout gives mem restriction tho
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
		self.arr = self.context.coerce_array(arr)
		assert len(arr) == len(subspace)
		assert len(self.shape) == len(context.spatial_domain)

	def geometric_derivative(self, mass=None, metric=None):
		op = self.algebra.operator.geometric_product(self.domain, self.subspace)
		return STAOperator(self.context, self, op)

	# def rollout(self, steps):
	# 	arr = self.context.allocate_array((steps,) + self.arr.shape)
	# 	print('GBs: ', arr.size / (2**30))
	# 	import time
	# 	op = self.geometric_derivative()
	# 	tt = time.time()
	# 	for t in range(steps):
	# 		# FIXME: kernel that does not update in place but writes direct to next step
	# 		#  would be more efficient? that said we do typically unroll into substeps anyway, so whatever?
	# 		arr.setitem(t, self.arr, self.context.queue)
	# 		for substep in range(3):
	# 			op.apply()
	# 	print('time: ', time.time() - tt)
	# 	n = arr.ndim
	# 	axes = [(a + 1) % n for a in range(n)]
	# 	arr = arr.transpose(axes)
	# 	return Field(self.context, self.subspace, self.domain, arr)

	def rollout(self, steps, **kwargs) -> Field:
		"""perform a rollout of a leapfrog field into a whole field"""
		# FIXME: better to allocate off-device, for mem availability? or leave that to generator?
		arr = self.context.allocate_array((steps,) + self.arr.shape)
		print('GBs: ', arr.size / (2**30))
		for t, slc in enumerate(self.rollout_generator(steps, **kwargs)):
			arr.setitem(t, slc.arr, self.context.queue)
		n = arr.ndim
		arr = arr.transpose([(a + 1) % n for a in range(n)])    # move t axis last
		return Field(self.context, self.subspace, self.domain, arr)

	def rollout_generator(self, steps, unroll=1, metric={}, **kwargs):
		"""perform a rollout of a leapfrog field"""
		# work safe CFL condition into metric scaling
		cfl_unroll = self.dimensions
		cfl_metric = {**metric, 't': metric.get('t', 1) / cfl_unroll}

		op = self.geometric_derivative()

		import time
		tt = time.time()

		for t in range(steps):
			yield self.to_numpy()
			for substep in range(cfl_unroll * unroll):
				op.apply()
		print('time: ', time.time() - tt)

	def to_numpy(self):
		"""Map opencl field to numpy field
		dont want to bother making our visualization code compatible with opencl nonsense
		"""
		from discrete.numpy.field import SpaceTimeField
		arr = self.arr.map_to_host(queue=self.context.queue)
		return SpaceTimeField(self.subspace, self.domain, arr)