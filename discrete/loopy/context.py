import numpy as np

import pyopencl as cl
from pyopencl import array as cla


from discrete.loopy.field import Field
from discrete.loopy.field_sta import SpaceTimeField


class Context:
	def __init__(self, algebra, domain=None):
		self.algebra = algebra
		self.domain = algebra.subspace.vector() if domain is None else domain
		self.ctx = cl.create_some_context()
		self.queue = cl.CommandQueue(self.ctx)
		self.dtype = np.float32

	def coerce_array(self, arr):
		if isinstance(arr, cla.Array):
			return arr
		if isinstance(arr, np.ndarray):
			return cla.to_device(self.queue, arr)
	def allocate_array(self, shape):
		return cla.zeros(self.queue, shape, self.dtype)

	def make_field(self, subspace, shape):
		return Field(
			self,
			subspace,
			domain=self.domain,
			arr=self.allocate_array((len(subspace),) + shape)
		)


class SpaceTimeContext(Context):
	def __init__(self, algebra, domain=None):
		super(SpaceTimeContext, self).__init__(algebra, domain)
		self.temporal_domain = algebra.subspace.t
		self.spatial_domain = self.domain.difference(self.temporal_domain)

	def make_field(self, subspace, shape):
		return SpaceTimeField(
			self,
			subspace,
			domain=self.domain,
			arr=self.allocate_array((len(subspace),) + shape)
		)
