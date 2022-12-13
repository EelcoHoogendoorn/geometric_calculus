import numpy as np
from pyopencl import array as cla

from discrete.loopy.field import Field
from discrete.loopy.field_sta import SpaceTimeField


class Context:
	def __init__(self, algebra, domain=None):
		self.algebra = algebra
		self.domain = algebra.subspace.vector() if domain is None else domain
		import pyopencl as cl
		self.ctx = cl.create_some_context()
		self.queue = cl.CommandQueue(self.ctx)
		self.dtype = np.float32

	def make_field(self, subspace, shape):
		shape = (len(subspace),) + shape
		return Field(
			self,
			subspace,
			domain=self.domain,
			arr=cla.zeros(self.queue, shape, self.dtype)
		)


class SpaceTimeContext:
	def __init__(self, algebra, domain=None):
		self.algebra = algebra
		self.domain = algebra.subspace.vector() if domain is None else domain
		self.temporal_domain = algebra.subspace.t
		self.spatial_domain = self.domain.difference(self.temporal_domain)
		import pyopencl as cl
		self.ctx = cl.create_some_context()
		self.queue = cl.CommandQueue(self.ctx)
		self.dtype = np.float32

	def make_field(self, subspace, shape):
		shape = (len(subspace),) + shape
		return SpaceTimeField(
			self,
			subspace,
			domain=self.domain,
			arr=cla.zeros(self.queue, shape, self.dtype)
		)
