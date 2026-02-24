import numpy as np

import pyopencl as cl
from pyopencl import array as cla

from discrete.opencl.field import Field
from discrete.opencl.field_slice import FieldSlice


class Context:
	def __init__(self, algebra, domain=None):
		self.algebra = algebra
		self.domain = algebra.subspace.vector() if domain is None else domain
		# self.ctx = cl.Context(devices=cl.get_platforms()[0].get_devices()[2:3])
		self.ctx = cl.create_some_context()
		# import pyopencl
		# print(pyopencl.get_platforms()[0].get_devices())

		self.queue = cl.CommandQueue(self.ctx)
		self.dtype = np.float32

	@property
	def temporal_domain(self):
		return self.algebra.subspace.t
	@property
	def spatial_domain(self):
		return self.domain.difference(self.temporal_domain)

	def coerce_array(self, arr):
		if isinstance(arr, cla.Array):
			return arr
		if isinstance(arr, np.ndarray):
			return cla.to_device(self.queue, arr.astype(self.dtype))
	def allocate_array(self, shape):
		return cla.zeros(self.queue, shape, self.dtype)

	def make_field(self, subspace, shape):
		return Field(
			self,
			subspace,
			domain=self.domain,
			arr=self.allocate_array((len(subspace),) + shape)
		)

	def make_field_slice(self, subspace, shape):
		return FieldSlice(
			self,
			subspace,
			domain=self.domain,
			arr=self.allocate_array((len(subspace),) + shape)
		)
