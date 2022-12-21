import numpy as np

import os
import imageio.v3 as iio

import discrete.util


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

	def process_op(self, op):
		"""preprocess operator into easily consumable terms"""
		from collections import namedtuple
		# FIXME: add string names here
		Term = namedtuple('Term', ['contraction', 'd_idx', 'f_idx', 'sign'])

		is_id = lambda di, fi: 1 if np.bitwise_and(self.domain.blades[di], self.subspace.blades[fi]) else 0
		return tuple([
			(eqi, tuple([
				Term(
					contraction=is_id(di, fi),
					d_idx=int(di),
					f_idx=int(fi),
					sign=int(op.kernel[di, fi, eqi])
				)
				for di, fi in zip(*np.nonzero(op.kernel[..., eqi]))
			]))
			for eqi in range(len(op.subspace))
		])

	def term_to_str(self):
		domain = discrete.util.split(',')
		field = discrete.util.split(',')
		sign = {-1: '-', +1: '+'}
		ei = {0: 'e', 1: 'i'}
		def inner(t):
			return f'{sign[t.sign]}{ei[t.contraction]}d{domain[t.d_idx]}({field[t.f_idx]})'
		return inner

	def generate(self, op) -> str:
		"""textual version of a derivative operator"""
		output = discrete.util.split(',')
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


	# utility functions
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

	def smooth_noise(self, sigma):
		arr = np.random.normal(size=self.arr.shape)
		import scipy
		s = [0] * arr.ndim
		s[1:] = sigma
		arr = scipy.ndimage.gaussian_filter(arr, sigma=s, mode='wrap')
		return self.copy(arr=arr)

	def random_gaussian(self, sigma, location=0, seed=None):
		"""initialize a field with a random gaussian"""
		if seed:
			np.random.seed(seed)
		gauss = self.gauss(np.array(sigma), np.array(location))
		vec = np.random.normal(size=(self.components,))
		vec = vec / np.linalg.norm(vec)
		q = np.einsum('c, ...->c...', vec, gauss)
		return self.copy(arr=q)

	def upscale_array(self, arr, nums):
		import scipy.signal
		for i, s in enumerate(arr.shape):
			if nums[i] > 1:
				arr = scipy.signal.resample(arr, num=s*nums[i], axis=i)
		return arr

	def upscale(self, s=None):
		if s is None:
			s = [2]*self.dimensions
		return self.copy(arr=self.upscale_array(self.arr, [1]+s))


	# visualization functions

	def write_animation_base(self, image, basepath, components, pre='', post='', gamma=True, anim=True):
		basename = '_'.join([pre, str(self.shape), self.algebra.description.description_str, self.subspace.pretty_str, components, post])
		os.makedirs(basepath, exist_ok=True)

		def tonemap(image):
			# image = np.abs(image)
			if gamma:
				image = np.sqrt(image)
			scale = np.percentile(image.flatten(), 99)
			image = np.clip(image / scale * 255, 0, 255).astype(np.uint8)
			return image

		if image.ndim == 4:
			if anim:
				iio.imwrite(os.path.join(basepath, basename + '_anim.gif'), tonemap(image))
			iio.imwrite(os.path.join(basepath, basename + '_xt.gif'), tonemap(image[::-1, image.shape[1] // 2]))
		elif image.ndim == 3:
			iio.imwrite(os.path.join(basepath, basename + '_xt.gif'), tonemap(image[::-1]))
		else:
			raise

	def write_animation(self, selector, basepath, components, **kwargs):
		def get_components(f, components):
			"""sample field components as a numpy array, with field components last, for rendering"""
			return np.moveaxis(getattr(f, components), 0, -1)
		def to_animation(components):
			image = selector(get_components(self, components))
			return np.moveaxis(image, -2, 0)  # t first
		image = to_animation(components)
		self.write_animation_base(image, basepath, components, **kwargs)

	def write_gif_1d(self, **kwargs):
		def selector(arr):
			return np.abs(arr)
		self.write_animation(selector, **kwargs)
	def write_gif_2d(self, **kwargs):
		def selector(arr):
			return np.abs(arr)
		self.write_animation(selector, **kwargs)
	def write_gif_2d_compact(self, **kwargs):
		def selector(arr):
			return np.abs(arr).mean(0)
		self.write_animation(selector, **kwargs)
	def write_gif_3d(self, **kwargs):
		def selector(arr):
			mid = lambda a: a[a.shape[0] // 2]
			return mid(np.abs(arr))
		self.write_animation(selector, **kwargs)
	def write_gif_3d_compact(self, **kwargs):
		def selector(arr):
			return np.abs(arr).mean(0)
		self.write_animation(selector, **kwargs)
	def write_gif_4d_compact(self, **kwargs):
		def selector(arr):
			mean = lambda a: a.mean(0)
			mid = lambda a: a[a.shape[0] // 2]
			return mid(mean(np.abs(arr)))
		self.write_animation(selector, **kwargs)


