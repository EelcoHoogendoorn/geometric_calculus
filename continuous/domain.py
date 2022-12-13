import jax
from jax import numpy as jnp


import matplotlib.pyplot as plt
import matplotlib.collections


def plot_2d_0form(domain, model, n=128):
	assert model.subspace.equals.scalar()
	grid = domain.sample_grid(n)
	values = jax.vmap(jax.vmap(model))(grid)
	plt.imshow(values[..., 0].T, extent=(-1,+1,-1,+1), origin='lower')
	plt.colorbar()
	# plt.show()


def plot_2d_0form_contour(domain, model, n=128):
	# assert model.subspace.equals.scalar()
	grid = domain.sample_grid(n)
	values = jax.vmap(jax.vmap(model))(grid)
	plt.contour(values[..., 0])
	plt.colorbar()
	plt.show()


def plot_2d_1form(domain, model, n=32):
	assert model.subspace.equals.vector()
	grid = domain.sample_grid(n)
	values = jax.vmap(jax.vmap(model))(grid)
	# values = values - jnp.mean(values, axis=(0, 1), keepdims=True)
	plt.quiver(*grid.reshape(-1,2).T, *values.reshape(-1, 2).T)
	# plt.colorbar()
	# plt.show()

def plot_sampling(objectives):
	key = jax.random.PRNGKey(0)
	plt.figure()
	for _, sample, n, _ in objectives:
		key, _ = jax.random.split(key)
		keys = jax.random.split(key, n)
		samples = jax.vmap(sample)(keys)
		plt.scatter(*samples.T)
	plt.show()


def plot_2d_1form_grid(domain, field, n=32, scale=1e-2):
	def plot_grid(segs1, ax=None, **kwargs):
		ax = ax or plt.gca()
		segs2 = segs1.transpose(1, 0, 2)
		ax.add_collection(matplotlib.collections.LineCollection(segs1, **kwargs))
		ax.add_collection(matplotlib.collections.LineCollection(segs2, **kwargs))
		ax.autoscale()
	assert field.subspace.equals.vector()
	grid = domain.sample_grid(n)
	values = jax.vmap(jax.vmap(field))(grid)
	values = values - jnp.mean(values, axis=(0, 1), keepdims=True)
	size = sum(jnp.linalg.norm(jnp.gradient(values, axis=i), axis=-1).mean() for i in [0, 1]) / 2
	values = values / size * scale
	plot_grid(grid + values)


class Domain:
	def sample_grid(self, n):
		x = jnp.linspace(0, 1, n, endpoint=True) * 2 - 1
		grid = jnp.meshgrid(*[x]*self.n, indexing='ij')
		return jnp.moveaxis(jnp.array(grid), 0, -1)


class UnitCube(Domain):
	def __init__(self, geometry):
		self.geometry = geometry
		self.n = len(self.geometry.domain)

	def sample_interior(self, key):
		return jax.random.uniform(key, shape=(self.n,), minval=-1, maxval=+1)
	def sample_boundary(self, key):
		s = jax.random.uniform(key, shape=(self.n,), minval=-1, maxval=+1)
		return s / jnp.max(jnp.abs(s))

	def sample_boundary_axis(self, axis):
		def inner(key):
			s = jax.random.uniform(key, shape=(self.n,), minval=-1, maxval=+1)
			return s.at[axis].set(jnp.sign(s[axis]))
		return inner

	def max(self, x, d):
		"""true for boundary points on the minimum of axis d"""
		return jnp.all(-jnp.abs(x[d]) <= x)
	def min(self, x, d):
		return jnp.all(jnp.abs(x[d]) >= x)

	def is_max(self, x, d):
		"""true for boundary points on the minimum of axis d"""
		return jnp.all(x[d] >= x)

	def which_side(self, x, d):
		"""get sign of boundary"""
		return jnp.sign(x[d])


	def is_axis(self, x, d):
		return jnp.argmax(jnp.abs(x)) == d



class UnitSphere(Domain):
	def __init__(self, n):
		self.n = n
	def sample_interior(self, key):
		return jax.random.ball(key, d=self.n)
	def sample_boundary(self, key):
		s = jax.random.ball(key, d=self.n)
		return s / jnp.linalg.norm(s)
