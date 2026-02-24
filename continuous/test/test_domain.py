
import jax
from continuous.domain import UnitCube, UnitSphere


def test_cube():
	domain = UnitCube(2)
	key = jax.random.PRNGKey(0)
	import matplotlib.pyplot as plt
	ni = 128
	nb = 128
	plt.figure()
	plt.scatter(*jax.vmap(domain.sample_interior)(jax.random.split(key, ni)).T)
	plt.scatter(*jax.vmap(domain.sample_boundary)(jax.random.split(key, nb)).T)
	plt.show()
	plt.figure()
	plt.scatter(*domain.sample_grid(16).reshape(-1, 2).T)
	plt.show()


def test_sphere():
	domain = UnitSphere(2)
	key = jax.random.PRNGKey(0)
	ki, kb = jax.random.split(key, 2)
	import matplotlib.pyplot as plt
	ni = 128
	nb = 128
	plt.figure()
	plt.scatter(*jax.vmap(domain.sample_interior)(jax.random.split(ki, ni)).T)
	plt.scatter(*jax.vmap(domain.sample_boundary)(jax.random.split(kb, nb)).T)
	plt.show()

