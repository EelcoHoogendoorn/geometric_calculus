"""Some basic tests of continuous field objects
"""

import jax

from continuous.geometry import Geometry
from numga.algebra.algebra import Algebra


def test_basic_3():
	dg = Geometry(Algebra.from_pqr(3, 0, 0))
	# set up simple 0-form
	f = dg.k_field(lambda x: (x * x).sum(keepdims=True), k=0)

	g = f.exterior_derivative() # gradient
	c = g.exterior_derivative() # curl
	d = c.exterior_derivative() # divergence

	l = g.directional_derivative(f)
	print(l.subspace)
	l = g.directional_derivative(g)
	print(l.subspace)
	l = g.directional_derivative(c)
	print(l.subspace)


	key = jax.random.PRNGKey(0)
	x = jax.random.normal(key, shape=(dg.algebra.description.n_dimensions,))

	print('x', x)
	print('f', f(x))
	print('v', g(x))
	print('c', c(x))
	print('d', d(x))
	# print('l', l(x))
	print('laplacian', f.exterior_derivative().interior_derivative()(x))


def test_basic_2():
	dg = Geometry(Algebra.from_pqr(2, 0, 0))
	q= dg.algebra.pseudo_scalar_squared
	# set up simple 0-form
	f = dg.k_field(lambda x: (x * x).sum(keepdims=True), k=0)

	g = f.exterior_derivative() # gradient
	# d = g.exterior_derivative() # divergence

	l = g.directional_derivative(f)
	print(l.subspace)

	key = jax.random.PRNGKey(0)
	x = jax.random.normal(key, shape=(dg.algebra.description.n_dimensions,))

	print('x', x)
	print('f', f(x))
	print('v', g(x))
	print('l', l(x))
	print('laplacian', f.exterior_derivative().dual().exterior_derivative()(x))
