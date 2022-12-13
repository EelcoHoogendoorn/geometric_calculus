"""
exterior derivatives in euclidian space

https://arxiv.org/pdf/2007.10205.pdf
this paper does exactly that for 1d laplace; seems to work well enough
https://openreview.net/pdf?id=m4baHw5LZ7M
extended version. i really like it tbh
lack of BC requirement kinda cool?
how would be use this to model an open boundary?
would simply not doing anything outside the domain suffice?

eigen stuff does appear finicky.
can we try solving direct problems first?

"""

import jax
import jax.numpy as jnp

from calculus.geometry import Geometry
from calculus.models import make_model_modal


def eigen_loss(samples, function, lhs):
	"""

	Parameters
	----------
	samples: [n_samples, n_components]
		sample points in the domain
	function: callable
		batched mapping from domain to solution
	lhs: callable
		batched mapping from samples,
		to output residual space
	rhs: callable


	Returns
	-------
	scalar loss
	"""
	u = function(samples)  # [n_samples, n_modes, n_components]
	L = lhs(samples)     # [n_samples, n_modes, n_components]

	# rayleigh? need both lhs and rhs; rhs in divisor
	rayleigh = -jnp.einsum('smc, smc -> m', L, u) / jnp.einsum('smc, smc -> m', u, u)

	eq = L + u * rayleigh[None, :, None]

	l = (jnp.einsum('smc, smc -> m', u, u) / len(samples)) ** (0.5)
	inner = jnp.einsum('smc, snc -> mn', u, u) / len(samples) / jnp.outer(l, l)
	ortho = inner - jnp.eye(*inner.shape)
	return (
		# kinda nice if eig problem is satisfied
		jnp.abs(eq).mean() * 1e3 + \
		# orthonormality ftw
		jnp.square(ortho).mean() * 1e-3 + \
		jnp.square(l - 1).mean() * 1e-3
		# smaller eig estimates are preferred
		# jnp.square(rayleigh).mean() * 1e3
	)


def test_fancy():
	dg = Geometry(1)
	model, params = make_model_modal(
		n_components=1,
		n_modes=10,
		n_frequencies=64,
		n_hidden=64,
		scale=1e1,
	)

	n_samples = 64
	key = jax.random.PRNGKey(0)
	samples = jax.random.uniform(key, shape=(n_samples, 1))
	samples_boundary = jnp.arange(2).reshape(2, 1) * 1.0

	def shizzle(params):
		modes = lambda s: model.apply(params, s)
		eq = lambda x: dg.k_field(modes, k=0).laplace()(x)
		batched_modes = jax.vmap(modes)
		batched_eq = jax.vmap(eq)
		bv = batched_modes(samples_boundary)
		b_loss = jnp.square(bv).sum()
		return eigen_loss(samples, batched_modes, batched_eq) + b_loss * 1e3


	import optax
	learning_rate = 3e-3
	tx = optax.adam(learning_rate=learning_rate)
	opt_state = tx.init(params)
	loss_grad_fn = jax.jit(jax.value_and_grad(shizzle))
	for i in range(101):
		if i % 100 == 0:
			learning_rate *= 0.7
		loss_val, grads = loss_grad_fn(params)
		updates, opt_state = tx.update(grads, opt_state)
		params = optax.apply_updates(params, updates)
		if i % 10 == 0:
			print('Loss step {}: '.format(i), loss_val)

	sample_model = lambda s: model.apply(params, s)

	s = (jnp.arange(100) / 100).reshape(100, 1)
	sol = jax.vmap(sample_model)(s)
	print(sol.shape)
	import matplotlib.pyplot as plt
	plt.plot(s[..., 0], sol[..., 0])
	plt.show()


def test_exact_laplace():
	dg = Geometry(1)

	exact = dg.k_field(lambda x: jnp.sin(x * jnp.pi), k=0)

	def eq(s):
		return exact.laplace()(s) + exact(s) * 9.87
	eq = jax.vmap(eq)
	s = (jnp.arange(100) / 100).reshape(100, 1)
	res = eq(s)
	L = jax.vmap(exact.laplace())(s)
	u = exact(s)
	R = -jnp.einsum('sc,sc->', L, u) / jnp.einsum('sc,sc->', u, u)
	print(R)
	import matplotlib.pyplot as plt
	plt.plot(res[..., 0])
	plt.show()
	print(res)

