"""
TODO: add periodic input encoder options

"""
from continuous.field import Field

from typing import Dict

import jax
from jax import numpy as jnp

from flax.linen.initializers import xavier_uniform, zeros, ones, uniform

def make_model_modal(n_components, n_modes, n_frequencies, n_hidden, scale=1e1):
	key = jax.random.PRNGKey(0)
	key, subkey = jax.random.split(key)

	# FIXME: put in constructor
	V = jax.random.normal(subkey, shape=(n_frequencies, n_components)) * scale
	key, subkey = jax.random.split(key)
	b = jax.random.uniform(subkey, shape=(n_frequencies,)) * (jnp.pi * 2)

	import flax.linen as nn

	class Model(nn.Module):
		@nn.compact
		def __call__(self, x):
			"""
			Parameters
			----------
			x: [n_dim]

			Returns
			-------
			[n_modes, n_dim]
			"""
			x = jnp.cos(V.dot(x) + b)
			x = nn.Dense(features=n_hidden)(x)
			# x = nn.leaky_relu(x)
			x = jnp.tanh(x)
			# x = nn.Dense(features=256, kernel_init=initializer(128))(x)
			# x = nn.leaky_relu(x)
			x = nn.Dense(features=n_hidden)(x)
			x = jnp.tanh(x)
			# x = nn.leaky_relu(x)
			x = nn.Dense(features=n_modes * n_components)(x)
			return x.reshape(n_modes, n_components)

	model = Model()
	key, subkey = jax.random.split(key)
	x = jax.random.normal(subkey, (n_components,))  # Dummy input
	key, subkey = jax.random.split(key)
	params = model.init(subkey, x)  # Initialization call
	return model, params


def make_field_model(geometry, inputs: "Subspace", outputs: Dict, n_frequencies: int, n_hidden: tuple, scale=1e1, window=lambda f, x: f):

	# FIXME: feed in key
	key = jax.random.PRNGKey(3)
	key, subkey = jax.random.split(key)

	# FIXME: put in constructor
	V = jax.random.normal(subkey, shape=(n_frequencies, len(inputs))) * scale
	key, subkey = jax.random.split(key)
	b = jax.random.uniform(subkey, shape=(n_frequencies,)) * (jnp.pi * 2)
	def fourier_mapping(x):
		return jnp.cos(V.dot(x) + b)

	import flax.linen as nn
	# act = nn.tanh
	# https://arxiv.org/pdf/2111.15135.pdf
	# act = lambda x: jnp.exp(-x*x)
	# a = 1e+1
	act = lambda x, a: 1 / (1+x*x*a)
	class Model(nn.Module):
		@nn.compact
		def __call__(self, x):
			x = fourier_mapping(x)
			for i, n in enumerate(n_hidden):
				x = nn.Dense(features=n, kernel_init=xavier_uniform(), name=f'shared{i}')(x)
				# FIXME: would learnable a add any value?
				# a = self.param(f'actscale_{i}', ones, (n,))
				x = act(x, a=1)
			if isinstance(outputs, Dict):
				return {
					k: nn.Dense(features=len(s), name=f'affine_{k}')(x)
					for k, s in outputs.items()
				}
			else:
				return nn.Dense(features=len(outputs))(x)#, kernel_init=zeros)(x)


	model = Model()
	key, subkey = jax.random.split(key)
	x = jax.random.normal(subkey, (len(inputs),))  # Dummy input
	# apply =  lambda p: lambda x: model.apply(p, x)
	# return apply, params

	def apply(params):
		"""apply function that wraps output into fields"""
		if isinstance(outputs, Dict):
			# FIXME: this forces independent evals? is there a nicer way?
			#  pre-eval the shared backbone or something?
			def foo(k):
				return lambda  x: model.apply(params, x)[k]
			return {
				k: geometry.field(foo(k), subspace=o)
				for k, o in outputs.items()
			}
		else:
			def inner(x):
				return window(model.apply(params, x), x)
			return geometry.field(inner, subspace=outputs)

	key, subkey = jax.random.split(key)
	params = model.init(subkey, x)  # Initialization call

	return apply, params


