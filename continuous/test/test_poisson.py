import jax.numpy as jnp
import jax
from optax import huber_loss

from calculus.domain import plot_2d_0form, UnitCube, UnitSphere
from calculus.models import make_field_model
from calculus.geometry import Geometry
from calculus.optimize import optimize
from calculus.field import Field


def test_laplace():
	geometry = Geometry(2)
	domain = UnitCube(geometry)

	# create a parametric function that maps points in the domain, to a (dict of) calculus forms
	model, params = make_field_model(
		geometry=geometry,
		inputs=geometry.domain,
		outputs=geometry.algebra.subspace.scalar(),
		n_frequencies=64,
		n_hidden=[128] * 3,
		# why are we so sensitive to this parameter? poor weight init?
		scale=1e+0,
	)
	# visualize initial random starting field
	plot_2d_0form(domain, model(params))

	# the PDE to solve
	def laplacian(u):
		return u.exterior_derivative().interior_derivative()
		# return u.geometric_derivative().geometric_derivative()

	# satisfy laplace/poisson in interior of domain
	def objective_internal(model: Field, x):
		return huber_loss(
			laplacian(model)(x),
			0,
			1e-3
		)
	# dirichlet boundary condition on boundary of domain
	# +-1 on vertical/horizontal sides
	boundary_function = lambda x: jnp.sign(jnp.diff(jnp.abs(x))) * 1.0
	def objective_boundary(model, x):
		return huber_loss(
			model(x),
			boundary_function(x),
			1e-2
		)

	objectives = [
		(objective_internal, domain.sample_interior, 128, 1e-0),
		(objective_boundary, domain.sample_boundary, 128, 1e+1),
	]
	params = optimize(model, params, objectives, n_steps=201)

	# visualize solution
	plot_2d_0form(domain, model(params))
