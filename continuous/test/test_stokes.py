import jax.numpy as jnp
import jax
from optax import huber_loss

from continuous.domain import *
from continuous.models import make_field_model
from continuous.geometry import Geometry
from continuous.optimize import optimize
from continuous.field import Field
from numga.algebra.algebra import Algebra


def navier_stokes(phi: Field, Re=1) -> Field:
	"""steady-state incompressible navier-stokes"""
	velocity = phi.interior_derivative()            # 1-form
	vorticity = velocity.exterior_derivative()      # 2-form
	shear = vorticity.interior_derivative()         # 1-form
	diffusion = shear.exterior_derivative()         # 2-form
	advection = velocity.directional_derivative(vorticity)
	return diffusion + Re * advection               # momentum balance


def test_navier_stokes_potential_2d():
	algebra = Algebra.from_str('x+y+')
	geometry = Geometry(algebra)
	domain = UnitCube(geometry)

	# create a parametric function that maps points in the domain, to a (dict of) multivector fields
	model, params = make_field_model(
		geometry=geometry,
		inputs=geometry.domain,
		# bivector potential
		outputs=geometry.algebra.subspace.bivector(),
		n_frequencies=64,
		n_hidden=[64]*3,
		# why are we so sensitive to this parameter? poor weight init?
		scale=1e0,
	)
	# visualize initial random starting field
	plot_2d_0form_contour(domain, model(params).dual_inverse())
	# plot_2d_1form_grid(domain, model(params).interior_derivative())
	plt.show()

	# satisfy NS in interior of domain
	def objective_internal(phi: Field, x):
		return huber_loss(
			navier_stokes(phi)(x),
			0,
			1e-6
		)
	# pin phi to 0 on boundary; no normal flux
	def objective_boundary_phi(phi, x):
		return huber_loss(
			phi(x),
			0,
			1e-6
		)
	# lid-driven cavity; set gradients of phi
	boundary_function = lambda x: domain.max(x, d=0) * jnp.eye(domain.n)[0]
	def objective_boundary_v(phi, x):
		velocity = phi.interior_derivative()
		return huber_loss(
			velocity(x),
			boundary_function(x),
			1e-6
		)

	objectives = [
		(objective_internal, domain.sample_interior, 128, 1e-2),
		(objective_boundary_phi, domain.sample_boundary, 128, 1e+2),
		(objective_boundary_v, domain.sample_boundary, 128, 1e+1),
	]
	import time
	t = time.time()
	params = optimize(model, params, objectives, n_steps=501)
	print('time', time.time() - t)

	# visualize solution
	plot_2d_0form_contour(domain, model(params).dual_inverse())
	# plot_2d_1form_grid(domain, model(params).interior_derivative())
	plt.show()


def test_navier_stokes_potential_23d():
	algebra = Algebra.from_str('x+y+z+')
	geometry = Geometry(algebra, constant=algebra.subspace.z)
	domain = UnitCube(geometry)

	# create a parametric function that maps points in the domain, to a (dict of) calculus forms
	model, params = make_field_model(
		geometry=geometry,
		inputs=geometry.domain,
		# bivector potential; leave out z components for 2.5d case
		outputs=geometry.algebra.subspace.xy,
		n_frequencies=64,
		n_hidden=[64]*2,
		# why are we so sensitive to this parameter? poor weight init?
		scale=3e0,
	)
	# visualize initial random starting field
	plot_2d_0form_contour(domain, model(params).dual_inverse())

	# satisfy NS in interior of domain
	def objective_internal(phi: Field, x):
		return huber_loss(
			navier_stokes(phi)(x),
			0,
			1e-6
		)
	# pin phi to 0 on boundary; no normal flux
	def objective_boundary_phi(phi, x):
		return huber_loss(
			phi(x),
			0,
			1e-6
		)
	# lid-driven cavity; set gradients of phi
	boundary_function = lambda x: domain.max(x, d=0) * jnp.eye(domain.n)[0]
	def objective_boundary_v(phi, x):
		velocity = phi.interior_derivative()
		return huber_loss(
			velocity(x),
			boundary_function(x),
			1e-6
		)

	objectives = [
		(objective_internal, domain.sample_interior, 128, 1e-2),
		(objective_boundary_phi, domain.sample_boundary, 128, 1e+2),
		(objective_boundary_v, domain.sample_boundary, 128, 1e+1),
	]
	import time
	t = time.time()
	params = optimize(model, params, objectives, n_steps=201)
	print('time', time.time() - t)

	# visualize solution
	plot_2d_0form_contour(domain, model(params).dual_inverse())
