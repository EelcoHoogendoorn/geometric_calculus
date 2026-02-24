import jax.numpy as jnp
import jax
from optax import huber_loss

from calculus.domain import plot_2d_0form, UnitCube
from calculus.models import make_field_model
from calculus.geometry import Geometry
from calculus.optimize import optimize
from calculus.field import Field
from numga.algebra.algebra import Algebra

from numga.backend.jax.context import JaxContext


def maxwell(F, J=0):
	return F.geometric_derivative() - J


def maxwell_potential(A, J=0):
	F = A.exterior_derivative()
	return F.interior_derivative() - J

def proca(A, J=0, m=1):
	F = A.exterior_derivative()
	return F.interior_derivative() - J - m**2 * A

def proca_potential(P, J=0, m=0):
	A = P.interior_derivative()                     # divergence-free vector potential (satisfying loretnz condition)
	F = A.exterior_derivative()                     # bivector EM-field
	return F.interior_derivative() - J - m**2 * A

def maxwell_hyper_potential(P, J=0):
	A = P.interior_derivative()
	F = A.exterior_derivative()
	return F.interior_derivative() - J

def elastic_eigen(u, k=1, m=1):
	return k * u.geometric_derivative().geometric_derivative() - m * u


def klein_gordon(phi, A, m, e=1):
	flux = phi.geometric_derivative() - A * e
	return flux.geometric_derivative() - m * phi


# def test_maxwell():
# 	# ndim = 2
# 	dg = Geometry(p=3, r=1)
# 	domain = UnitCube(3+1)
#
# 	# create a parametric function that maps points in the domain, to a (dict of) calculus forms
# 	model, params = make_field_model(
# 		inputs=dg.algebra.subspace.vector(),
# 		outputs=dg.algebra.subspace.bivector(),
# 		n_frequencies=64,
# 		n_hidden=64,
# 		# why are we so sensitive to this parameter? poor weight init?
# 		scale=1e0,
# 	)
# 	# visualize initial random starting field
# 	#  FIXME: plot timelike slices?
# 	# plot_2d_0form(domain, model(params))
#
# 	# satisfy maxwell equations in interior of domain
# 	def objective_internal(model: Field, x):
# 		p = model.exterior_derivative()(x)
# 		d = model.dual().exterior_derivative()(x)
# 		J = 0
# 		return huber_loss(p, 0, 1e-3) + huber_loss(d, J, 1e-3)
# 	# dirichlet boundary condition on boundary of domain
# 	# +-1 on vertical/horizontal sides
# 	# boundary_function = lambda x: jnp.sign(jnp.diff(jnp.abs(x))) * 1.0
# 	# def objective_boundary(model, x):
# 	# 	return huber_loss(
# 	# 		model(x),
# 	# 		boundary_function(x),
# 	# 		1e-2
# 	# 	)
#
# 	objectives = [
# 		(objective_internal, domain.sample_interior, 128, 1e-0),
# 		# (objective_boundary, domain.sample_boundary, 128, 1e+1),
# 	]
# 	params = optimize(model, params, objectives, n_steps=501)
#
# 	# visualize solution
# 	# plot_2d_0form(domain, model(params))

def test_maxwell_geometric():
	algebra = Algebra.from_str('x+y+z+t-')
	# note: with zt constant, we have 2d statics problem
	#  with current densities in z and t direction,
	#  we get nonzero magnetic and electric field respectively.
	#  with only z constant, we can have TM and TE modes (see falstad)
	#  can also only vary xt, yz constant plane wave.

	geometry = Geometry(algebra, constant=algebra.subspace.z_t)
	domain = UnitCube(geometry)
	context = JaxContext(algebra)


	# # create a parametric function that maps points in the domain, to a (dict of) calculus forms
	# model, params = make_field_model(
	# 	geometry=geometry,
	# 	inputs=geometry.domain,
	# 	outputs=geometry.algebra.subspace.bivector(),
	# 	n_frequencies=64,
	# 	n_hidden=[64]*2,
	# 	# why are we so sensitive to this parameter? poor weight init?
	# 	scale=1e0,
	# )
	# visualize initial random starting field
	#  FIXME: plot timelike slices?
	# plot_2d_0form(domain, model(params))

	def maxwell(F, J=0):
		return F.geometric_derivative() - J

	# FIXME: highlights some of the integration work tbd
	#  does a field produce an mv when called?
	#  right now it does; suggesting it does not need its own subspace field?
	#  yet its needed for generator field operators no?
	#  always deduce field subspace by example domain input? feels awkward
	def plane_wave(F0, k) -> Field:
		return geometry.field(
			lambda x: F0 * x.inner(k).dual().exp(),
			F0.subspace.union(F0.subspace.dual())
		)

	k = context.multivector.x + context.multivector.y
	F0 = context.multivector.xy + context.multivector.yz

	field = plane_wave(F0, k)

	# make grid of mv inputs over domain
	x = context.multivector.x * 1e-2
	print(field.subspace)
	print(field(x))
	return


	# satisfy maxwell equations in interior of domain
	def objective_internal(F: Field, x, J=0):
		return huber_loss(maxwell(F, J)(x))

	objectives = [
		(objective_internal, domain.sample_interior, 128, 1e-0),
		# (objective_boundary, domain.sample_boundary, 128, 1e+1),
	]
	params = optimize(model, params, objectives, n_steps=501)

	# visualize solution
	# plot_2d_0form(domain, model(params))


def test_maxwell_static_2d():
	algebra = Algebra.from_str('x+y+z+t-')
	# note: with zt constant, we have 2d statics problem
	#  with current densities in z and t direction,
	#  we get nonzero magnetic and electric field respectively.
	#  with only z constant, we can have TM and TE modes (see falstad)
	#  can also only vary xt, yz constant plane wave.

	geometry = Geometry(algebra, constant=algebra.subspace.z_t)
	domain = UnitCube(geometry)
	# context = JaxContext(algebra)


	# create a parametric function that maps points in the domain, to a (dict of) calculus forms
	model, params = make_field_model(
		geometry=geometry,
		inputs=geometry.domain,
		# is this appropriate for electrostatics?
		outputs=geometry.algebra.subspace.xt_yt,
		n_frequencies=64,
		n_hidden=[64]*2,
		# why are we so sensitive to this parameter? poor weight init?
		scale=1e0,
	)
	plot_2d_0form(domain, model(params))

	def maxwell(F, J=0):
		return F.geometric_derivative() - J

	# satisfy maxwell equations in interior of domain
	def objective_internal(F: Field, x, J=0):
		return huber_loss(maxwell(F, J)(x))


	def objective_current(F: Field, x):
		J = jnp.sign(x[0]) # FIXME: make this current in temporal dir?
		return huber_loss(maxwell(F, J)(x))

	pos = jnp.array([0.5, 0], [-0.5, 0])
	def sample_wires(key):
		return jax.random.choice(key, pos)

	objectives = [
		(objective_internal, domain.sample_interior, 128, 1e-0),
		(objective_current, sample_wires, 128, 1e+1),
	]
	params = optimize(model, params, objectives, n_steps=501)

	# visualize solution
	# plot_2d_0form(domain, model(params))


# def test_maxwell_potential():
# 	dg = Geometry(p=1, q=1)
# 	domain = UnitCube(1+1)
#
# 	# potential 1-form formulation
# 	model, params = make_field_model(
# 		inputs=dg.algebra.subspace.vector(),
# 		outputs=dg.algebra.subspace.vector(),
# 		n_frequencies=64,
# 		n_hidden=64,
# 		# why are we so sensitive to this parameter? poor weight init?
# 		scale=1e1,
# 	)
# 	# visualize initial random starting field
# 	#  FIXME: plot timelike slices?
# 	plot_2d_0form(domain, model(params).exterior_derivative().dual())
#
# 	# satisfy maxwell equations in interior of domain
# 	def objective_internal(model: Field, x):
# 		J = 0   # free field equation
# 		return huber_loss(
# 			model.exterior_derivative().dual().exterior_derivative()(x),
# 			J
# 		)
# 	# force a traveling wave at t=0
# 	t0_sampler = lambda key: jax.random.uniform(key, shape=(1+1,), minval=-1, maxval=+1) * jnp.array([1, 0])
# 	boundary_function = lambda x: jnp.array([jnp.cos(x[0]*10), jnp.sin(x[0]*10)])
# 	def objective_boundary(model, x):
# 		return huber_loss(
# 			model(x),
# 			boundary_function(x),
# 			1e-2
# 		)
#
# 	objectives = [
# 		(objective_internal, domain.sample_interior, 128, 1e+0),
# 		# (objective_boundary, t0_sampler, 128, 1e+2),
# 	]
# 	params = optimize(model, params, objectives, n_steps=101, learning_rate=3e-4)
#
# 	# visualize solution
# 	plot_2d_0form(domain, model(params).exterior_derivative().dual())
