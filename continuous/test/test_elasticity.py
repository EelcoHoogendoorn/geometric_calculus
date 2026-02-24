"""
should we add manual function massaging, for initial bc satisfaction?
tried, but its broken? dont understand why

would also want to contrast bi-potential approach; displacement as phi.di+psi.de
compare that to full-first-order approach? model each field quantity directly
and apply each constraint?
which is most appropriate for high ratios of material parameters?

clearly, phi-only formulation converges a lot more robustly.
0-2 grade potential same thing; though dunno how well it handles stifness
0-2 potential seems to run about twice as slow as grade-2 only potential

should we add a cylinder case?

seems that only incompressible case is really working still...
seems to be an issue with the convergence of our NN based solver,
rather than the fundamental math involved
"""
from typing import Dict

import jax.numpy as jnp
import jax
from optax import huber_loss

from continuous.domain import *
from continuous.models import make_field_model
from continuous.geometry import Geometry
from continuous.optimize import optimize
from continuous.field import Field
from numga.algebra.algebra import Algebra
import matplotlib.pyplot as plt


def elasticity(displacement: Field, mu=1, lamb=1) -> Field:
	"""Elastostatics"""
	compression = displacement.interior_derivative()    # 0-field
	pressure = compression.exterior_derivative()        # 1-field
	rotation = displacement.exterior_derivative()       # 2-field
	shear = rotation.interior_derivative()              # 1-field
	momentum = shear * mu + pressure * lamb             # momentum balance
	return momentum.geometric_derivative()              # lets nuke those gradients


def elasticity_rubber(phi: Field) -> Field:
	"""incompresible elastostatics"""
	displacement = phi.interior_derivative()            # 1-field
	rotation = displacement.exterior_derivative()       # 2-field
	shear = rotation.interior_derivative()              # 1-field
	return shear.exterior_derivative()


def elasticity_multi(fields: Dict[str, Field], mu=.1, lamb=10) -> Field:
	"""elastostatics in multi-component formulation;
	only first order derivatives between loss and fields
	want to be able to compare convergence properties
	"""
	displacement = fields['displacement']
	compression = fields['compression']
	rotation = fields['rotation']

	pressure = compression.exterior_derivative()        # 1-field
	shear = rotation.interior_derivative()              # 1-field
	return (
		(shear * mu + pressure * lamb).geometric_derivative(), # momentum balance
		compression - displacement.interior_derivative(),
		rotation - displacement.exterior_derivative()       # 2-field
	)


def test_elastic_2d_compact():
	"""attempt at most compact formulation of elasticity"""
	geometry = Geometry(2)
	domain = UnitCube(geometry)

	# create a parametric function that maps points in the domain, to a (dict of) multivector fields
	model, params = make_field_model(
		geometry=geometry,
		inputs=geometry.domain,
		outputs=geometry.algebra.subspace.vector(),
		n_frequencies=64,
		n_hidden=[64]*3,
		scale=3e-1,
	)

	delta = jnp.array([0, -1])
	# satisfy bivector potential equation in interior of domain
	def objective_internal(displacement: Field, x):
		return huber_loss(displacement.gd().gd().gd()(x), 0, 1e-6)
	# hydraulic press; prescribe displacement along one axis
	def objective_boundary(displacement: Field, x):
		return huber_loss(displacement(x), domain.which_side(x, d=1) * delta, 1e-6)

	objectives = [
		(objective_internal, domain.sample_interior, 256, 1e-0),
		(objective_boundary, domain.sample_boundary_axis(1), 64, 1e+1),
	]

	params = optimize(model, params, objectives, n_steps=301)

	displacement = model(params)
	plot_2d_1field_grid(domain, displacement)
	plt.show()


def test_elastic_2d_rubber_compression():
	"""incompressible elasticity"""
	geometry = Geometry(2)
	domain = UnitCube(geometry)

	dir = 1
	delta = jnp.array([0, -1])

	# FIXME: this window applies to displacement, not potential!
	def window(f, x):
		# return f
		c = x[dir]
		q = (1+c) * (1-c)
		return f + c * delta
	# create a parametric function that maps points in the domain, to a (dict of) multivector fields
	model, params = make_field_model(
		geometry=geometry,
		inputs=geometry.domain,
		outputs=geometry.algebra.subspace.bivector(),
		n_frequencies=64,
		n_hidden=[64]*3,
		scale=3e-1,
		# window=window,
	)

	if False:
		# visualize initial random starting field
		displacement = model(params).interior_derivative()
		plt.figure()
		plot_2d_1field_grid(domain, displacement)
		plt.figure()
		plot_2d_0field(domain, displacement.exterior_derivative().dual())
		plt.show()

	# satisfy NS in interior of domain
	def objective_internal(phi: Field, x):
		return huber_loss(
			elasticity_rubber(phi)(x),
			0,
			1e-6
		)
	# hydraulic press; prescribe displacement along one axis
	def objective_boundary(phi: Field, x):
		return huber_loss(
			phi.interior_derivative()(x),
			domain.which_side(x, dir) * delta,
			1e-6
		)

	objectives = [
		(objective_internal, domain.sample_interior, 256, 1e-0),
		(objective_boundary, domain.sample_boundary_axis(dir), 64, 1e+1),
	]

	# plot_sampling(objectives)

	import time
	t = time.time()
	params = optimize(model, params, objectives, n_steps=301)
	print('time', time.time() - t)

	if True:
		# # visualize solution
		displacement = model(params).interior_derivative()
		plt.figure()
		plot_2d_1field_grid(domain, displacement)
		plt.figure()
		plot_2d_0field(domain, displacement.exterior_derivative().dual())
		plt.show()


def test_elastic_2d_rubber_shear():
	"""incompressible elasticity"""
	geometry = Geometry(2)
	domain = UnitCube(geometry)

	dir = 1
	delta = jnp.array([1, 0])

	# create a parametric function that maps points in the domain, to a (dict of) multivector fields
	model, params = make_field_model(
		geometry=geometry,
		inputs=geometry.domain,
		outputs=geometry.algebra.subspace.bivector(),
		n_frequencies=64,
		n_hidden=[64]*3,
		scale=3e-1,
	)

	if False:
		# visualize initial random starting field
		displacement = model(params).interior_derivative()
		plt.figure()
		plot_2d_1field_grid(domain, displacement)
		plt.figure()
		plot_2d_0field(domain, displacement.exterior_derivative().dual())
		plt.show()

	# satisfy NS in interior of domain
	def objective_internal(phi: Field, x):
		return huber_loss(
			elasticity_rubber(phi)(x),
			0,
			1e-6
		)
	# hydraulic press; prescribe displacement along one axis
	def objective_boundary(phi: Field, x):
		return huber_loss(
			phi.interior_derivative()(x),
			domain.which_side(x, d=dir) * delta,
			1e-6
		)

	objectives = [
		(objective_internal, domain.sample_interior, 256, 1e-0),
		(objective_boundary, domain.sample_boundary_axis(dir), 64, 1e+1),
	]

	# plot_sampling(objectives)

	import time
	t = time.time()
	params = optimize(model, params, objectives, n_steps=301)
	print('time', time.time() - t)

	if True:
		# # visualize solution
		displacement = model(params).interior_derivative()
		plt.figure()
		plot_2d_1field_grid(domain, displacement)
		plt.figure()
		plot_2d_0field(domain, displacement.exterior_derivative().dual())
		plt.show()


def test_elastic_2d_potential02():
	"""0-2 grade potential elasticity"""
	geometry = Geometry(2)
	domain = UnitCube(geometry)

	dir = 0
	delta = -jnp.eye(domain.n)[-1]

	def window(f, c):
		# return f
		x, y = c
		q = (1+x) * (1-x)
		return f * q + x * delta
	# create a parametric function that maps points in the domain, to a (dict of) multivector fields
	model, params = make_field_model(
		geometry=geometry,
		inputs=geometry.domain,
		outputs=geometry.algebra.subspace.from_grades([0, 2]),
		n_frequencies=64,
		n_hidden=[64]*3,
		scale=3e-1,
		# window=window,
	)
	def get_displacement(potential):
		return potential.geometric_derivative(subspace=geometry.algebra.subspace.vector())

	if False:
		# visualize initial random starting field
		displacement = get_displacement(model(params))
		plt.figure()
		plot_2d_1field_grid(domain, displacement)
		plt.figure()
		plot_2d_0field(domain, displacement.exterior_derivative().dual())
		plt.figure()
		plot_2d_0field(domain, displacement.interior_derivative())
		plt.show()

	# satisfy biharmonic potential in interior of domain
	def objective_internal(phi: Field, x):
		return huber_loss(
			elasticity(get_displacement(phi), mu=1, lamb=1)(x),
			0,
			1e-6
		)
	# hydraulic press; prescribe displacement along one axis
	def objective_boundary(phi: Field, x):
		return huber_loss(
			get_displacement(phi)(x),
			domain.which_side(x, d=dir) * delta,
			1e-6
		)

	objectives = [
		(objective_internal, domain.sample_interior, 256, 1e-0),
		(objective_boundary, domain.sample_boundary_axis(dir), 64, 1e+1),
	]

	# plot_sampling(objectives)

	import time
	t = time.time()
	params = optimize(model, params, objectives, n_steps=301)
	print('time', time.time() - t)


	if True:
		# # visualize solution
		displacement = get_displacement(model(params))
		plt.figure()
		plot_2d_1field_grid(domain, displacement)
		plt.figure()
		plot_2d_0field(domain, displacement.exterior_derivative().dual())
		plt.figure()
		plot_2d_0field(domain, displacement.interior_derivative())
		plt.show()


def test_elastic_2d():
	geometry = Geometry(2)
	domain = UnitCube(geometry)

	delta = jnp.array([0, -1])

	# FIXME: still fail to understand why this works so awefully
	def window(f, x):
		c = x[0]
		q = (1+c) * (1-c)
		return f + c * delta
	# create a parametric function that maps points in the domain, to a (dict of) multivector fields
	model, params = make_field_model(
		geometry=geometry,
		inputs=geometry.domain,
		outputs=geometry.algebra.subspace.vector(),
		n_frequencies=64,
		n_hidden=[64]*3,
		scale=3e-1,
	)

	if True:
		# visualize initial random starting field
		displacement = model(params)
		plt.figure()
		plot_2d_1field_grid(domain, displacement)
		plt.figure()
		plot_2d_0field(domain, displacement.interior_derivative())
		plt.figure()
		plot_2d_0field(domain, displacement.exterior_derivative().dual())
		plt.show()

	# satisfy NS in interior of domain
	def objective_internal(displacement: Field, x):
		return huber_loss(
			elasticity(displacement, mu=1e-1, lamb=1e+1)(x),
			0,
			1e-6
		)
	# hydraulic press; prescribe displacement along one axis
	def objective_boundary(displacement: Field, x):
		return huber_loss(
			displacement(x),
			domain.which_side(x, d=0) * delta,
			1e-6
		)

	objectives = [
		(objective_internal, domain.sample_interior, 256, 1e-0),
		(objective_boundary, domain.sample_boundary_axis(0), 64, 1e+1),
	]

	import time
	t = time.time()
	params = optimize(model, params, objectives, n_steps=1001)
	print('time', time.time() - t)


	if True:
		# # visualize solution
		displacement = model(params)
		plt.figure()
		plot_2d_1field_grid(domain, displacement)
		plt.figure()
		plot_2d_0field(domain, displacement.interior_derivative())
		plt.figure()
		plot_2d_0field(domain, displacement.exterior_derivative().dual())
		plt.show()


def test_elastic_2d_multi():
	geometry = Geometry(2)
	domain = UnitCube(geometry)

	dir = 0
	delta = jnp.array([1, 0])

	def window(f, c):
		# return f
		x, y = c
		q = (1+x) * (1-x)
		return f * q + x * delta
	# create a parametric function that maps points in the domain, to a (dict of) multivector fields
	model, params = make_field_model(
		geometry=geometry,
		inputs=geometry.domain,
		outputs={
			'displacement': geometry.algebra.subspace.vector(),
			'compression': geometry.algebra.subspace.scalar(),
			'rotation': geometry.algebra.subspace.bivector(),
		},
		n_frequencies=32,
		n_hidden=[64]*3,
		scale=1e0,
		# window=window,
	)

	if True:
		# visualize initial random starting field
		fields = model(params)
		plt.figure()
		plot_2d_1field_grid(domain, fields['displacement'])
		plt.figure()
		plot_2d_0field(domain, fields['rotation'].dual())
		plt.figure()
		plot_2d_0field(domain, fields['compression'])
		plt.show()

	# satisfy NS in interior of domain
	def objective_internal(fields: Field, x):
		residuals = elasticity_multi(fields)
		return sum(huber_loss(
			r(x),
			0,
			1e-6
		).sum() for r in residuals)
	# hydraulic press; prescribe displacement along one axis
	# boundary_function = lambda x: domain.is_max(x, d=0) * jnp.eye(domain.n)[0]
	# boundary_function = lambda x: domain.is_max(x, d=0) * jnp.ones(domain.n)
	def objective_boundary(fields: Field, x):
		return huber_loss(
			fields['displacement'](x),
			domain.which_side(x, dir) * delta,
			1e-6
		)

	objectives = [
		(objective_internal, domain.sample_interior, 256, 1e-0),
		(objective_boundary, domain.sample_boundary_axis(dir), 64, 1e+1),
	]

	# plot_sampling(objectives)

	import time
	t = time.time()
	params = optimize(model, params, objectives, n_steps=301)
	print('time', time.time() - t)

	if True:
		# # visualize solution
		fields = model(params)
		plt.figure()
		plot_2d_1field_grid(domain, fields['displacement'])
		plt.figure()
		plot_2d_0field(domain, fields['rotation'].dual())
		plt.figure()
		plot_2d_0field(domain, fields['compression'])
		plt.show()
