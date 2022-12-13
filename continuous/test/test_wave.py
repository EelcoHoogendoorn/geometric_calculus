import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from optax import huber_loss

from continuous.domain import *
from continuous.models import make_field_model
from continuous.geometry import Geometry
from continuous.optimize import optimize
from continuous.field import Field

from numga.algebra.algebra import Algebra
from numga.backend.jax.context import JaxContext


def classical_wave_equation(phi: Field) -> Field:
	"""scalar wave equation"""
	# return phi.exterior_derivative().dual().exterior_derivative().dual_inverse()     # 0-form
	# return phi.exterior_derivative().interior_derivative()     # 0-form
	# return phi.laplacian()
	# return phi.geometric_derivative().anti_geometric_derivative()     # 0-form
	return phi.exterior_derivative().anti_interior_derivative()     # 0-form

def relativistic_wave_equation(phi: Field) -> Field:
	"""scalar wave equation"""
	return phi.exterior_derivative().interior_derivative()     # 0-form


def test_wave():
	algebra = Algebra.from_str('x+t0')
	geometry = Geometry(algebra)
	domain = UnitCube(geometry)
	context = JaxContext(algebra)
	mv = context.multivector


	# # create a parametric function that maps points in the domain, to a (dict of) multivector fields
	# model, params = make_field_model(
	# 	geometry=geometry,
	# 	inputs=geometry.domain,
	# 	# bivector potential
	# 	outputs=geometry.algebra.subspace.scalar(),
	# 	n_frequencies=64,
	# 	n_hidden=[64]*2,
	# 	# why are we so sensitive to this parameter? poor weight init?
	# 	scale=1e0,
	# )
	# visualize initial random starting field

	k = (mv.x * 5 + mv.t * 5)
	print(k.norm())
	# FIXME: implement better interop between field and multivectors so this isnt so hideous!
	g = lambda x: jnp.exp(-((x ** 2) / 1e-1).sum())*1
	plane_wave = geometry.k_field(lambda x: jnp.cos(mv.vector(x).regressive(k).values) + g(x), k=0)

	plt.figure()
	plot_2d_0form(domain, plane_wave)
	residual = classical_wave_equation(plane_wave)
	print(residual.subspace)
	plt.figure()
	plot_2d_0form(domain, residual)
	plt.show()
	# print(x.inner(k))
	return

	def plane_wave(k):
		def inner(x):
			# FIXME: dont want to lose t to degeneracy.
			return jnp.cos(x.regressive(k))
		return inner

	plot_2d_0form_contour(domain, model(params))
	plot_2d_0form_contour(domain, residual)
