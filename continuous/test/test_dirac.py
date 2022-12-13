import jax
import jax.numpy as jnp

from calculus.domain import UnitCube
from calculus.geometry import Geometry
from calculus.models import make_field_model


def test_dirac_geometric():
	dg = Geometry(p=3, q=1)
	domain = UnitCube(3+1)

	# create a parametric function that maps points in the domain, to a (dict of) calculus forms
	model, params = make_field_model(
		inputs=dg.algebra.subspace.vector(),
		outputs=dg.algebra.subspace.even_grade(),
		n_frequencies=64,
		n_hidden=64,
		# why are we so sensitive to this parameter? poor weight init?
		scale=1e0,
	)

	x, y, z, t = dg.algebra.subspace.basis()
	from calculus.field import Field
	t = Field(f=lambda x: jnp.array([1]), subspace=t)

	def dirac(phi, m=1):
		return phi.geometric_derivative() - phi.geometric_product(t * m)

	key = jax.random.PRNGKey(0)
	x = domain.sample_interior(key)
	res = dirac(model(params))(x)
	print(res)
