# Geometric Calculus
Geometric Calculus over continuous fields, leveraging JAX autodiff


## Examples

Using this library, many continuous physical field equations can be modelled
```python
def navier_stokes(phi: Field, Re=100) -> Field:
	"""incompressible navier-stokes"""
	velocity = phi.interior_derivative()            # 1-form
	vorticity = velocity.exterior_derivative()      # 2-form
	shear = vorticity.interior_derivative()         # 1-form
	diffusion = shear.exterior_derivative()         # 2-form
	advection = velocity.directional_derivative(vorticity)
	return diffusion + Re * advection               # momentum balance
```

We may fund solutions to such equations through opimization of the weights of a neural network describing `phi`.


TODO
====
unify concrete multivector and field types
A lot of stuff really; this is very much some thrown together experiments