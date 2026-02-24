"""Generate article figures"""

import jax.numpy as jnp
import numpy as np

from numga.algebra.algebra import Algebra
from discrete.jax.field_slice import FieldSlice


def test_1d():
	algebra = Algebra.from_str('x+t-')
	shape, steps = (256, ), 256
	field = FieldSlice.from_subspace(algebra.subspace.multivector(), shape)
	field = field.random_gaussian(0.1, seed=0)
	field = field.rollout(steps)
	field.write_gif_1d('.', 'x_t_xt', pre='flat')


def test_2d_compact():
	algebra = Algebra.from_str('w+x+t-')
	shape, steps = (2, 256), 256
	field = FieldSlice.from_subspace(algebra.subspace.multivector(), shape)
	field = field.random_gaussian(0.1, seed=1)
	field = field.rollout(steps)
	field.write_gif_2d_compact('.', 'x_t_xt', pre='flat')


def test_2d():
	algebra = Algebra.from_str('x+y+t-')
	shape, steps = (128, 128), 256
	field = FieldSlice.from_subspace(algebra.subspace.multivector(), shape)
	field = field.random_gaussian(0.1, [0, 0.3])
	mass = 0.4 + field.quadratic() / 2
	field = field.rollout(steps, mass=mass, metric={})
	field.write_gif_2d('.', 'xt_yt_xy', pre='mass_quad', norm=99)
