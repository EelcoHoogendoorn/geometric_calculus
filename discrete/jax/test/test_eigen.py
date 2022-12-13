
import jax
import jax.numpy as jnp
import numpy as np
import scipy
from discrete.jax.field import Field


def eigen(lhs, rhs, x, M=None, nullspace=None):
	"""lhs and rhs wrap flatten vector interface"""

	# if preconditioner == 'amg':
	# 	M = self.amg_solver.aspreconditioner()
	# else:
	# 	M = preconditioner

	# NOTE: zero-centered initialization is apparently important here!
	X = np.random.normal(size=(A.shape[0], K))

	# monkey patch this dumb assert
	from scipy.sparse.linalg.eigen.lobpcg import lobpcg
	lobpcg._assert_symmetric = lambda x: None

	try:
		if nullspace is None:
			Y = self.null_space  # giving lobpcg the null space helps with numerical stability
		if nullspace is False:
			Y = None
		# assert is_symmetric(A)
		# assert is_symmetric(B)
		v, V = scipy.sparse.linalg.lobpcg(A=A, B=B, X=X, tol=tol, M=M, Y=Y, largest=False, verbosityLevel=1)
	except:
		v, V = scipy.sparse.linalg.lobpcg(A=A, B=B, X=X, tol=tol, M=M, largest=False, verbosityLevel=1)

	return V, v


def test_scalar_wave_1d():
	# FIXME: set up variable material property fields
	#  would that help with funny linear propagation? why so lnear now?
	from numga.algebra.algebra import Algebra
	algebra = Algebra.from_str('x+')
	shape = (64,)
	displacement = Field.from_subspace(algebra.subspace.scalar(), shape)
	velocity = displacement * 0
	x = displacement.meshgrid()
	x2 = (x ** 2).sum(axis=0, keepdims=False)
	gauss = jnp.exp(-x2*10*2)
	displacement.arr = displacement.arr.at[0].set(gauss)
	lame = Field.from_subspace(algebra.subspace.vector(), shape)
	window = jnp.all(jnp.abs(x) < 0.5, axis=0) + 1e-2
	lame.arr = lame.arr.at[0].set(window)
	density = Field.from_subspace(algebra.subspace.scalar(), shape)
	density = density.arr.at[0].set(window)
	# density = density.arr.at[1].set(window)
	# print(field.arr.shape)
	# return
	@jax.jit
	def step(displacement, velocity):
		grad = displacement.geometric_derivative()
		force = (grad*lame).geometric_derivative(displacement.subspace)
		velocity = velocity + force * 0.33
		return displacement + velocity, velocity

	@jax.jit
	def lhs(displacement):
		grad = displacement.geometric_derivative()
		return (grad*lame).geometric_derivative(displacement.subspace)
	def rhs(dispacement):
		return displacement


	frames = []
	for i in range(90):
		displacement, velocity = step(displacement, velocity)
		frames.append(displacement.arr.T)
		print(jnp.linalg.norm(velocity.arr))

	import matplotlib.pyplot as plt

	path = r'../../output/wave_scalar_1d_0'
	import imageio.v3 as iio
	import os
	os.makedirs(path, exist_ok=True)
	frames = jnp.array(frames)
	frames = jnp.abs(frames)
	print(frames.shape)
	# frames = frames / 0.15  # arr.max()
	frames = frames / jnp.percentile(frames.flatten(), 95)
	frames = jnp.clip(frames * 255, 0, 255).astype(jnp.uint8)
	frames = np.array(frames)
	print(frames.shape, type(frames))
	iio.imwrite(os.path.join(path, 'anim.gif'), frames[..., [0, 0, 0]])
	# return
	# fig, ax = plt.subplots()
	# ax.imshow(animation)
	# if False:
	# 	fig, ax = plt.subplots(2, 1, squeeze=False)
	# 	for i in range(1):
	# 		ax[0, i].plot(grad.arr[i])
	# 		ax[1, i].plot(force.arr[i])
	# 		# ax[i].colorbar()
	# plt.show()
