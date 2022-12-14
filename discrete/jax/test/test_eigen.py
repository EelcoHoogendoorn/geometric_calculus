
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

def smooth_noise(field, sigma):
	arr = np.random.normal(size=field.arr.shape)
	import scipy
	sigma = [0] + [sigma] * (field.dimensions - 1) + [0]
	arr = scipy.ndimage.gaussian_filter(arr, sigma=sigma, mode='wrap')
	return field.copy(arr=jnp.array(arr))


def solve_eigen(field, l, mass):
	"""solve first order eigenproblem via periodic discretized time axis

	not very fast; and not sure its bugfree either
	the nice thing is that we need not make any assumption about temporal axis
	"""
	def func(params):
		arr, l = params
		q = field.copy(arr=arr)
		residual = q.geometric_derivative(metric={'t': l})
		res = residual.arr - arr * mass
		return optax.huber_loss(res, delta=1).sum() + optax.huber_loss((arr**2).sum(), 1, delta=1)

	l = jnp.array(l)
	params = field.arr, l

	import optax
	learning_rate = 3e-3
	tx = optax.adam(learning_rate=learning_rate)
	opt_state = tx.init(params)
	loss_grad_fn = jax.jit(jax.value_and_grad(func))
	for i in range(10001):
		if i % 100 == 0:
			learning_rate *= 0.7
		loss_val, grads = loss_grad_fn(params)
		updates, opt_state = tx.update(grads, opt_state)
		params = optax.apply_updates(params, updates)
		if i % 10 == 0:
			print('Loss step {}: '.format(i), loss_val)

	new_field, new_l = params
	print(new_l)
	field = field.copy(arr=new_field)
	slc = field.slice(0)
	return slc, new_l


from discrete.jax.test.test_leapfrog import write_gif_1d, write_gif_2d
from numga.algebra.algebra import Algebra

def test_first_order_wave_11():
	"""compute eigenmodes by expliticly constructing a periodic time domain"""
	algebra = Algebra.from_str('x+t-')
	shape = (64, 2)
	field = Field.from_subspace(algebra.subspace.full(), shape)
	field = smooth_noise(field, 4)
	mass = 1 - field.gauss(jnp.array([0.4, 1e16])) / 2

	slc, new_l = solve_eigen(field, 5.0, mass)

	mass = mass[..., 0]
	anim = slc.rollout(64, metric={'t': new_l / 128}, mass=mass)
	write_gif_1d('../../output', anim, 'x_t_xt', post='eigen', norm=99, gamma=False)


def test_first_order_wave_21():
	"""compute eigenmodes by expliticly constructing a periodic time domain"""
	algebra = Algebra.from_str('x+y+t-')
	shape = (64, 64, 4)
	field = Field.from_subspace(algebra.subspace.full(), shape)
	field = smooth_noise(field, 1)
	mass = 1 - field.gauss(jnp.array([0.6, 0.6, 1e16])) / 3

	slc, new_l = solve_eigen(field, 40., mass)

	mass = mass[..., 0]
	# NOTE: need division here, since we have dt metric term on 'wrong side' in leapfrog
	anim = slc.rollout(64, metric={'t': 8/new_l}, mass=mass)
	write_gif_2d('../../output', anim, 'xy_xt_yt', post='eigen', norm=99, gamma=False)
