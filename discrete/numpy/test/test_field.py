"""
3d volume rendered smoke.
https://www.shadertoy.com/view/wlG3RW
https://www.shadertoy.com/view/3lffD8
"""

from discrete.numpy.field import *
import numpy as np

import os
import matplotlib.pyplot as plt


def save_animation(path, frames, overwrite=False, dpi=300):
    if path is not None:
        try:
            os.makedirs(path, exist_ok=True)
        except:
            if not overwrite:
                raise

    for i in range(frames):
        yield i
        if path is not None:
            plt.savefig(os.path.join(path, 'frame_{i}.png'.format(i=i)), dpi=dpi)
            plt.close()
            print('saved frame {i}'.format(i=i))


def test_bivec2d():
	field = BivectorField3(shape=(256, 256))
	# field.arr[0, 128, 128] = 1
	x = field.meshgrid()
	field.arr[0] = np.exp(-(x**2 * 30**2).sum(0))
	for i in range(100):
		field.step()
	arr = (field.arr[0])
	plt.imshow(arr)
	plt.colorbar()
	plt.show()


def test_dirac_2():
	field = RotorField3((64, 64))
	x = field.meshgrid()
	field.arr[0] = np.exp(-(x**2 * 4**2).sum(0))
	field.arr[1] = np.exp(-(x**2 * 4**2).sum(0))

	path = r'../output/dirac_gauss_2_mass'

	import imageio.v3 as iio

	frames = []
	for i in range(500):
		arr = (field.arr[1:])
		frames.append(np.copy(arr.T))
		for _ in range(1):
			field.step()

	os.makedirs(path, exist_ok=True)
	frames = np.array(frames)
	frames = np.abs(frames)
	# frames = frames / 0.15  # arr.max()
	frames = frames / np.percentile(frames.flatten(), 95)
	frames = np.clip(frames * 255, 0, 255).astype(np.uint8)
	iio.imwrite(os.path.join(path, 'anim.gif'), frames)

	# plt.colorbar()
	# plt.show()


def test_maxwell_3():
	field = BivectorField4((64, 64, 64))
	x = field.meshgrid()
	field.arr[0] = np.exp(-(x**2 * 4**2).sum(0))
	field.arr[4] = np.exp(-(x**2 * 4**2).sum(0))
	field.arr[1] = np.exp(-(x**2 * 4**2).sum(0))
	field.arr[2] = np.exp(-(x**2 * 4**2).sum(0)) / 2
	# field.arr[7] = np.exp(-(x**2 * 4**2).sum(0))

	path = r'../output/maxwell_31_mass'

	import imageio.v3 as iio
	# import matplotlib.pyplot as plt
	# img = iio.imread('imageio:astronaut.png')

	frames = []
	for i in range(500):
	# for i in save_animation(path, 30, overwrite=True):
	# 	plt.figure()
		arr = (field.arr[:3, :, 32])   # a y-slice
	# 	plt.imshow(arr.T)
		frames.append(np.copy(arr.T))
		for _ in range(1):
			field.step_mass()

	os.makedirs(path, exist_ok=True)
	frames = np.array(frames)
	frames = np.abs(frames)
	# frames = frames / 0.15  # arr.max()
	frames = frames / np.percentile(frames.flatten(), 95)
	frames = np.clip(frames * 255, 0, 255).astype(np.uint8)
	iio.imwrite(os.path.join(path, 'anim.gif'), frames)

	# plt.colorbar()
	# plt.show()


def test_dirac_3():
	field = RotorField4((64, 64, 64))
	x = field.meshgrid()
	field.arr[0] = np.exp(-(x**2 * 4**2).sum(0))
	field.arr[4] = np.exp(-(x**2 * 4**2).sum(0))
	field.arr[1] = np.exp(-(x**2 * 4**2).sum(0))
	field.arr[2] = np.exp(-(x**2 * 4**2).sum(0)) / 2
	field.arr[7] = np.exp(-(x**2 * 4**2).sum(0))

	path = r'../output/dirac_gauss_31_mass'

	import imageio.v3 as iio
	# import matplotlib.pyplot as plt
	# img = iio.imread('imageio:astronaut.png')

	frames = []
	for i in range(500):
	# for i in save_animation(path, 30, overwrite=True):
	# 	plt.figure()
		arr = (field.arr[4:7, :, 32])   # a y-slice
	# 	plt.imshow(arr.T)
		frames.append(np.copy(arr.T))
		for _ in range(1):
			field.step_mass()

	os.makedirs(path, exist_ok=True)
	frames = np.array(frames)
	frames = np.abs(frames)
	# frames = frames / 0.15  # arr.max()
	frames = frames / np.percentile(frames.flatten(), 95)
	frames = np.clip(frames * 255, 0, 255).astype(np.uint8)
	iio.imwrite(os.path.join(path, 'anim.gif'), frames)

	# plt.colorbar()
	# plt.show()


def test_dirac_3_odd():
	field = OddField4((64, 64, 64))
	x = field.meshgrid()
	field.arr[0] = np.exp(-(x**2 * 4**2).sum(0))
	field.arr[4] = np.exp(-(x**2 * 4**2).sum(0))
	field.arr[1] = np.exp(-(x**2 * 4**2).sum(0))
	field.arr[2] = np.exp(-(x**2 * 4**2).sum(0)) / 2
	field.arr[7] = np.exp(-(x**2 * 4**2).sum(0))

	path = r'../output/dirac_odd_gauss'

	import imageio.v3 as iio
	# import matplotlib.pyplot as plt
	# img = iio.imread('imageio:astronaut.png')

	frames = []
	for i in range(500):
	# for i in save_animation(path, 30, overwrite=True):
	# 	plt.figure()
		arr = (field.arr[:3, :, 32])   # a y-slice
	# 	plt.imshow(arr.T)
		frames.append(np.copy(arr.T))
		for _ in range(1):
			field.step()

	os.makedirs(path, exist_ok=True)
	frames = np.array(frames)
	frames = np.abs(frames)
	# frames = frames / 0.15  # arr.max()
	frames = frames / np.percentile(frames.flatten(), 95)
	frames = np.clip(frames * 255, 0, 255).astype(np.uint8)
	iio.imwrite(os.path.join(path, 'anim.gif'), frames)

	# plt.colorbar()
	# plt.show()


def test_multivector_4():
	field = MultiVectorField4((64, 64, 64))
	x = field.meshgrid()
	gauss = np.exp(-(x ** 2 * 4 ** 2).sum(0))
	field.arr = np.random.normal(0 ,1, (16, 1, 1, 1)) * gauss

	path = r'../output/mv_gauss_31_mass'

	import imageio.v3 as iio
	# import matplotlib.pyplot as plt
	# img = iio.imread('imageio:astronaut.png')

	frames = []
	for i in range(500):
	# for i in save_animation(path, 30, overwrite=True):
	# 	plt.figure()
		arr = (field.arr[5:8, :, 32])   # a y-slice
	# 	plt.imshow(arr.T)
		frames.append(np.copy(arr.T))
		for _ in range(1):
			field.step_mass(m=-0.2)

	os.makedirs(path, exist_ok=True)
	frames = np.array(frames)
	frames = np.abs(frames)
	# frames = frames / 0.15  # arr.max()
	frames = frames / np.percentile(frames.flatten(), 95)
	frames = np.clip(frames * 255, 0, 255).astype(np.uint8)
	iio.imwrite(os.path.join(path, 'anim_neg.gif'), frames)


def test_multivector_4_compactified():
	compact = 2
	field = MultiVectorField4((compact, 256, 256))
	x = field.meshgrid()

	gauss = np.exp(-(x[1:] ** 2 * 32 ** 2).sum(0, keepdims=True))
	# with all components in V summed
	V = (x[:] ** 2).sum(0) *1.5
	# dropping the harsh dV over the compact dim
	V = (x[1:] ** 2).sum(0) *1.5
	# gauss = np.exp(-(x ** 2 * 4 ** 2).sum(0))
	# field.arr = np.random.normal(0 ,1, (16, 1, 1, 1)) * gauss

	for s in [40]:#, -20, 20, 150]:
		q = np.random.normal(0, 1, (16, compact, 1, 1))
		q = q / np.linalg.norm(q) * gauss
		# print(q.shape)
		f = np.roll(q, s, axis=-1)
		field.arr += f
	mixing = np.ones((compact,compact)) + np.eye(compact)
	field.arr = np.einsum('cxyz,xo->coyz', field.arr, mixing)

	path = r'../../output/mv_gauss_31_compact_256_4'

	import imageio.v3 as iio

	frames = []
	for i in range(500):
	# for i in save_animation(path, 30, overwrite=True):
	# 	plt.figure()
		arr = (field.arr[8:11, 0])   # a x-slice
	# 	plt.imshow(arr.T)
		frames.append(np.copy(arr.T))
		print(np.linalg.norm(arr))
		for _ in range(2):
			field.step_mass(m=0.0+V)

	os.makedirs(path, exist_ok=True)
	frames = np.array(frames)
	frames = np.abs(frames)
	# frames = frames / 0.15  # arr.max()
	frames = frames / np.percentile(frames.flatten(), 95) / 4
	frames = (np.sqrt(np.clip(frames, 0, 1)) *255).astype(np.uint8)
	iio.imwrite(os.path.join(path, 'anim.gif'), frames)


def test_multivector_3():
	field = MultiVectorField3((128, 128))
	x = field.meshgrid()
	gauss = np.exp(-(x ** 2 * 8 ** 2).sum(0))
	# FIXME: note full coupling like this is graviry potential; not EM!
	V = (x ** 2).sum(0) / 2
	for s in [-20, 20]:
		q = np.random.normal(0, 1, (8, 1, 1))
		q = q / np.linalg.norm(q) * gauss
		f = np.roll(q, s, axis=-1) #- np.roll(q, -64, axis=-1)) *1
		field.arr += f

	path = r'../output/mv_gauss_21_mass_q0'

	import imageio.v3 as iio

	frames = []
	for i in range(500):
		arr = (field.arr[5:8])  # bivector part
		frames.append(np.copy(arr.T))
		for _ in range(1):
			field.step_mass_q(m=0.4+V, q=0.0)

	os.makedirs(path, exist_ok=True)
	frames = np.array(frames)
	frames = np.abs(frames) * 4
	# frames = frames / np.percentile(frames.flatten(), 99)
	frames = np.clip(frames * 255, 0, 255).astype(np.uint8)
	iio.imwrite(os.path.join(path, 'anim.gif'), frames)


def test_multivector_3_compactified():
	compact = 32

	field = MultiVectorField3((compact, 1024))
	x = field.meshgrid()[:, :, :]
	gauss = np.exp(-(x[1:] ** 2 * 32 ** 2).sum(0, keepdims=True))
	V = (x[1:] ** 2).sum(0) *1.5
	print(V.shape)
	# print(V[0].min(),V[0].max())
	# print(V[1].min(),V[1].max())

	i = 0
	for s in [-100]:#, -20, 20, 150]:
		i = (i+1) % 2
		q = np.random.normal(0, 1, (8, 1, 1)) + np.random.normal(0, 1, (8, compact, 1))
		q = q / np.linalg.norm(q) * gauss
		# print(q.shape)
		f = np.roll(q, s, axis=-1) #- np.roll(q, -64, axis=-1)) *1
		# print(f.shape)
		# print(field.arr.shape)
		# field.arr[:, i] += f[:, i]
		field.arr += f

	path = r'../../output/mv_gauss_21c1_4'

	import imageio.v3 as iio

	frames = []
	for i in range(1024):
		arr = (field.arr[5:8, 0])  # bivector part
		print(np.linalg.norm(arr))
		frames.append(np.copy(arr.T))
		for _ in range(2):
			field.step_mass_q(m=0.0+V*0, q=0.0)

	os.makedirs(path, exist_ok=True)
	frames = np.array(frames)[::-1]
	frames = np.abs(frames) * 4
	# frames = frames / np.percentile(frames.flatten(), 99)
	frames = np.clip(frames * 255, 0, 255).astype(np.uint8)
	iio.imwrite(os.path.join(path, 'anim.gif'), frames)


def test_multivector_2():
	field = MultiVectorField2((1024,))
	x = field.meshgrid()
	gauss = np.exp(-(x ** 2 * 32 ** 2).sum(0))
	V = x[0] ** 2 / 4
	print(V.shape, V.max())
	rev = np.array([1, 1, 1, -1])[:, None]
	dual = np.array([1, 1, -1, 1])[:, None]
	for s in [-148, -0, 100]:
		q = np.random.normal(0, 1, (4, 1))
		q = q / np.linalg.norm(q) * gauss
		f = np.roll(q, s, axis=-1) #- np.roll(q, -64, axis=-1)) *1
		field.arr += f
	# q = np.random.normal(0, 1, (4, 1))
	# q = q / np.linalg.norm(q) * gauss
	# f = np.roll(q, -64, axis=-1) #- np.roll(q, -64, axis=-1)) *1
	# field.arr += f

	path = r'../../output/mv_gauss_11_g_0'

	import imageio.v3 as iio

	frames = []
	for i in range(1000):
		arr = (field.arr[0:-1])  # bivector part
		frames.append(np.copy(arr.T))
		# FIXME: m=3 seems to be the max?
		for _ in range(2):
			field.step_mass(m=0 + V, q=0.00)

	os.makedirs(path, exist_ok=True)
	frames = np.array(frames)[::-1]
	print(frames.shape)
	frames = np.abs(frames)
	# frames = frames / np.percentile(frames.flatten(), 95)
	frames = frames * 2
	frames = np.clip(frames * 255, 0, 255).astype(np.uint8)
	iio.imwrite(os.path.join(path, 'anim.gif'), frames)


def test_multivector_2_blackhole():
	field = MultiVectorField2((1024,))
	x = field.meshgrid()
	gauss = lambda sigma: np.exp(-(x ** 2 * sigma ** 2).sum(0))
	# V = x[0] ** 2 / 4
	V = (1 - gauss(4)) /2
	print(V.shape, V.max())
	rev = np.array([1, 1, 1, -1])[:, None]
	dual = np.array([1, 1, -1, 1])[:, None]
	for s in [-150, 50]:
		q = np.random.normal(0, 1, (4, 1))
		q = q / np.linalg.norm(q) * gauss(32)
		f = np.roll(q, s, axis=-1) #- np.roll(q, -64, axis=-1)) *1
		field.arr += f
	# q = np.random.normal(0, 1, (4, 1))
	# q = q / np.linalg.norm(q) * gauss
	# f = np.roll(q, -64, axis=-1) #- np.roll(q, -64, axis=-1)) *1
	# field.arr += f

	path = r'../../output/mv_gauss_11_gbh_0'

	import imageio.v3 as iio

	frames = []
	for i in range(1000*2):
		arr = (field.arr[0:-1])  # bivector part
		frames.append(np.copy(arr.T))
		# FIXME: m=3 seems to be the max?
		for _ in range(2):
			field.step_mass(m=0 + V, q=0.00)

	os.makedirs(path, exist_ok=True)
	frames = np.array(frames)[::-1]
	print(frames.shape)
	frames = np.abs(frames)
	# frames = frames / np.percentile(frames.flatten(), 95)
	frames = frames * 2
	frames = np.clip(frames * 255, 0, 255).astype(np.uint8)
	iio.imwrite(os.path.join(path, 'anim.gif'), frames)


def test_multivector_2_dual():
	field = DualMultiVectorField2((1024,))
	x = field.meshgrid()
	gauss = lambda sigma: np.exp(-(x ** 2 * sigma ** 2).sum(0))
	# V = (x ** 2).sum(axis=0) / 4
	V = (1 - gauss(1)) / 3
	print(V.shape, V.max())
	# rev = np.array([1, 1, 1, -1])[:, None]
	# dual = np.array([1, 1, -1, 1])[:, None]
	for s in [-150]:
		for pd in [0, 1]:
			q = np.random.normal(0, 1, (4, 1))
			q = q / np.linalg.norm(q) * gauss(32)
			f = np.roll(q, s+pd*10, axis=-1) #- np.roll(q, -64, axis=-1)) *1
			field.arr[pd] += f

	path = r'../../output/mv_gauss_11_dual_3'

	import imageio.v3 as iio

	frames = []
	for i in range(1000*2):
		arr = (field.arr[0, 0:-1])  # bivector part
		frames.append(np.copy(arr.T))
		# FIXME: m=3 seems to be the max?
		for _ in range(3):
			field.step_mass(m=0.1 + V*1, c=0.05)

	os.makedirs(path, exist_ok=True)
	frames = np.array(frames)[::-1]
	print(frames.shape)
	frames = np.abs(frames)
	# frames = frames / np.percentile(frames.flatten(), 95)
	frames = frames * 2
	frames = (np.sqrt(np.clip(frames, 0, 1)) *255).astype(np.uint8)
	# frames = np.clip(frames * 255, 0, 255).astype(np.uint8)
	iio.imwrite(os.path.join(path, 'anim.gif'), frames)


def test_multivector_3_dual():
	field = DualMultiVectorField3((128, 128))
	x = field.meshgrid()
	gauss = np.exp(-(x ** 2 * 16 ** 2).sum(0))
	V = (x ** 2).sum(0) * 3
	for s in [-20, 20]:
		q = np.random.normal(0, 1, (2, 8, 1, 1))
		q = q / np.linalg.norm(q) * gauss
		f = np.roll(q, s, axis=-1) #- np.roll(q, -64, axis=-1)) *1
		field.arr += f

	path = r'../../output/mv_gauss_21_dual_3'

	import imageio.v3 as iio

	frames = []
	for i in range(500):
		arr = (field.arr[0, 5:8])  # bivector part
		frames.append(np.copy(arr.T))
		for _ in range(2):
			field.step_mass(m=-1.0+V, c=0.02)

	os.makedirs(path, exist_ok=True)
	frames = np.array(frames)
	frames = np.abs(frames) * 10
	# frames = frames / np.percentile(frames.flatten(), 99)
	frames = (np.sqrt(np.clip(frames, 0, 1)) *255).astype(np.uint8)
	# frames = np.clip(frames * 255, 0, 255).astype(np.uint8)
	iio.imwrite(os.path.join(path, 'anim.gif'), frames)


def kink_field(field, k):
	# FIXME: construct rotor over field
	# from numga.backend.numpy.context import NumpyContext
	# context = NumpyContext('t-x+')
	k = np.array(k)
	x = k.dot(field.meshgrid())
	field.arr[0, 0] *= np.cos(x)
	field.arr[-1, 0] *= np.sin(x)

def kink_field_3d(field, k):
	# FIXME: construct rotor over field
	#  dot x coord with k; givs scalar, dual into xyzt, and exp to get rotor?
	# from numga.backend.numpy.context import NumpyContext
	# context = NumpyContext('t-x+')
	k = np.array(k)
	x = k.dot(field.meshgrid())
	field.arr[0, 0] *= np.cos(x)
	field.arr[-1, 0] *= np.sin(x)


def test_multimultivector_2():
	# FIXME: try and debug cubic terms

	field = MultiMultiVectorField2((1024,), n=2)
	field.M[0, 0] = -.3
	# field.M[1, 0] = +.1
	# field.M[0, 1] = +.1
	# field.M[1, 1] = -.1*0
	# field.Mx[0, 0] = -.3

	# field.Q[0, 0, 1] = -0.1
	# field.Q[1, 0, 0] = -0.1
	# field.Q[0, 0, 0] = -.15
	# field.Q[1, 1, 1] = .5

	x = field.meshgrid()
	gauss = lambda sigma: np.exp(-(x ** 2 * sigma ** 2).sum(0))
	V = (x ** 2).sum(axis=0) * 1
	# V = (1 - gauss(1)) / 3
	V = np.einsum('x, i,j->ijx', V, [1, 0], [1, 0])
	# field.M = np.einsum('ij,x, k->kx', field.M, V, [1, 0])
	# field.M = field.M[..., None] + V
	print(V.shape, V.max())
	for s in [-150, 53]:
		for pd in [0, 1]:
			q = np.random.normal(0, 1, (4, 2, 1))
			q = q / np.linalg.norm(q) * gauss(32)
			f = np.roll(q, s+pd*10, axis=-1) #- np.roll(q, -64, axis=-1)) *1
			field.arr[:, pd] += f[:, 0]

	kink_field(field, [250])
	# field.arr[1:3, :] = 0   # knock out odd grade
	path = r'../../output/mmv_gauss_11_kink_mx'

	import imageio.v3 as iio

	frames = []
	for i in range(1024):
		arr = (field.arr[[0, 1, 3], 0])  # bivector part of massive field
		frames.append(np.copy(arr.T))
		# FIXME: m=3 seems to be the max?
		for _ in range(3):
			field.step_mass_q()

	os.makedirs(path, exist_ok=True)
	frames = np.array(frames)[::-1]
	print(frames.shape)
	frames = np.abs(frames)
	# frames = frames / np.percentile(frames.flatten(), 95)
	frames = frames * 2
	frames = (np.sqrt(np.clip(frames, 0, 1)) *255).astype(np.uint8)
	# frames = np.clip(frames * 255, 0, 255).astype(np.uint8)
	iio.imwrite(os.path.join(path, 'anim__.gif'), frames)


def test_multimultivector_4_compactified():
	"""once compactified, to 2d"""
	compact = 1
	n_fields = 1
	field = MultiMultiVectorField4((compact, 128, 128), n_fields)
	field.M[0, 0] = +.4
	# field.Mx[0, 0] = +.4
	# field.M[1, 0] = +.5
	# field.M[0, 1] = +.5
	# field.M[1, 1] = -.1


	x = field.meshgrid()


	# [1:] to skip the first compact coord; would collapse our gauss
	gauss = np.exp(-(x[1:] ** 2 * 8 ** 2).sum(0, keepdims=True))
	# with all components in V summed
	# V = (x[:] ** 2).sum(0) *1.5
	# dropping the harsh dV over the compact dim
	V = (x[1:] ** 2).sum(0) / 2
	# print(V.shape)
	# return
	# gauss = np.exp(-(x ** 2 * 4 ** 2).sum(0))
	# field.arr = np.random.normal(0 ,1, (16, 1, 1, 1)) * gauss

	for s in [20]:#, -20, 20, 150]:
		for n in [0]:
			q = np.random.normal(0, 1, (16, n_fields, 1, 1, 1))
			q = q / np.linalg.norm(q) * gauss
			# print(q.shape)
			f = np.roll(q, s, axis=-1)
			field.arr += f
	# mixing = np.ones((compact,compact)) + np.eye(compact)
	# field.arr = np.einsum('cnxyz,xo->cnoyz', field.arr, mixing)
	# funnly if zero the readout field we get grayscale output
	#  cant say i understand
	# field.arr[:, 1, 0] = 0

	path = r'../../output/mmv_31_1_1_recreate'

	import imageio.v3 as iio

	frames = []
	for i in range(500):
		# arr = (field.arr[5:8, 0, compact//2])   # bivector, massive, x-slice
		# arr = (field.arr[[5, 8, 9], 0, compact//2])   # xyt bivectors, massive, x-slice
		arr = (field.arr[[7, 9, 10], 0, compact//2])   # yzt bivectors, massive, x-slice
		frames.append(np.copy(arr.T))
		print(np.linalg.norm(arr))
		for _ in range(2):
			field.step(V)

	os.makedirs(path, exist_ok=True)
	frames = np.array(frames)
	frames = np.abs(frames)
	# frames = frames / 0.15  # arr.max()
	frames = frames / np.percentile(frames.flatten(), 95) / 4
	frames = ((np.clip(frames, 0, 1) ** 1) *255).astype(np.uint8)
	iio.imwrite(os.path.join(path, 'anim_s12.gif'), frames)


def test_multimultivector_4_1d():
	"""doubly-compactified 3+1 -> 1+1"""
	n_fields = 1
	compact = 1
	compact_y = 1
	field = MultiMultiVectorField4((compact, compact_y, 256), n_fields)
	field.Mx[0, 0] = -.5
	# field.M[1, 0] = +.9
	# field.M[0, 1] = +.9
	# field.M[1, 1] = -.1


	x = field.meshgrid()


	# [1:] to skip the compact coords; would collapse our gauss
	gauss = np.exp(-(x[2:] ** 2 * 32 ** 2).sum(0, keepdims=True))
	# with all components in V summed
	# V = (x[:] ** 2).sum(0) *1.5
	# dropping the harsh dV over the compact dims
	V = (x[2:] ** 2).sum(0) *1.5 * 0
	# print(V.shape)
	# return
	# gauss = np.exp(-(x ** 2 * 4 ** 2).sum(0))
	# field.arr = np.random.normal(0 ,1, (16, 1, 1, 1)) * gauss

	for s in [10]:#, -20, 20, 150]:
		for n in [0]:
			q = np.random.normal(0, 1, (16, n_fields, compact, compact_y, 1))
			q = q / np.linalg.norm(q) * gauss
			# print(q.shape)
			f = np.roll(q, s, axis=-1)
			field.arr += f
	# mixing = np.ones((compact,compact)) + np.eye(compact)
	# field.arr = np.einsum('cnxyz,xo->cnoyz', field.arr, mixing)
	# funnly if zero the readout field we get grayscale output
	#  cant say i understand
	# field.arr[:, 1, 0] = 0

	path = r'../../output/mmv_31_1_1_1_mx'

	import imageio.v3 as iio

	frames = []
	for i in range(500):
		# arr = (field.arr[5:8, 0, 0, 0])   # bivector, massive, x-slice
		# arr = (field.arr[8:11, 0, 0, 0])   # bivector, massive, x-slice
		arr = (field.arr[[0, 8, 15], 0, 0, 0])   # bivector, massive, x-slice
		frames.append(np.copy(arr.T))
		print(np.linalg.norm(arr))
		for _ in range(3):
			field.step(V)

	os.makedirs(path, exist_ok=True)
	frames = np.array(frames)
	frames = np.array(frames)[::-1]
	frames = np.abs(frames)
	# frames = frames / 0.15  # arr.max()
	frames = frames / np.percentile(frames.flatten(), 95) / 4
	frames = (np.sqrt(np.clip(frames, 0, 1)) *255).astype(np.uint8)
	iio.imwrite(os.path.join(path, 'anim.gif'), frames)
