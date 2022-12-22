
from numga.algebra.algebra import Algebra
from discrete.numpy.field_slice import FieldSlice


def test_xt_full_mass():
	print()
	algebra = Algebra.from_str('x+t-')
	shape, steps = (256,), 256
	field = FieldSlice.from_subspace(algebra.subspace.multivector(), shape)

	field = field.random_gaussian([0.1])
	mass = field.quadratic() / 4# + 0.1
	# mass = 0.1

	# dimple = -(1-field.gauss([0.6])*0.5) / 4
	# metric = {'t': dimple / 4}
	# metric = {'t': 0.33}

	field.write_gif_1d_generator(
		field.rollout_generator(steps, mass=mass),
		basepath='../../output', components='x_t_xt', pre='numpy',
	)


def test_wxt_even_mw():
	print()
	algebra = Algebra.from_str('w+x+t-')
	shape, steps = (2, 256,), 512
	field = FieldSlice.from_subspace(algebra.subspace.even_grade(), shape)
	print(field.subspace)

	field = field.random_gaussian([0.3])
	# mass = 0.4 + field.quadratic() / 2# + 0.1

	dimple = (1-field.gauss([0.6])*0.5)
	metric = {'w': dimple / 4}

	field.write_gif_2d_generator_compact(
		field.rollout_generator(steps, metric=metric),
		basepath='../../output', components='wx_wt_xt', pre='numpy',
	)


def test_wxt_full_mass():
	print()
	algebra = Algebra.from_str('w+x+t-')
	shape, steps = (2, 256,), 512
	field = FieldSlice.from_subspace(algebra.subspace.full(), shape)
	print(field.subspace)

	field = field.random_gaussian([0.3])
	# mass = 0.4 + field.quadratic() / 2# + 0.1

	# dimple = (1-field.gauss([0.6])*0.5)
	# metric = {'w': dimple / 4}
	mass = 0.2

	field.write_gif_2d_generator_compact(
		field.rollout_generator(steps, mass=mass),
		basepath='../../output', components='wx_wt_xt', pre='mass',
	)


def test_xyt_full_mass():
	print()
	algebra = Algebra.from_str('x+y+t-')
	shape = (128, 128)
	steps = 256
	field = FieldSlice.from_subspace(algebra.subspace.multivector(), shape)
	field = field.random_gaussian(0.1)
	mass = 0.4 + field.quadratic() / 2# + 0.1

	# metric = {'t': (1-field.gauss(0.3)*0.5)}

	field.write_gif_2d_generator(
		field.rollout_generator(steps, mass=mass),
		basepath='../../output', components='xt_yt_xy', post='mass',
	)


def test_wxyt_even_mw():
	print()
	algebra = Algebra.from_str('w+x+y+t-')
	shape, steps = (2, 256, 256), 512
	field = FieldSlice.from_subspace(algebra.subspace.even_grade(), shape)

	field = field.random_gaussian([0.3], [0, 0, 0.1])
	dimple = (1-field.gauss([1e16, 0.6, 0.6])*0.5)
	metric = {'w': dimple / 4}

	bivecs = ['wt_xt_yt', 'wx_wy_wt']
	for bv in bivecs:
		field.write_gif_3d_generator(
			field.rollout_generator(steps, metric=metric),
			basepath='../../output', components=bv, pre='numpy',
		)


def test_wxyt_even_conservation():
	"""test amplitude conservation"""
	print()
	algebra = Algebra.from_str('w+x+y+t-')
	shape, steps = (16, 16, 16), 32
	field = FieldSlice.from_subspace(algebra.subspace.even_grade(), shape)

	field = field.random_gaussian([0.3, 0.3, 0.3], [0, 0, 0.1])
	dimple = (1-field.gauss([0.6, 0.6, 0.6])*0.5)
	metric = {'w': dimple / 4}

	ff = field.rollout(steps, metric=metric)
	print(ff.arr.sum((0, 1, 2, 3)))


def test_xyzt_full_mass():
	print()
	algebra = Algebra.from_str('x+y+z+t-')
	shape = (64, 64, 64)
	steps = 128
	field = FieldSlice.from_subspace(algebra.subspace.multivector(), shape)
	field = field.random_gaussian(0.1)
	mass = 0.4 + field.quadratic() / 2# + 0.1

	# metric = {'t': (1-field.gauss(0.3)*0.5)}

	field.write_gif_3d_generator(
		field.rollout_generator(steps, mass=mass),
		basepath='../../output', components='xt_yt_zt', post='mass',
	)
