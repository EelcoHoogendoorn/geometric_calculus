
import numpy as np

from numga.algebra.algebra import Algebra
from discrete.numpy.field_slice import FieldSlice


def test_wxyt_even_mw():
	print()
	algebra = Algebra.from_str('w+x+y+t-')
	shape, steps = (2, 256, 256), 512
	field = FieldSlice.from_subspace(algebra.subspace.even_grade(), shape)

	field = field.random_gaussian([0.3], [0, 0, 0.1])
	dimple = (1-field.gauss(np.array([1e16, 0.6, 0.6]))*0.5)
	metric = {'w': dimple / 4}

	bivecs = ['wt_xt_yt', 'wx_wy_wt']
	for bv in bivecs:
		field.write_gif_3d_generator(
			field.rollout_generator(steps, metric=metric),
			basepath='../../output', components=bv, pre='numpy',
		)
