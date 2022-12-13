import numpy as np


def offsets(n):
	o = [0, 1, 2]
	return np.array(np.meshgrid(*[o]*n, indexing='ij'))


def deltas(i, o):
	"""spatial distance between dofs of same integer coordinate"""
	bi = i.bit_blades().astype(np.int)
	bo = o.bit_blades().astype(np.int)
	return bi[:, None, :] - bo[:, :, None]   # distance vector of bitpatterns
