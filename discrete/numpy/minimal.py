"""minimal self contained numpy example"""
import numpy as np

def ed(a, d):
	"""first order exterior derivative with periodic bcs"""
	return np.diff(a, axis=d, append=np.take(a, [0], axis=d))
def id(a, d):
	"""first order interior derivative with periodic bcs"""
	return np.diff(a, axis=d, prepend=np.take(a, [-1], axis=d))

speed = 0.5
def edt(lhs, rhs):
	lhs -= rhs * speed
def edx(a):
	return ed(a, 0)
def edy(a):
	return ed(a, 1)
def edz(a):
	return ed(a, 2)

def idt(lhs, rhs):
	lhs -= rhs * speed
def idx(a):
	return id(a, 0)
def idy(a):
	return id(a, 1)
def idz(a):
	return id(a, 2)


class Field:
	def __init__(self, shape, components):
		self.shape = shape
		self.ndim = len(shape)
		self.dtype = np.float32
		self.arr = np.zeros((components,) + shape, dtype=self.dtype)

	def meshgrid(self):
		xs = [np.linspace(-1, 1, s) for s in self.shape]
		return np.array(np.meshgrid(*xs, indexing='ij'))


class MultiVectorField4(Field):
	def __init__(self, shape):
		assert len(shape) == 3
		super(MultiVectorField4, self).__init__(shape, 16)

	def step_mass(self, m=1):
		s, x, y, z, t, xy, xz, yz, xt, yt, zt, xyz, xyt, xzt, yzt, xyzt = self.arr
		edt(s,    m*t    +idx(xt)+idy(yt)+idz(zt))
		edt(x,    m*xt   +edx(t)-idy(xyt)-idz(xzt))
		edt(y,    m*yt   +idx(xyt)+edy(t)-idz(yzt))
		edt(z,    m*zt   +idx(xzt)+idy(yzt)+edz(t))
		edt(xy,   m*xyt  +edx(yt)-edy(xt)+idz(xyzt))
		edt(xz,   m*xzt  +edx(zt)-idy(xyzt)-edz(xt))
		edt(yz,   m*yzt  +idx(xyzt)+edy(zt)-edz(yt))
		edt(xyz,  m*xyzt +edx(yzt)-edy(xzt)+edz(xyt))

		idt(t,    -m*s    +idx(x)+idy(y)+idz(z))
		idt(xt,   -m*x    +edx(s)-idy(xy)-idz(xz))
		idt(yt,   -m*y    +idx(xy)+edy(s)-idz(yz))
		idt(zt,   -m*z    +idx(xz)+idy(yz)+edz(s))
		idt(xyt,  -m*xy   +edx(y)-edy(x)+idz(xyz))
		idt(xzt,  -m*xz   +edx(z)-idy(xyz)-edz(x))
		idt(yzt,  -m*yz   +idx(xyz)+edy(z)-edz(y))
		idt(xyzt, -m*xyz  +edx(yz)-edy(xz)+edz(xy))
