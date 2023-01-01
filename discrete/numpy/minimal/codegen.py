"""Code generation tools for leapfrog operator in numpy syntax"""

import numpy as np

from numga.algebra.algebra import Algebra
from discrete.numpy.field_slice import FieldSlice


def operator_to_str(field, op, aux=None) -> str:
	"""Text representation of a leapfrog geometric derivative operator in numpy syntax"""
	term_to_str = field.term_to_str()
	equation = op.subspace.named_str.replace('1', 's').split(',')

	def line(eq):
		eq_idx, (tt, ts) = eq
		s, *r, _ = term_to_str(tt)
		r = ''.join(r)
		rhs = ''.join([term_to_str(t) for t in ts])
		e = equation[eq_idx]    # element this equation pertains to
		if aux is not None:
			rhs = f'{rhs} {aux[e]}'
		return f'{r}, {s}({rhs}))' + f'\t # {e}'

	T, S = field.process_op_leapfrog(op)
	return '\n'.join(line(l) for l in T + S)



def geometric_to_str(field, aux=None) -> str:
	return operator_to_str(
		field,
		# FIXME: would be nice to have support for both left/right derivatives
		# field.algebra.operator.geometric_product(field.subspace, field.domain),
		field.algebra.operator.geometric_product(field.domain, field.subspace),
		aux
	)


def output_space(field):
	"""construct the output space of a geometric derivative over a field"""
	return field.algebra.operator.geometric_product(field.domain, field.subspace).subspace


def pseudoscalar(field):
	return field.algebra.subspace.pseudoscalar()

def sps(field):
	return pseudoscalar(field).inner(field.algebra.subspace.t)



def direct_mass_term(field, symbol='m'):
	"""string representation of direct mass term"""
	input = output_space(field).named_str.replace('1', 's').split(',')
	return {i: f'+{symbol}*{i}' for i in (input)}

# def t_mass_term(field, symbol='m'):
# 	"""string representation of t-mass term"""
# 	op = field.algebra.operator.product(field.subspace, field.algebra.subspace.t)
# 	input = field.subspace.named_str.replace('1', 's').split(',')
# 	output = op.subspace.named_str.replace('1', 's').split(',')
# 	sign = {-1: '-', +1: '+'}
# 	kernel = op.kernel[:, 0]
# 	# return [(input[i], output[o], sign[kernel[i,o]]) for i,o in zip(*np.nonzero(kernel))]
# 	return {output[o]: f'{sign[kernel[i,o]]}{symbol}*{input[i]}' for i,o in zip(*np.nonzero(kernel))}


def sps_mass_term(field, symbol='m', side='right'):
	"""string representation of sps mass term"""
	sps_str = sps(field).named_str

	def interpolate(q):
		axes=['+1' if c in q else '-1' for c in sps_str]
		axes = ','.join(axes)
		return f'interpolate({q}, {axes})'

	# toggle left/right dualization
	output = output_space(field)
	if side == 'right':
		op = field.algebra.operator.product(output, sps(field))
		kernel = op.kernel[:, 0]
		output, _, input = op.axes
	else:
		op = field.algebra.operator.product(sps(field), output)
		kernel = op.kernel[0]
		_, output, input = op.axes

	# print(op)
	input = input.named_str.replace('1', 's').split(',')
	output = output.named_str.replace('1', 's').split(',')
	sign = {-1: '-', +1: '+'}
	# return {output[o]: f'{sign[kernel[i,o]]}{symbol}*{interpolate(input[i])}' for i,o in zip(*np.nonzero(kernel))}
	return {output[i]: f'{sign[kernel[i,o]]}{symbol}*{interpolate(input[o])}' for i,o in zip(*np.nonzero(kernel))}


def xyz_mass_term(field, symbol='m', side='right'):
	"""string representation of sps mass term"""
	dualizer = field.algebra.subspace.xyz
	xyz_str = dualizer.named_str
	sps_str = sps(field).named_str

	def interpolate(q):
		axes=['0' if not c in xyz_str else '+1' if c in q else '-1' for c in sps_str]
		axes = ','.join(axes)
		return f'interpolate({q}, {axes})'

	# toggle left/right dualization
	output = output_space(field)
	if side == 'right':
		op = field.algebra.operator.product(output, dualizer)
		kernel = op.kernel[:, 0]
		output, _, input = op.axes
	else:
		op = field.algebra.operator.product(dualizer, output)
		kernel = op.kernel[0]
		_, output, input = op.axes

	# print(op)
	input = input.named_str.replace('1', 's').split(',')
	output = output.named_str.replace('1', 's').split(',')
	sign = {-1: '-', +1: '+'}
	# return {output[o]: f'{sign[kernel[i,o]]}{symbol}*{interpolate(input[i])}' for i,o in zip(*np.nonzero(kernel))}
	return {output[i]: f'{sign[kernel[i,o]]}{symbol}*{interpolate(input[o])}' for i,o in zip(*np.nonzero(kernel))}


# def dual_mass_term(field, symbol='m'):
# 	"""string representation of dual mass term"""
# 	op = field.algebra.operator.product(field.subspace, pseudoscalar(field))
# 	input = field.subspace.named_str.replace('1', 's').split(',')
# 	output = op.subspace.named_str.replace('1', 's').split(',')
# 	sign = {-1: '-', +1: '+'}
# 	kernel = op.kernel[:, 0]
# 	return {output[o]: f'{sign[kernel[i,o]]}{symbol}*{input[i]}' for i,o in zip(*np.nonzero(kernel))}


def test_xyz_term():
	print()
	algebra = Algebra.from_str('w+x+y+z+t-')
	# algebra = Algebra.from_str('x-y-t+')
	shape = (2, 32, 32, 32)
	field = FieldSlice.from_subspace(algebra.subspace.even_grade(), shape)

	# op = xyz_mass_term(field)
	# print(op)
	print(field.subspace)
	print(geometric_to_str(field, xyz_mass_term(field)))




def test_xyt_full_sps():
	print()
	algebra = Algebra.from_str('w+x+y+t-')
	# algebra = Algebra.from_str('x-y-t+')
	shape = (2, 32, 32)
	field = FieldSlice.from_subspace(algebra.subspace.even_grade(), shape)
	print(field.subspace)
	print(geometric_to_str(field, sps_mass_term(field)))

def test_xt_full_sps():
	print()
	algebra = Algebra.from_str('x+t-')
	shape = (32, )
	field = FieldSlice.from_subspace(algebra.subspace.full(), shape)
	print(field.subspace)
	print(geometric_to_str(field, sps_mass_term(field)))


def test_xt_even_direct():
	print()
	algebra = Algebra.from_str('x+t-')
	shape = (32, )
	field = FieldSlice.from_subspace(algebra.subspace.even_grade(), shape)
	print(field.subspace)
	print(geometric_to_str(field, direct_mass_term(field)))


def test_wxyzt_full_sps_direct():
	print()
	algebra = Algebra.from_str('w+x+y+z+t-')
	# algebra = Algebra.from_str('x-y-t+')
	shape = (2, 32, 32, 32)
	field = FieldSlice.from_subspace(algebra.subspace.full(), shape)

	sps = sps_mass_term(field, 'm')
	direct = direct_mass_term(field, 'M')
	mass = {k: f'{sps[k]} {direct[k]}' for k in sps}
	print(field.subspace)
	print(geometric_to_str(field, mass))


def test_xyt_full_sps_direct():
	print()
	algebra = Algebra.from_str('x-y-t+')
	# algebra = Algebra.from_str('x-y-t+')
	shape = (32, 32)
	field = FieldSlice.from_subspace(algebra.subspace.full(), shape)

	sps = sps_mass_term(field, 'm')
	direct = direct_mass_term(field, 'M')
	mass = {k: f'{sps[k]} {direct[k]}' for k in sps}
	print(field.subspace)
	print(geometric_to_str(field, mass))


def test_xyzt_bivector():
	print()
	algebra = Algebra.from_str('x+y+z+t-')
	# algebra = Algebra.from_str('x-y-t+')
	shape = (32, 32, 32)
	field = FieldSlice.from_subspace(algebra.subspace.even_grade(), shape)

	print(field.subspace)
	print(geometric_to_str(field, xyz_mass_term(field)))
