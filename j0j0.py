#!/usr/bin/env python3
r"""
Test on the following analytic case with 2 j_1's
​
\int 1/(1+x^2) j_1(ax)j_1(bx) x^2dx = pi/(2a^2b^2)
    / (1+b) e^-b (a cosha - sinha),  a<=b
    \ (1+a) e^-a (b coshb - sinhb),  a>b
​
This is one of the simplest cases where not all the terms of the sin-cos
expansion converges.
​
The output of each sine or cosine integrals evaluated with FFTLog algorithm is
logarithmically spaced, from which interpolation is needed at (a+b) or (a-b).
​
Compared to cubic spline (C2), (cubic) Hermite interpolation is only C1 but
local (so fast).
Because the user need to provide derivative, therefore supposedly it should be
more accurate than spline when interpolating for derivatives.
​
In the case of the multiple-j algorithm here, for example two j's, when one of
the two output arguments is much bigger than the other, the product-to-sum
identities would sometimes produce cancellations that needs accurate
interpolation for the first derivatives.
And for more j's, even higher derivatives are needed for accuracy when one of
the output argument is much bigger than all others.
​
This test compares CubicSpline to BPoly.from_derivatives for this purpose.
"""

import numpy as np
from mcfit.transforms import FourierSine, FourierCosine
from scipy.interpolate import CubicSpline, BPoly
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

#lgxmin, lgxmax = -2, 4
#Nx_perdex = 50
#Nx = int(Nx_perdex * (lgxmax - lgxmin))
#x = np.logspace(lgxmin, lgxmax, num=Nx, endpoint=False)
#F = 1 / (1 + x*x)

def rotation_method_bessel_j0j0(x,F,a,b):
	# Rotation method for spherical bessel integrals j0j0
	# following https://arxiv.org/pdf/1912.00065.pdf
	# I took the script at
	# https://github.com/eelregit/sbf_rotation/blob/master/j1j1.py
	# and modified trigsum_cspline, trigsum_hermite, and trigsum_hermite5
	# for the j0j0 case rather than the j1j1 case (eqns. 33 and 34 in the paper)
	# given some function k P(k), r, and rprime, this will give
	# \int F(x) j_0(ax)j_0(bx) x^2dx
	# using the quintic Hermite interpolation
	Fc0 = np.sqrt(np.pi / 2) * F * x**2
	Fsm1 = np.sqrt(np.pi / 2) * F * x
	Fcm2 = np.sqrt(np.pi / 2) * F
	Fsm3 = np.sqrt(np.pi / 2) * F / x
	Fcm4 = np.sqrt(np.pi / 2) * F / x**2

	extrap = True
	N = 8096

	def symmetrize(y, G, dGdy=None, d2Gdy2=None, parity=0):
		"""Symmetrize G(y) and G'(y) before interpolation (particularly for cubic
		spline because Hermite interp does not need the full negative half but just
		one segment) to cover [0, ymin) and to respect the symmetry.
		"""
		y = np.concatenate((- y[::-1], y))
		G = np.concatenate(((-1)**parity * G[::-1], G))
		if dGdy is not None:
			dGdy = np.concatenate((-(-1)**parity * dGdy[::-1], dGdy))
			G = np.column_stack((G, dGdy))
			if d2Gdy2 is not None:
				d2Gdy2 = np.concatenate(((-1)**parity * d2Gdy2[::-1], d2Gdy2))
				G = np.column_stack((G, d2Gdy2))
		return y, G

	qc0 = 2.5
	#print('Fourier Cosine transform of x^2 / (1+x^2), with tilt q =', qc0)
	Tc0 = FourierCosine(x, q=qc0, N=N, lowring=False)
	Tc0.check(Fc0)
	yc0, C0 = Tc0(Fc0, extrap=extrap)
	y = yc0

	qsm1 = qc0 - 1
	#print('Fourier Sine transform of x^2 / [(1+x^2) x^1], with tilt q =', qsm1)
	Tsm1 = FourierSine(x, q=qsm1, N=N, lowring=False)
	Tsm1.check(Fsm1)
	ysm1, Sm1 = Tsm1(Fsm1, extrap=extrap)
	assert all(y == ysm1)

	qcm2 = qsm1 - 1
	#print('Fourier Cosine transform of x^2 / [(1+x^2) x^2], with tilt q =', qcm2)
	Tcm2 = FourierCosine(x, q=qcm2, N=N, lowring=False)
	Tcm2.check(Fcm2)
	ycm2, Cm2 = Tcm2(Fcm2, extrap=extrap)
	assert all(y == ycm2)
	Cm2_cspline = CubicSpline(* symmetrize(ycm2, Cm2, parity=0))
	Cm2_hermite = BPoly.from_derivatives(* symmetrize(ycm2, Cm2, dGdy=-Sm1, parity=0))
	Cm2_hermite5 = BPoly.from_derivatives(* symmetrize(ycm2, Cm2, dGdy=-Sm1, d2Gdy2=-C0, parity=0))

	qsm3 = qcm2 - 1
	#print('Fourier Sine transform of x^2 / [(1+x^2) x^3], with tilt q =', qsm3)
	Tsm3 = FourierSine(x, q=qsm3, N=N, lowring=False)
	Tsm3.check(Fsm3)
	ysm3, Sm3 = Tsm3(Fsm3, extrap=extrap)
	assert all(y == ysm3)
	Sm3_cspline = CubicSpline(* symmetrize(ysm3, Sm3, parity=1))
	Sm3_hermite = BPoly.from_derivatives(* symmetrize(ysm3, Sm3, dGdy=Cm2, parity=1))
	Sm3_hermite5 = BPoly.from_derivatives(* symmetrize(ysm3, Sm3, dGdy=Cm2, d2Gdy2=-Sm1, parity=1))

	qcm4 = qsm3 - 1
	#print('Fourier Cosine transform of x^2 / [(1+x^2) x^4], with tilt q =', qcm4)
	Tcm4 = FourierCosine(x, q=qcm4, N=N, lowring=False)
	Tcm4.check(Fcm4)
	ycm4, Cm4 = Tcm4(Fcm4, extrap=extrap)
	assert all(y == ycm4)
	Cm4_cspline = CubicSpline(* symmetrize(ycm4, Cm4, parity=0))
	Cm4_hermite = BPoly.from_derivatives(* symmetrize(ycm4, Cm4, dGdy=-Sm3, parity=0))
	Cm4_hermite5 = BPoly.from_derivatives(* symmetrize(ycm4, Cm4, dGdy=-Sm3, d2Gdy2=-Cm2, parity=0))

	def trigsum_cspline(a, b):
		return ((Cm2_cspline(a-b) + Cm2_cspline(a+b))) / (a * b)

	def trigsum_hermite(a, b):
		return ((Cm2_hermite(a-b) + Cm2_hermite(a+b))) / (a * b)


	def trigsum_hermite5(a, b):
		return ((Cm2_hermite5(a-b) + Cm2_hermite5(a+b))) / (a * b)


	num_hermite5 = trigsum_hermite5(a, b)
	return num_hermite5