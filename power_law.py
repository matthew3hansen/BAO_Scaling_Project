import numpy as np
import sympy as sp
import scipy as sci
import scipy.optimize
import time
import scipy.integrate as integrate
import scipy.special as special
import math
import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import CAMB_General_Code 
import helper2

def run(alpha=1.):
	radial_bin = np.linspace(30,180,100)
	data_list = np.zeros(len(radial_bin))

	for i, x in enumerate(radial_bin):
		data_list[i] = x**-2.


	def fixed_bias():
		this_delta_alpha = np.linspace(-0.1, 0.1, 100)
		list_exp = np.zeros(len(this_delta_alpha))

		for i, delta_alpha in enumerate(this_delta_alpha):
			chi2 = np.sum((data_list - (radial_bin * (1 + delta_alpha))**-2.)**2.)
		
			list_exp[i] = np.exp(-0.5 * chi2)

		plt.plot(this_delta_alpha, list_exp)
		plt.title('Fixed bias, np.exp(-0.5 * chi2)')
		plt.show()


	def fixed_alpha():
		delta_alpha = 0 
		this_bias = np.linspace(-5, 5, 100)
		list_exp = np.zeros(len(this_bias))

		for i, bias in enumerate(this_bias):
			chi2 = np.sum((data_list - bias * (radial_bin * (1 + delta_alpha))**-2.)**2.)
	
			list_exp[i] = np.exp(-0.5 * chi2)

		plt.plot(this_bias, list_exp)
		plt.title('Fixed alpha, np.exp(-0.5 * chi2)')
		plt.show()


	def marginal_delta_alpha():
		this_b1 = np.linspace(-5, 5, 100)
		this_delta_alpha = np.linspace(-0.1, 0.1, 100)
		constant = 0.5
		list_exp = np.zeros((len(this_delta_alpha), len(this_delta_alpha)))
		solution_list = np.zeros(len(this_delta_alpha))

		covariance_matrix = np.diag((0.1 * radial_bin **-2)**2.)
		precision = np.linalg.inv(covariance_matrix)

		for i, delta_alpha in enumerate(this_delta_alpha):
			a = 0.0
			b = 0.0 
			c = 0.0

			for l in range(len(radial_bin)):
				for m in range(len(radial_bin)):
					a += constant * precision[l][m] * (radial_bin[l] * (1 + delta_alpha))**-2. * (radial_bin[m] * (1 + delta_alpha))**-2.
					b += - constant * precision[l][m] * (- data_list[l] * (radial_bin[m] * (1 + delta_alpha))**-2. \
										- data_list[m] * (radial_bin[l] * (1 + delta_alpha))**-2 )
					c += - constant * precision[l][m] * data_list[l] * data_list[m]

			analytic_solution = np.sqrt(np.pi / a) * np.exp(b**2 / (4 * a) + c)
			solution_list[i] = analytic_solution
			list_exp[:, i] = np.exp(-a * this_b1 **2 + b * this_b1 + c)


		#np.savetxt('marginal_delta_alpha.txt', list_exp)

		print(this_delta_alpha[np.argmax(solution_list)])
		plt.plot(this_delta_alpha, solution_list)
		plt.title('Marginal, np.sqrt(np.pi / a) * np.exp(b**2 / (4 * a) + c)')
		plt.show()
		

	return marginal_delta_alpha()