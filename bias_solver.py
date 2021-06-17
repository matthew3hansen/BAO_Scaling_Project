'''
Coding up our solution to the BAO Fast Scaling problem. 
December 2020
Author(s): Matt Hansen, Alex Krolewski
'''
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
	helper_object = helper2.Info(alpha)
	helper_object.calc_CF()

	#I don't know how the data vector will look in the end, but right now I'm thinking that it can be a list where each index represents a 
	#radial bin, and its corresponding value is the data at that radial bin
	data_list = helper_object.get_data()

	covariance_matrix = helper_object.get_covariance_matrix()

	xi_IRrs = helper_object.templates()

	xi_IRrs_prime = helper_object.templates_deriv()

	xi_IRrs_prime2 = helper_object.templates_deriv2()

	#helper_object.graphing_avg()
	#helper_object.dlog_dalpha()
	#helper_object.templates_times_alpha()
	#helper_object.templates_times_alpha_loop()
	p1 = helper_object.get_biases()

	parameters = [p1]
	radial_bin_l_ = helper_object.get_r()
	radial_bin_m_ = helper_object.get_r()
	#These two variables can represent what two radials bins we will select to due calculation off of. This should be easily converted into a 
	#scalable model where we can take an arbitary number of bins and compute them with regards to two of them. I can even make this a OOP problem
	#radial_bin_l = radial_bin_l_
	#radial_bin_m = radial_bin_m_

	'''
	It might be easier to not do it the way we originally planned with xi1 - xi4 lists. I think having one "tensor" like 
	how we had it in our paper might be better to program. I will try a simple case to see
	'''
	#radial bin at each index, 3x3 matrix at each radial bin
	''' This would be at each index
	[p1, p2, p3, p4] [1 2 3]
	 				 [4 5 6]
	 				 [7 8 9]
	 Takes the form of [xi_i][xi_i'][xi_i''] these represent columns in 'vector space'
	 				 [10 11 12]
	 I am just giving random values for now, so that the code is able to run
	'''
	xi_tensor = [[] for _ in range(len(xi_IRrs))]
	for i in range(len(xi_IRrs)):
		xi_tensor[i].append([xi_IRrs[i], xi_IRrs_prime[i], xi_IRrs_prime2[i]])

	#I'm giving the precision matrix a value of 1 right now so the code will compile. I do not know how it will look when we have the actual values
	#We should talk about how to represent this, I imagine it would be some sort of list or 2d matrix. Either way this will be easy
	#to change the code around since the variable is only used in the beginning of each definition and no where else
	
	#precision_matrix_lm = covariance_matrix[radial_bin_l][radial_bin_m]

	def product_(p1):
		p = np.array(p1)
		p = p.transpose()
		print('p: ', p)
		print('xi: ', xi_tensor[0])
		product = np.matmul(p, np.array(xi_tensor))
		print(product)
		return product

	#Model as a list with each index being a radial bin
	def model_l_(p1, radial_bin_l):
		productl = product_l_(p1, radial_bin_l)
		return productl[0] + productl[1] + productl[2]

	def model_m_(p1, radial_bin_m):
		productm = product_m_(p1, radial_bin_m)
		return productm[0] + productm[1] + productm[2]


	def quad_coeff_alpha_(p1): #this is a
		temp = 0.0
		product = product_(p1)
		precision_matrix = np.linalg.inv(helper_object.covariance_matrix)
		for l in range(len(radial_bin_l_)):
			for m in range(len(radial_bin_m_)):
				precision_matrix_lm = precision_matrix[l][m]
				temp += precision_matrix_lm * (product[l][1] * product[m][2] + product[l][2] * product[m][1])
		#print('quad: ', temp)
		return 2. * temp

	def linear_coeff_alpha_(p1): #This is b
		temp = 0.0
		product = product_(p1)
		precision_matrix = np.linalg.inv(helper_object.covariance_matrix)
		for l in range(len(radial_bin_l_)):
			for m in range(len(radial_bin_m_)):
				precision_matrix_lm = precision_matrix[l][m]
				temp += precision_matrix_lm * (product[l][0] * product[m][2] + product[l][2] * product[m][0] + 2 * product[l][1] * product[m][1] \
			   - data_list[l] * product[m][2] - data_list[m] * product[l][2])
		#print('lin: ', temp)
		return temp

	def const_coeff_alpha_(p1): #This is c
		temp = 0.0
		product = product_(p1)
		precision_matrix = np.linalg.inv(helper_object.covariance_matrix)
		for l in range(len(radial_bin_l_)):
			for m in range(len(radial_bin_m_)):
				precision_matrix_lm = precision_matrix[l][m]
				temp += precision_matrix_lm * (product[l][0] * product[m][1] + product[l][1] * product[m][0] - data_list[l] * product[m][1] - \
					data_list[m] * product[l][1])
		#print('con: ', temp)
		return temp

	def get_terms(p1):
		xi_IRrs_alpha = np.zeros(len(helper_object.r))

		precision = np.linalg.inv(helper_object.covariance_matrix)

		for i in range(len(helper_object.r)):
			xi_IRrs_alpha[i] = np.sum(1 / (2 * math.pi**2) * helper_object.ks**2 * special.spherical_jn(0, alpha * helper_object.ks * helper_object.r_bins[i]) * np.exp(-helper_object.ks**2) * helper_object.P_IRres * np.gradient(helper_object.ks))
		
		constant_term = 0
		linear_term = 0
		quadratic_term = 0
		third_term = 0
		for l in range(len(helper_object.r)):
			for m in range(len(helper_object.r)):
				constant_term += precision[l][m] * (- p1 * data_list[l] * p1 * helper_object.xi_IRrs_prime[m] - p1 * data_list[m] * p1 * helper_object.xi_IRrs_prime[l] \
						+ p1 * helper_object.xi_IRrs_prime[l] * p1 * helper_object.xi_IRrs[m] + p1 * helper_object.xi_IRrs_prime[m] * p1 * helper_object.xi_IRrs[l])

				linear_term += precision[l][m] * (- p1 * data_list[l] * p1 * helper_object.xi_IRrs_prime2[m] - p1 * data_list[m] * p1 * helper_object.xi_IRrs_prime2[l] \
								+ 2 * p1 * helper_object.xi_IRrs_prime[l] * p1 * helper_object.xi_IRrs_prime[m] + p1 * helper_object.xi_IRrs_prime2[l] * p1 * helper_object.xi_IRrs[m]\
								+ p1 * helper_object.xi_IRrs_prime2[m] * p1 * helper_object.xi_IRrs[l])

				quadratic_term += precision[l][m] * (0.5 * p1 * helper_object.xi_IRrs_prime[l] * p1 * helper_object.xi_IRrs_prime2[m] \
								+ p1 * helper_object.xi_IRrs_prime2[l] * p1 * helper_object.xi_IRrs_prime[m] + 0.5 * p1 * helper_object.xi_IRrs_prime[m] * p1 * helper_object.xi_IRrs_prime2[l] \
								+ p1 * helper_object.xi_IRrs_prime2[m] * p1 * helper_object.xi_IRrs_prime[l]) 

				third_term += precision[l][m] * (0.5 * helper_object.xi_IRrs_prime2[l] * helper_object.xi_IRrs_prime2[m] + 0.5 * helper_object.xi_IRrs_prime2[l] * helper_object.xi_IRrs_prime2[m])

		return (constant_term, linear_term, quadratic_term, third_term)


	#I'm not sure what delta_alpha we want to use
	def delta_alpha_():
		'''
		quad_coeff_alpha = quad_coeff_alpha_(p1)
		linear_coeff_alpha = linear_coeff_alpha_(p1)
		const_coeff_alpha = const_coeff_alpha_(p1)
		'''
		const_coeff_alpha, linear_coeff_alpha, quad_coeff_alpha, third_term = get_terms(p1)
		print('Correct:')
		print('const: ', const_coeff_alpha)
		print('lin: ', linear_coeff_alpha)
		print('quad: ', quad_coeff_alpha)
		delta = helper_object.alpha - 1.
		sum_ = const_coeff_alpha + linear_coeff_alpha * delta + quad_coeff_alpha * delta**2
		print(helper_object.alpha, ': ', sum_)

		return ((-linear_coeff_alpha + np.sqrt(linear_coeff_alpha**2 - 4 * quad_coeff_alpha * const_coeff_alpha)) / ( 2 * quad_coeff_alpha), \
		(-linear_coeff_alpha - np.sqrt(linear_coeff_alpha**2 - 4 * quad_coeff_alpha * const_coeff_alpha)) / ( 2 * quad_coeff_alpha))
	
	
	#Define all the alpha derivatives with respect to its coefficients
	def dalpha_dquad_(p1):
		quad_coeff_alpha = quad_coeff_alpha_(p1)
		linear_coeff_alpha = linear_coeff_alpha_(p1)
		const_coeff_alpha = const_coeff_alpha_(p1)
		return linear_coeff_alpha / (2 * quad_coeff_alpha**2) + np.sqrt(linear_coeff_alpha**2 - 4 * quad_coeff_alpha * const_coeff_alpha) \
	 			   / (2 * quad_coeff_alpha**2) + const_coeff_alpha / (quad_coeff_alpha * np.sqrt(linear_coeff_alpha**2 \
	 			   - 4 * quad_coeff_alpha * const_coeff_alpha))

	def dalpha_dlin_(p1):
		quad_coeff_alpha = quad_coeff_alpha_(p1)
		linear_coeff_alpha = linear_coeff_alpha_(p1)
		const_coeff_alpha = const_coeff_alpha_(p1)
		return - 1 / (2 * quad_coeff_alpha) + linear_coeff_alpha / ( 2 * quad_coeff_alpha * np.sqrt(linear_coeff_alpha**2 \
	 			  - 4 * quad_coeff_alpha * const_coeff_alpha))

	def dalpha_dconst_(p1):
		quad_coeff_alpha = quad_coeff_alpha_(p1)
		linear_coeff_alpha = linear_coeff_alpha_(p1)
		const_coeff_alpha = const_coeff_alpha_(p1)
		return const_coeff_alpha * np.sqrt(linear_coeff_alpha**2 - 4 * quad_coeff_alpha * const_coeff_alpha)


	#Define derivatives of the quadratic delta_alpha coefficient term (a in paper) with respect to the parameters
	def dquad_dp1_(p1):
		product = product_(p1)
		precision = np.linalg.inv(helper_object.covariance_matrix)
		temp = 0.0

		for m in range(len(helper_object.r)):
			for l in range(len(helper_object.r)):
				temp += 2 * precision[l][m] * (xi_tensor[l][0][1] * product[m][2] + xi_tensor[m][0][2] * product[l][1] \
				+ xi_tensor[m][0][1] * product[l][2] + xi_tensor[l][0][2] * product[m][1])

		return temp

	#Define derivatives of the linear delta_alpha coefficient term (a in paper) with respect to the parameters
	def dlin_dp1_(p1):
		product = product_(p1)
		temp = 0.0

		for m in range(len(helper_object.r)):
			for l in range(len(helper_object.r)):
				temp += precision[l][m] * (xi_tensor[l][0][0] * product[m][2] + product[l][0] * xi_tensor[m][0][2] \
			   + xi_tensor[l][0][1] * product[m][0] + product[l][1] * xi_tensor[m][0][0] \
			   - data_list[l] * xi_tensor[m][0][1] - data_list[m] * xi_tensor[l][0][1])

		return temp


	#Define derivatives of the linear delta_alpha coefficient term (a in paper) with respect to the parameters
	def dconst_dp1_(p1):
		product = product_(p1)
		temp = 0.0 

		for m in range(len(helper_object.r)):
			for l in range(len(helper_object.r)):
				temp += precision[l][m] * (xi_tensor[radial_bin_l][0][0] * product[m][1] + product[l][0] * xi_tensor[m][0][1] \
				 + xi_tensor[l][0][1] * product[m][0] + productl[1] * xi_tensor[m][0][0] - data_list[l] \
				 * xi_tensor[m][0][1] - data_list[m] * xi_tensor[l][0][1])

		return temp


	#Define alpah derivatives with respect to the parameters. Given the values of 1 as a temporary holder
	def dalpha_dp1_(p1):
		dalpha_dquad = dalpha_dquad_(p1)
		dquad_dp1 = dquad_dp1_(p1)
		dalpha_dlin = dalpha_dlin_(p1)
		dlin_dp1 = dlin_dp1_(p1)
		dalpha_dconst = dalpha_dconst_(p1)
		dconst_dp1 = dconst_dp1_(p1)

		return dalpha_dquad * dquad_dp1 + dalpha_dlin * dlin_dp1 + dalpha_dconst * dconst_dp1


	#Model derivatives
	def dmodel_l_dp1_(p1, dalpha):
		delta_alpha = dalpha
		dalpha_dp1 = dalpha_dp1_(p1)
		product = product_(p1)
		temp = 0.0

		for m in range(len(helper_object.r)):
			for l in range(len(helper_object.r)):
				temp += xi_tensor[radial_bin_l][0][0] + xi_tensor[l][0][1] * delta_alpha + xi_tensor[l][0][2] * delta_alpha**2 + \
				   dalpha_dp1 * (product[l][1] + delta_alpha * product[l][2])

		return temp

	def dmodel_m_dp1_(p1, dalpha):
		delta_alpha = dalpha
		dalpha_dp1 = dalpha_dp1_(p1)
		product = product_(p1)
		temp = 0.0

		for m in range(len(helper_object.r)):
			for l in range(len(helper_object.r)):
				temp += xi_tensor[radial_bin_m][0][0] + xi_tensor[m][0][1] * delta_alpha + xi_tensor[m][0][2] * delta_alpha**2 + \
				   dalpha_dp1 * (product[m][1] + delta_alpha * product[m][2])

		return temp


	#Loglikelihood equations, the derivatives with respect to the parameters
	#dLikelihood_dparam = precision_matrix * (-data_l * dmodel_m_dparam - data_m * dmodel_l_dparam + dmodel_l_dparam * model_m + dmodel_m_dparam * model_l)
	def dlog_dp1_(p1, dalpha):
		dmodel_l_dp1 = dmodel_l_dp1_(p1)
		dmodel_m_dp1 = dmodel_m_dp1_(p1)
		model_l = model_l_(p1)
		model_m = model_m_(p1)
		temp = 0.0

		for m in range(len(helper_object.r)):
			for l in range(len(helper_object.r)):
				temp += precision[l][m] * (-data_list[l] * dmodel_m_dp1 - data_list[m] * dmodel_l_dp1 + dmodel_l_dp1 * model_m \
			   + dmodel_m_dp1 * model_l)

		return temp

	def marginal_delta_alpha():
		green_a = 0.0
		green_b = 0.0
		green_c = 0.0
		green_p = 0.0
		green_q = 0.0

		red_a = 0.0
		red_b = 0.0
		red_c = 0.0

		black_a = 0.0
		black_b = 0.0
		black_c = 0.0
		black_d = 0.0
		black_e = 0.0

		temp_c = 0.0

		precision = np.linalg.inv(helper_object.covariance_matrix)

		print('Data_list: ', data_list[5])

		true_delta_alpha_list = np.linspace(-0.02, 0, 30)
		sums_b1 = np.zeros(len(true_delta_alpha_list))
		computation_list = np.zeros(len(true_delta_alpha_list))

		for l in range(len(helper_object.r)):
			for m in range(len(helper_object.r)):
				green_a += 0.25 * precision[l][m] * (data_list[l] * xi_IRrs_prime2[m] + data_list[m] * xi_IRrs_prime2[l])
				green_b += 0.5 * precision[l][m] * (data_list[l] * xi_IRrs_prime[m] + data_list[m] * xi_IRrs_prime[l])
				green_c += 0.5 * precision[l][m] * (data_list[l] * xi_IRrs[m] + data_list[m] * xi_IRrs[l])
				green_p += 0.5 * precision[l][m] * (data_list[l] * xi_IRrs_prime2[m] + data_list[m] * xi_IRrs_prime2[l])
				green_q += 0.5 * precision[l][m] * (data_list[l] * xi_IRrs_prime[m] + data_list[m] * xi_IRrs_prime[l])

				red_a += 0.75 * precision[l][m] * (xi_IRrs_prime[m] * xi_IRrs_prime2[l] + xi_IRrs_prime2[m] * xi_IRrs_prime[l])
				red_b += 0.5 * precision[l][m] * (xi_IRrs[m] * xi_IRrs_prime2[l] + 2 * xi_IRrs_prime[m] * xi_IRrs_prime[l] + xi_IRrs_prime2[m] * xi_IRrs[l])
				red_c += 0.5 * precision[l][m] * (xi_IRrs[m] * xi_IRrs_prime[l] + xi_IRrs_prime[m] * xi_IRrs[l])

				black_a += 0.125 * precision[l][m] * (xi_IRrs_prime2[m] * xi_IRrs_prime2[l])
				black_b += 0.25 * precision[l][m] * (xi_IRrs_prime[m] * xi_IRrs_prime2[l] + xi_IRrs_prime2[m] * xi_IRrs_prime[l])
				black_c += 0.5 * precision[l][m] * (0.5 * xi_IRrs[m] * xi_IRrs_prime2[l] + xi_IRrs_prime[m] * xi_IRrs_prime[l] \
							+ 0.5 * xi_IRrs_prime2[m] * xi_IRrs[l])
				black_d += 0.5 * precision[l][m] * (xi_IRrs[m] * xi_IRrs_prime[l] + xi_IRrs_prime[m] * xi_IRrs[l])
				black_e += 0.5 * precision[l][m] * (xi_IRrs[m] * xi_IRrs[l])

				temp_c += -.5 * precision[l][m] * data_list[l] * data_list[m]

				'''
				print('black_e: ', black_e)
				print('green_c: ', green_c)
				print('temp_c: ', temp_c)
				'''
				'''
				print('sum a, b, c: ', (- a * p1 + b * np.sqrt(p1) + c))

				print('data_list[l] * data_list[m]: ', data_list[l] * data_list[m])
				print('first: ', data_list[l] * (xi_IRrs[m] + xi_IRrs_prime[m] * true_delta_alpha + .5 * xi_IRrs_prime2[m] * true_delta_alpha**2) )
				print('second: ', data_list[m] * (xi_IRrs[l] + xi_IRrs_prime[l] * true_delta_alpha + .5 * xi_IRrs_prime2[l] * true_delta_alpha**2))
				print('third: ',  (xi_IRrs[m] + xi_IRrs_prime[m] * true_delta_alpha + .5 * xi_IRrs_prime2[m] * true_delta_alpha**2) \
						* (xi_IRrs[l] + xi_IRrs_prime[l] * true_delta_alpha + .5 * xi_IRrs_prime2[l] * true_delta_alpha**2))
				'''
				true_delta_alpha = -0.01
				data_both = data_list[l] * data_list[m]
				first = data_list[l] * (xi_IRrs[m] + xi_IRrs_prime[m] * true_delta_alpha + .5 * xi_IRrs_prime2[m] * true_delta_alpha**2) 
				second = data_list[m] * (xi_IRrs[l] + xi_IRrs_prime[l] * true_delta_alpha + .5 * xi_IRrs_prime2[l] * true_delta_alpha**2)
				third = (xi_IRrs[m] + xi_IRrs_prime[m] * true_delta_alpha + .5 * xi_IRrs_prime2[m] * true_delta_alpha**2) \
						* (xi_IRrs[l] + xi_IRrs_prime[l] * true_delta_alpha + .5 * xi_IRrs_prime2[l] * true_delta_alpha**2)
				

				#print('Sum: ', (data_both - first - second + third))
		'''
		for i, true_delta_alpha in enumerate(true_delta_alpha_list):
			a = black_a * true_delta_alpha**4 + black_b * true_delta_alpha**3 + black_c * true_delta_alpha**2 + black_d * true_delta_alpha + black_e
			b = green_a * true_delta_alpha**2 + green_b * true_delta_alpha + green_c
			c = temp_c

			b1_list = np.linspace(.2, 1.8, 10000)
			total_b1 = 0.0
			for b1 in b1_list:
				#print('b1: ',b1, ' ', (- a * b1**2 + b * b1 + c))
				total_b1 += np.exp(- a * b1**2 + b * b1 + c)

			total_b1 *= (b1_list[1] - b1_list[0])
			computation = np.exp(b*b / (4 * a) + c) * np.sqrt(np.pi / a)
			computation_list[i] = computation

			sums_b1[i] = total_b1
			print('Total: ', total_b1)
			print('computation: ', computation)


		plt.figure()
		plt.plot(true_delta_alpha_list, sums_b1, label='Numerical')
		plt.plot(true_delta_alpha_list, computation_list, label='Analytic')
		plt.legend()
		plt.show()
		'''

		beta = .5 * green_a * green_q / black_e + .5 * green_b * green_p / black_e - .5 * green_b * black_d * green_q / black_e**2 \
				- .5 * green_c * black_d * green_p / black_e**2 + .5 * green_c * (black_d**2 / black_e**3 - black_c / black_e**2) * green_q \
				- .25 * ((2 * green_a * green_c + green_b**2) / black_e**2) * red_c - .5 * green_b * green_c * red_b / black_e**2 \
				+ green_b * green_c * black_d * red_c / black_e**3 - .25 * green_c**2 * red_a / black_e**2 \
				+ .5 * green_c**2 * black_d * red_b / black_e**3 - .125 * green_c**2 * (6 * black_d**2 / black_e**4 - 4 * black_c / black_e**3) * red_c \
				- 0.5 * red_a / black_e + 0.5 * black_d * red_b / black_e**2 - 0.25 * ( 2 * black_d**2 / black_e**3 - 2 * black_c / black_e**2 ) * red_c

		gamma = .5 * green_b * green_q / black_e + .5 * green_c * green_p / black_e - .5 * green_c * black_d * green_q / black_e**2 \
				- .5 * green_b * green_c * red_c / black_e**2 - .25 * green_c**2 * red_b / black_e**2 + .5 * green_c**2 * black_d * red_c / black_e**3 \
				- 0.5 * red_b / black_e + 0.5 * black_d * red_c / black_e**2

		delta = .5 * green_c * green_q / black_e - .25 * green_c**2 * red_c / black_e**2 - 0.5 * red_c / black_e

		print('discriminat: ', gamma**2 - 4 * beta * delta)

		return (-gamma - np.sqrt(gamma**2 - 4 * beta * delta)) / (2 * beta) 

	#Code to solve the system of equations
	def functions_(parameters, dalpha):
		log1 = dlog_dp1_(p1, dalpha)

		return log1
	
	return marginal_delta_alpha()