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
import helper

def run(alpha, radial_bin_l_, radial_bin_m_):
	helper_object = helper.Info(alpha)
	helper_object.calc_CF()

	#I don't know how the data vector will look in the end, but right now I'm thinking that it can be a list where each index represents a 
	#radial bin, and its corresponding value is the data at that radial bin
	data_list = helper_object.get_data()

	covariance_matrix = helper_object.get_covariance_matrix()

	xi_IRrs, xi_d1d2, xi_d2d2, xi_d1s2, xi_d2s2, xi_s2s2, xi_d1p3 = helper_object.templates()

	xi_IRrs_prime, xi_d1d2_prime, xi_d2d2_prime, xi_d1s2_prime, xi_d2s2_prime, xi_s2s2_prime, xi_d1p3_prime = helper_object.templates_deriv()

	xi_IRrs_prime2, xi_d1d2_prime2, xi_d2d2_prime2, xi_d1s2_prime2, xi_d2s2_prime2, xi_s2s2_prime2, xi_d1p3_prime2 = helper_object.templates_deriv2()

	p1, p2, p3, p4, p5, p6, p7 = helper_object.get_biases()

	parameters = [p1, p2, p3, p4, p5, p6, p7]

	#These two variables can represent what two radials bins we will select to due calculation off of. This should be easily converted into a 
	#scalable model where we can take an arbitary number of bins and compute them with regards to two of them. I can even make this a OOP problem
	radial_bin_l = radial_bin_l_
	radial_bin_m = radial_bin_m_

	'''
	It might be easier to not do it the way we originally planned with xi1 - xi4 lists. I think having one "tensor" like 
	how we had it in our paper might be better to program. I will try a simple case to see
	'''
	#radial bin at each index, 3x3 matrix at each radial bin
	''' This would be at each index
	[p1, p2, p3, p4] [1 2 3]
	 				 [4 5 6]
	 				 [7 8 9]
	 				 [10 11 12]
	 Takes the form of [xi_i][xi_i'][xi_i''] these represent columns in 'vector space'
	 I am just giving random values for now, so that the code is able to run
	'''
	xi_tensor = [[] for _ in range(len(xi_IRrs))]
	for i in range(len(xi_IRrs)):
		xi_tensor[i].append([xi_IRrs[i], xi_IRrs_prime[i], xi_IRrs_prime2[i]])
		xi_tensor[i].append([xi_d1d2[i], xi_d1d2_prime[i], xi_d1d2_prime2[i]])
		xi_tensor[i].append([xi_d2d2[i], xi_d2d2_prime[i], xi_d2d2_prime2[i]])
		xi_tensor[i].append([xi_d1s2[i], xi_d1s2_prime[i], xi_d1s2_prime2[i]])
		xi_tensor[i].append([xi_d2s2[i], xi_d2s2_prime[i], xi_d2s2_prime2[i]])
		xi_tensor[i].append([xi_s2s2[i], xi_s2s2_prime[i], xi_s2s2_prime2[i]])
		xi_tensor[i].append([xi_d1p3[i], xi_d1p3_prime[i], xi_d1p3_prime2[i]])

	#I'm giving the precision matrix a value of 1 right now so the code will compile. I do not know how it will look when we have the actual values
	#We should talk about how to represent this, I imagine it would be some sort of list or 2d matrix. Either way this will be easy
	#to change the code around since the variable is only used in the beginning of each definition and no where else
	precision_matrix_lm = covariance_matrix[radial_bin_l][radial_bin_m]

	def product_l_(p1, p2, p3, p4, p5, p6, p7):
		p = np.array([p1, p2, p3, p4, p5, p6, p7])
		p = p.transpose()
		return np.matmul(p, np.array(xi_tensor[radial_bin_l])).reshape(3)

	def product_m_(p1, p2, p3, p4, p5, p6, p7):
		p = np.array([p1, p2, p3, p4, p5, p6, p7])
		p = p.transpose()
		return np.matmul(p, np.array(xi_tensor[radial_bin_m])).reshape(3)


	#Model as a list with each index being a radial bin
	def model_l_(p1, p2, p3, p4, p5, p6, p7):
		productl = product_l_(p1, p2, p3, p4, p5, p6, p7)
		return productl[0] + productl[1] + productl[2]

	def model_m_(p1, p2, p3, p4, p5, p6, p7):
		productm = product_m_(p1, p2, p3, p4, p5, p6, p7)
		return productm[0] + productm[1] + productm[2]


	def quad_coeff_alpha_(p1, p2, p3, p4, p5, p6, p7):
		productl = product_l_(p1, p2, p3, p4, p5, p6, p7)
		productm = product_m_(p1, p2, p3, p4, p5, p6, p7)
		return 2 * precision_matrix_lm * (productl[1] * productm[2] + productl[2] * productm[1])

	def linear_coeff_alpha_(p1, p2, p3, p4, p5, p6, p7):
		productl = product_l_(p1, p2, p3, p4, p5, p6, p7)
		productm = product_m_(p1, p2, p3, p4, p5, p6, p7)
		return precision_matrix_lm * (productl[0] * productm[1] + productl[2] * productm[0] + 2 * productl[1] * productm[1] \
			   - data_list[radial_bin_l] * productm[2] - data_list[radial_bin_m] * productl[2])

	def const_coeff_alpha_(p1, p2, p3, p4, p5, p6, p7):
		productl = product_l_(p1, p2, p3, p4, p5, p6, p7)
		productm = product_m_(p1, p2, p3, p4, p5, p6, p7)
		return precision_matrix_lm * (productl[0] * productm[1] + productl[1] * productm[0] - data_list[radial_bin_l] * productm[1] - \
			   data_list[radial_bin_m] * productl[1])

	#I'm not sure what delta_alpha we want to use
	def delta_alpha_(p1, p2, p3, p4, p5, p6, p7):
		quad_coeff_alpha = quad_coeff_alpha_(p1, p2, p3, p4, p5, p6, p7)
		linear_coeff_alpha = linear_coeff_alpha_(p1, p2, p3, p4, p5, p6, p7)
		const_coeff_alpha = const_coeff_alpha_(p1, p2, p3, p4, p5, p6, p7)
		return ((-linear_coeff_alpha + np.sqrt(linear_coeff_alpha**2 - 4 * quad_coeff_alpha * const_coeff_alpha)) / ( 2 * quad_coeff_alpha), \
		(-linear_coeff_alpha - np.sqrt(linear_coeff_alpha**2 - 4 * quad_coeff_alpha * const_coeff_alpha)) / ( 2 * quad_coeff_alpha))

	#Define all the alpha derivatives with respect to its coefficients
	def dalpha_dquad_(p1, p2, p3, p4, p5, p6, p7):
		quad_coeff_alpha = quad_coeff_alpha_(p1, p2, p3, p4, p5, p6, p7)
		linear_coeff_alpha = linear_coeff_alpha_(p1, p2, p3, p4, p5, p6, p7)
		const_coeff_alpha = const_coeff_alpha_(p1, p2, p3, p4, p5, p6, p7)
		return linear_coeff_alpha / (2 * quad_coeff_alpha**2) + np.sqrt(linear_coeff_alpha**2 - 4 * quad_coeff_alpha * const_coeff_alpha) \
	 			   / (2 * quad_coeff_alpha**2) + const_coeff_alpha / (quad_coeff_alpha * np.sqrt(linear_coeff_alpha**2 \
	 			   - 4 * quad_coeff_alpha * const_coeff_alpha))

	def dalpha_dlin_(p1, p2, p3, p4, p5, p6, p7):
		quad_coeff_alpha = quad_coeff_alpha_(p1, p2, p3, p4, p5, p6, p7)
		linear_coeff_alpha = linear_coeff_alpha_(p1, p2, p3, p4, p5, p6, p7)
		const_coeff_alpha = const_coeff_alpha_(p1, p2, p3, p4, p5, p6, p7)
		return - 1 / (2 * quad_coeff_alpha) + linear_coeff_alpha / ( 2 * quad_coeff_alpha * np.sqrt(linear_coeff_alpha**2 \
	 			  - 4 * quad_coeff_alpha * const_coeff_alpha))

	def dalpha_dconst_(p1, p2, p3, p4, p5, p6, p7):
		quad_coeff_alpha = quad_coeff_alpha_(p1, p2, p3, p4, p5, p6, p7)
		linear_coeff_alpha = linear_coeff_alpha_(p1, p2, p3, p4, p5, p6, p7)
		const_coeff_alpha = const_coeff_alpha_(p1, p2, p3, p4, p5, p6, p7)
		return const_coeff_alpha * np.sqrt(linear_coeff_alpha**2 - 4 * quad_coeff_alpha * const_coeff_alpha)


	#Define derivatives of the quadratic delta_alpha coefficient term (a in paper) with respect to the parameters
	def dquad_dp1_(p1, p2, p3, p4, p5, p6, p7):
		productl = product_l_(p1, p2, p3, p4, p5, p6, p7)
		productm = product_m_(p1, p2, p3, p4, p5, p6, p7)
		return 2 * precision_matrix_lm * (xi_tensor[radial_bin_l][0][1] * productm[2] + xi_tensor[radial_bin_m][0][2] * productl[1] \
				+ xi_tensor[radial_bin_m][0][1] * productl[2] + xi_tensor[radial_bin_l][0][2] * productm[1])

	def dquad_dp2_(p1, p2, p3, p4, p5, p6, p7):
		productl = product_l_(p1, p2, p3, p4, p5, p6, p7)
		productm = product_m_(p1, p2, p3, p4, p5, p6, p7)
		return 2 * precision_matrix_lm * (xi_tensor[radial_bin_l][1][1] * productm[2] + xi_tensor[radial_bin_m][1][2] * productl[1] \
				+ xi_tensor[radial_bin_m][1][1] * productl[2] + xi_tensor[radial_bin_l][1][2] * productm[1])

	def dquad_dp3_(p1, p2, p3, p4, p5, p6, p7):
		productl = product_l_(p1, p2, p3, p4, p5, p6, p7)
		productm = product_m_(p1, p2, p3, p4, p5, p6, p7)
		return 2 * precision_matrix_lm * (xi_tensor[radial_bin_l][2][1] * productm[2] + xi_tensor[radial_bin_m][2][2] * productl[1] \
				+ xi_tensor[radial_bin_m][2][1] * productl[2] + xi_tensor[radial_bin_l][2][2] * productm[1])

	def dquad_dp4_(p1, p2, p3, p4, p5, p6, p7):
		productl = product_l_(p1, p2, p3, p4, p5, p6, p7)
		productm = product_m_(p1, p2, p3, p4, p5, p6, p7)
		return 2 * precision_matrix_lm * (xi_tensor[radial_bin_l][3][1] * productm[2] + xi_tensor[radial_bin_m][3][2] * productl[1] \
				+ xi_tensor[radial_bin_m][3][1] * productl[2] + xi_tensor[radial_bin_l][3][2] * productm[1])


	#Define derivatives of the linear delta_alpha coefficient term (a in paper) with respect to the parameters
	def dlin_dp1_(p1, p2, p3, p4, p5, p6, p7):
		productl = product_l_(p1, p2, p3, p4, p5, p6, p7)
		productm = product_m_(p1, p2, p3, p4, p5, p6, p7)
		return precision_matrix_lm * (xi_tensor[radial_bin_l][0][0] * productm[2] + productl[0] * xi_tensor[radial_bin_m][0][2] \
			   + xi_tensor[radial_bin_l][0][1] * productm[0] + productl[1] * xi_tensor[radial_bin_m][0][0] \
			   - data_list[radial_bin_l] * xi_tensor[radial_bin_m][0][1] - data_list[radial_bin_m] * xi_tensor[radial_bin_l][0][1])

	def dlin_dp2_(p1, p2, p3, p4, p5, p6, p7):
		productl = product_l_(p1, p2, p3, p4, p5, p6, p7)
		productm = product_m_(p1, p2, p3, p4, p5, p6, p7)
		return precision_matrix_lm * (xi_tensor[radial_bin_l][1][0] * productm[2] + productl[0] * xi_tensor[radial_bin_m][1][2] \
			   + xi_tensor[radial_bin_l][1][1] * productm[0] + productl[1] * xi_tensor[radial_bin_m][1][0] \
			   - data_list[radial_bin_l] * xi_tensor[radial_bin_m][1][1] - data_list[radial_bin_m] * xi_tensor[radial_bin_l][1][1])

	def dlin_dp3_(p1, p2, p3, p4, p5, p6, p7):
		productl = product_l_(p1, p2, p3, p4, p5, p6, p7)
		productm = product_m_(p1, p2, p3, p4, p5, p6, p7)
		return precision_matrix_lm * (xi_tensor[radial_bin_l][2][0] * productm[2] + productl[0] * xi_tensor[radial_bin_m][2][2] \
			   + xi_tensor[radial_bin_l][2][1] * productm[0] + productl[1] * xi_tensor[radial_bin_m][2][0] \
			   - data_list[radial_bin_l] * xi_tensor[radial_bin_m][2][1] - data_list[radial_bin_m] * xi_tensor[radial_bin_l][2][1])

	def dlin_dp4_(p1, p2, p3, p4, p5, p6, p7):
		productl = product_l_(p1, p2, p3, p4, p5, p6, p7)
		productm = product_m_(p1, p2, p3, p4, p5, p6, p7)
		return precision_matrix_lm * (xi_tensor[radial_bin_l][3][0] * productm[2] + productl[0] * xi_tensor[radial_bin_m][3][2] \
			   + xi_tensor[radial_bin_l][3][1] * productm[0] + productl[1] * xi_tensor[radial_bin_m][3][0] \
			   - data_list[radial_bin_l] * xi_tensor[radial_bin_m][3][1] - data_list[radial_bin_m] * xi_tensor[radial_bin_l][3][1])


	#Define derivatives of the linear delta_alpha coefficient term (a in paper) with respect to the parameters
	def dconst_dp1_(p1, p2, p3, p4, p5, p6, p7):
		productl = product_l_(p1, p2, p3, p4, p5, p6, p7)
		productm = product_m_(p1, p2, p3, p4, p5, p6, p7)
		return precision_matrix_lm * (xi_tensor[radial_bin_l][0][0] * productm[1] + productl[0] * xi_tensor[radial_bin_m][0][1] \
				 + xi_tensor[radial_bin_l][0][1] * productm[0] + productl[1] * xi_tensor[radial_bin_m][0][0] - data_list[radial_bin_l] \
				 * xi_tensor[radial_bin_m][0][1] - data_list[radial_bin_m] * xi_tensor[radial_bin_l][0][1])

	def dconst_dp2_(p1, p2, p3, p4, p5, p6, p7):
		productl = product_l_(p1, p2, p3, p4, p5, p6, p7)
		productm = product_m_(p1, p2, p3, p4, p5, p6, p7)
		return precision_matrix_lm * (xi_tensor[radial_bin_l][1][0] * productm[1] + productl[0] * xi_tensor[radial_bin_m][1][1] \
				 + xi_tensor[radial_bin_l][1][1] * productm[0] + productl[1] * xi_tensor[radial_bin_m][1][0] - data_list[radial_bin_l] \
				 * xi_tensor[radial_bin_m][1][1] - data_list[radial_bin_m] * xi_tensor[radial_bin_l][1][1])

	def dconst_dp3_(p1, p2, p3, p4, p5, p6, p7):
		productl = product_l_(p1, p2, p3, p4, p5, p6, p7)
		productm = product_m_(p1, p2, p3, p4, p5, p6, p7)
		return precision_matrix_lm * (xi_tensor[radial_bin_l][2][0] * productm[1] + productl[0] * xi_tensor[radial_bin_m][2][1] \
				 + xi_tensor[radial_bin_l][2][1] * productm[0] + productl[1] * xi_tensor[radial_bin_m][2][0] - data_list[radial_bin_l] \
				 * xi_tensor[radial_bin_m][2][1] - data_list[radial_bin_m] * xi_tensor[radial_bin_l][2][1])

	def dconst_dp4_(p1, p2, p3, p4, p5, p6, p7):
		productl = product_l_(p1, p2, p3, p4, p5, p6, p7)
		productm = product_m_(p1, p2, p3, p4, p5, p6, p7)
		return precision_matrix_lm * (xi_tensor[radial_bin_l][3][0] * productm[1] + productl[0] * xi_tensor[radial_bin_m][3][1] \
				 + xi_tensor[radial_bin_l][3][1] * productm[0] + productl[1] * xi_tensor[radial_bin_m][3][0] - data_list[radial_bin_l] \
				 * xi_tensor[radial_bin_m][3][1] - data_list[radial_bin_m] * xi_tensor[radial_bin_l][3][1])


	#Define alpah derivatives with respect to the parameters. Given the values of 1 as a temporary holder
	def dalpha_dp1_(p1, p2, p3, p4, p5, p6, p7):
		dalpha_dquad = dalpha_dquad_(p1, p2, p3, p4, p5, p6, p7)
		dquad_dp1 = dquad_dp1_(p1, p2, p3, p4, p5, p6, p7)
		dalpha_dlin = dalpha_dlin_(p1, p2, p3, p4, p5, p6, p7)
		dlin_dp1 = dlin_dp1_(p1, p2, p3, p4, p5, p6, p7)
		dalpha_dconst = dalpha_dconst_(p1, p2, p3, p4, p5, p6, p7)
		dconst_dp1 = dconst_dp1_(p1, p2, p3, p4, p5, p6, p7)

		return dalpha_dquad * dquad_dp1 + dalpha_dlin * dlin_dp1 + dalpha_dconst * dconst_dp1

	def dalpha_dp2_(p1, p2, p3, p4, p5, p6, p7):
		dalpha_dquad = dalpha_dquad_(p1, p2, p3, p4, p5, p6, p7)
		dquad_dp2 = dquad_dp2_(p1, p2, p3, p4, p5, p6, p7)
		dalpha_dlin = dalpha_dlin_(p1, p2, p3, p4, p5, p6, p7)
		dlin_dp2 = dlin_dp2_(p1, p2, p3, p4, p5, p6, p7)
		dalpha_dconst = dalpha_dconst_(p1, p2, p3, p4, p5, p6, p7)
		dconst_dp2 = dconst_dp2_(p1, p2, p3, p4, p5, p6, p7)

		return dalpha_dquad * dquad_dp2 + dalpha_dlin * dlin_dp2 + dalpha_dconst * dconst_dp2

	def dalpha_dp3_(p1, p2, p3, p4, p5, p6, p7):
		dalpha_dquad = dalpha_dquad_(p1, p2, p3, p4, p5, p6, p7)
		dquad_dp3 = dquad_dp3_(p1, p2, p3, p4, p5, p6, p7)
		dalpha_dlin = dalpha_dlin_(p1, p2, p3, p4, p5, p6, p7)
		dlin_dp3 = dlin_dp3_(p1, p2, p3, p4, p5, p6, p7)
		dalpha_dconst = dalpha_dconst_(p1, p2, p3, p4, p5, p6, p7)
		dconst_dp3 = dconst_dp3_(p1, p2, p3, p4, p5, p6, p7)

		return dalpha_dquad * dquad_dp3 + dalpha_dlin * dlin_dp3 + dalpha_dconst * dconst_dp3

	def dalpha_dp4_(p1, p2, p3, p4, p5, p6, p7):
		dalpha_dquad = dalpha_dquad_(p1, p2, p3, p4, p5, p6, p7)
		dquad_dp4 = dquad_dp4_(p1, p2, p3, p4, p5, p6, p7)
		dalpha_dlin = dalpha_dlin_(p1, p2, p3, p4, p5, p6, p7)
		dlin_dp4 = dlin_dp4_(p1, p2, p3, p4, p5, p6, p7)
		dalpha_dconst = dalpha_dconst_(p1, p2, p3, p4, p5, p6, p7)
		dconst_dp4 = dconst_dp4_(p1, p2, p3, p4, p5, p6, p7)
		return dalpha_dquad * dquad_dp4 + dalpha_dlin * dlin_dp4 + dalpha_dconst * dconst_dp4


	#Model derivatives
	def dmodel_l_dp1_(p1, p2, p3, p4, p5, p6, p7):
		delta_alpha = delta_alpha_(p1, p2, p3, p4, p5, p6, p7)
		dalpha_dp1 = dalpha_dp1_(p1, p2, p3, p4, p5, p6, p7)
		productl = product_l_(p1, p2, p3, p4, p5, p6, p7)
		productm = product_m_(p1, p2, p3, p4, p5, p6, p7)

		return xi_tensor[radial_bin_l][0][0] + xi_tensor[radial_bin_l][0][1] * delta_alpha + xi_tensor[radial_bin_l][0][2] * delta_alpha**2 + \
				   dalpha_dp1 * (productl[1] + delta_alpha * productl[2])

	def dmodel_m_dp1_(p1, p2, p3, p4, p5, p6, p7):
		delta_alpha = delta_alpha_(p1, p2, p3, p4, p5, p6, p7)
		dalpha_dp1 = dalpha_dp1_(p1, p2, p3, p4, p5, p6, p7)
		productl = product_l_(p1, p2, p3, p4, p5, p6, p7)
		productm = product_m_(p1, p2, p3, p4, p5, p6, p7)

		return xi_tensor[radial_bin_m][0][0] + xi_tensor[radial_bin_m][0][1] * delta_alpha + xi_tensor[radial_bin_m][0][2] * delta_alpha**2 + \
				   dalpha_dp1 * (productm[1] + delta_alpha * productm[2])

	def dmodel_l_dp2_(p1, p2, p3, p4, p5, p6, p7):
		delta_alpha = delta_alpha_(p1, p2, p3, p4, p5, p6, p7)
		dalpha_dp2 = dalpha_dp2_(p1, p2, p3, p4, p5, p6, p7)
		productl = product_l_(p1, p2, p3, p4, p5, p6, p7)
		productm = product_m_(p1, p2, p3, p4, p5, p6, p7)

		return xi_tensor[radial_bin_l][1][0] + xi_tensor[radial_bin_l][1][1] * delta_alpha + xi_tensor[radial_bin_l][1][2] * delta_alpha**2 + \
				   dalpha_dp2 * (productl[1] + delta_alpha * productl[2])

	def dmodel_m_dp2_(p1, p2, p3, p4, p5, p6, p7):
		delta_alpha = delta_alpha_(p1, p2, p3, p4, p5, p6, p7)
		dalpha_dp2 = dalpha_dp2_(p1, p2, p3, p4, p5, p6, p7)
		productl = product_l_(p1, p2, p3, p4, p5, p6, p7)
		productm = product_m_(p1, p2, p3, p4, p5, p6, p7)

		return xi_tensor[radial_bin_m][1][0] + xi_tensor[radial_bin_m][1][1] * delta_alpha + xi_tensor[radial_bin_m][1][2] * delta_alpha**2 + \
				   dalpha_dp2 * (productm[1] + delta_alpha * productm[2])

	def dmodel_l_dp3_(p1, p2, p3, p4, p5, p6, p7):
		delta_alpha = delta_alpha_(p1, p2, p3, p4, p5, p6, p7)
		dalpha_dp3 = dalpha_dp3_(p1, p2, p3, p4, p5, p6, p7)
		productl = product_l_(p1, p2, p3, p4, p5, p6, p7)
		productm = product_m_(p1, p2, p3, p4, p5, p6, p7)

		return xi_tensor[radial_bin_l][2][0] + xi_tensor[radial_bin_l][2][1] * delta_alpha + xi_tensor[radial_bin_l][2][2] * delta_alpha**2 + \
				   dalpha_dp3 * (productl[1] + delta_alpha * productl[2])

	def dmodel_m_dp3_(p1, p2, p3, p4, p5, p6, p7):
		delta_alpha = delta_alpha_(p1, p2, p3, p4, p5, p6, p7)
		dalpha_dp3 = dalpha_dp3_(p1, p2, p3, p4, p5, p6, p7)
		productl = product_l_(p1, p2, p3, p4, p5, p6, p7)
		productm = product_m_(p1, p2, p3, p4, p5, p6, p7)

		return xi_tensor[radial_bin_m][2][0] + xi_tensor[radial_bin_m][2][1] * delta_alpha + xi_tensor[radial_bin_m][2][2] * delta_alpha**2 + \
				   dalpha_dp3 * (productm[1] + delta_alpha * productm[2])

	def dmodel_l_dp4_(p1, p2, p3, p4, p5, p6, p7):
		delta_alpha = delta_alpha_(p1, p2, p3, p4, p5, p6, p7)
		dalpha_dp4 = dalpha_dp4_(p1, p2, p3, p4, p5, p6, p7)
		productl = product_l_(p1, p2, p3, p4, p5, p6, p7)
		productm = product_m_(p1, p2, p3, p4, p5, p6, p7)

		return xi_tensor[radial_bin_l][3][0] + xi_tensor[radial_bin_l][3][1] * delta_alpha + xi_tensor[radial_bin_l][3][2] * delta_alpha**2 + \
				   dalpha_dp4 * (productl[1] + delta_alpha * productl[2])

	def dmodel_m_dp4_(p1, p2, p3, p4, p5, p6, p7):
		delta_alpha = delta_alpha_(p1, p2, p3, p4, p5, p6, p7)
		dalpha_dp4 = dalpha_dp4_(p1, p2, p3, p4, p5, p6, p7)
		productl = product_l_(p1, p2, p3, p4, p5, p6, p7)
		productm = product_m_(p1, p2, p3, p4, p5, p6, p7)

		return xi_tensor[radial_bin_m][3][0] + xi_tensor[radial_bin_m][3][1] * delta_alpha + xi_tensor[radial_bin_m][3][2] * delta_alpha**2 + \
				   dalpha_dp4 * (productm[1] + delta_alpha * productm[2])


	#Loglikelihood equations, the derivatives with respect to the parameters
	#dLikelihood_dparam = precision_matrix * (-data_l * dmodel_m_dparam - data_m * dmodel_l_dparam + dmodel_l_dparam * model_m + dmodel_m_dparam * model_l)
	def dlog_dp1_(p1, p2, p3, p4, p5, p6, p7):
		dmodel_l_dp1 = dmodel_l_dp1_(p1, p2, p3, p4, p5, p6, p7)
		dmodel_m_dp1 = dmodel_m_dp1_(p1, p2, p3, p4, p5, p6, p7)
		model_l = model_l_(p1, p2, p3, p4, p5, p6, p7)
		model_m = model_m_(p1, p2, p3, p4, p5, p6, p7)

		return precision_matrix_lm * (-data_list[radial_bin_l] * dmodel_m_dp1 - data_list[radial_bin_m] * dmodel_l_dp1 + dmodel_l_dp1 * model_m \
			   + dmodel_m_dp1 * model_l)

	def dlog_dp2_(p1, p2, p3, p4, p5, p6, p7):
		dmodel_l_dp2 = dmodel_l_dp2_(p1, p2, p3, p4, p5, p6, p7)
		dmodel_m_dp2 = dmodel_m_dp2_(p1, p2, p3, p4, p5, p6, p7)
		model_l = model_l_(p1, p2, p3, p4, p5, p6, p7)
		model_m = model_m_(p1, p2, p3, p4, p5, p6, p7)

		return precision_matrix_lm * (-data_list[radial_bin_l] * dmodel_m_dp2 - data_list[radial_bin_m] * dmodel_l_dp2 + dmodel_l_dp2 * model_m \
			   + dmodel_m_dp2 * model_l)

	def dlog_dp3_(p1, p2, p3, p4, p5, p6, p7):
		dmodel_l_dp3 = dmodel_l_dp3_(p1, p2, p3, p4, p5, p6, p7)
		dmodel_m_dp3 = dmodel_m_dp3_(p1, p2, p3, p4, p5, p6, p7)
		model_l = model_l_(p1, p2, p3, p4, p5, p6, p7)
		model_m = model_m_(p1, p2, p3, p4, p5, p6, p7)

		return precision_matrix_lm * (-data_list[radial_bin_l] * dmodel_m_dp3 - data_list[radial_bin_m] * dmodel_l_dp3 + dmodel_l_dp3 * model_m \
			   + dmodel_m_dp3 * model_l)

	def dlog_dp4_(p1, p2, p3, p4, p5, p6, p7):
		dmodel_l_dp4 = dmodel_l_dp4_(p1, p2, p3, p4, p5, p6, p7)
		dmodel_m_dp4 = dmodel_m_dp4_(p1, p2, p3, p4, p5, p6, p7)
		model_l = model_l_(p1, p2, p3, p4, p5, p6, p7)
		model_m = model_m_(p1, p2, p3, p4, p5, p6, p7)

		return precision_matrix_lm * (-data_list[radial_bin_l] * dmodel_m_dp4 - data_list[radial_bin_m] * dmodel_l_dp4 + dmodel_l_dp4 * model_m \
			   + dmodel_m_dp4 * model_l)


	#Code to solve the system of equations
	def functions_(parameters):
		p1, p2, p3, p4, p5, p6, p7 = parameters[:7]

		log1 = dlog_dp1_(p1, p2, p3, p4, p5, p6, p7)
		log2 = dlog_dp2_(p1, p2, p3, p4, p5, p6, p7)
		log3 = dlog_dp3_(p1, p2, p3, p4, p5, p6, p7)
		log4 = dlog_dp4_(p1, p2, p3, p4, p5, p6, p7)

		return(log1, log2, log3, log4)

	return delta_alpha_(p1, p2, p3, p4, p5, p6, p7)