'''
Coding up our solution to the BAO Fast Scaling problem. 
December 2020
Author(s): Matt Hansen
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
from mcfit import P2xi
from scipy.interpolate import InterpolatedUnivariateSpline as Spline


#I don't know how the data vector will look in the end, but right now I'm thinking that it can be a list where each index represents a 
#radial bin, and its corresponding value is the data at that radial bin
data_list = [12, 12]

#These two variables can represent what two radials bins we will select to due calculation off of. This should be easily converted into a 
#scalable model where we can take an arbitary number of bins and compute them with regards to two of them. I can even make this a OOP problem
radial_bin_l = 0
radial_bin_m = 1

'''
It might be easier to not do it the way we originally planned with xi1 - xi4 lists. I think having one "tensor" like 
how we had it in our paper might be better to program. I will try a simple case to see
'''
#radial bin at each index, 3x3 matrix at each radial bin
''' This would be at each index
[1 2 3
 4 5 6
 7 8 9
 10 11 12]
 Takes the form of [xi_i][xi_i'][xi_i''] these represent columns in 'vector space'
 I am just giving random values for now, so that the code is able to run
'''
xi_tensor = [np.array([[1, 4, 7], 
	                   [2, 5, 8], 
	                   [3, 6, 9],
	                   [10, 11, 12]]), np.array([[11, 14, 17], 
	                   							 [21, 15, 18],
	                   							 [13, 16, 19],
	                   							 [1, 31, 2]])]

#I'm giving the precision matrix a value of 1 right now so the code will compile. I do not know how it will look when we have the actual values
#We should talk about how to represent this, I imagine it would be some sort of list or 2d matrix. Either way this will be easy
#to change the code around since the variable is only used in the beginning of each definition and no where else
precision_matrix_lm = 1

def product_l_(p1, p2, p3, p4):
	p = [p1, p2, p3, p4]
	return np.matmul(p, xi_tensor[radial_bin_l])

def product_m_(p1, p2, p3, p4):
	p = [p1, p2, p3, p4]
	return np.matmul(p, xi_tensor[radial_bin_m])


#Model as a list with each index being a radial bin
def model_l_(p1, p2, p3, p4):
	productl = product_l_(p1, p2, p3, p4)
	return productl[0] + productl[1] + productl[2]

def model_m_(p1, p2, p3, p4):
	productm = product_m_(p1, p2, p3, p4)
	return productm[0] + productm[1] + productm[2]


def quad_coeff_alpha_(p1, p2, p3, p4):
	productl = product_l_(p1, p2, p3, p4)
	productm = product_m_(p1, p2, p3, p4)
	return 2 * precision_matrix_lm * (productl[1] * productm[2] + productl[2] * productm[1])

def linear_coeff_alpha_(p1, p2, p3, p4):
	productl = product_l_(p1, p2, p3, p4)
	productm = product_m_(p1, p2, p3, p4)
	return precision_matrix_lm * (productl[0] * productm[1] + productl[2] * productm[0] + 2 * productl[1] * productm[1] \
		   - data_list[radial_bin_l] * productm[2] - data_list[radial_bin_m] * productl[2])

def const_coeff_alpha_(p1, p2, p3, p4):
	productl = product_l_(p1, p2, p3, p4)
	productm = product_m_(p1, p2, p3, p4)
	return precision_matrix_lm * (productl[0] * productm[1] + productl[1] * productm[0] - data_list[radial_bin_l] * productm[1] - \
		   data_list[radial_bin_m] * productl[1])

#I'm not sure what delta_alpha we want to use
def delta_alpha_(p1, p2, p3, p4):
	quad_coeff_alpha = quad_coeff_alpha_(p1, p2, p3, p4)
	linear_coeff_alpha = linear_coeff_alpha_(p1, p2, p3, p4)
	const_coeff_alpha = const_coeff_alpha_(p1, p2, p3, p4)
	return (-linear_coeff_alpha + np.sqrt(linear_coeff_alpha**2 - 4 * quad_coeff_alpha * const_coeff_alpha) / ( 2 * quad_coeff_alpha))
#delta_alpha_minus = (- linear_coeff_alpha - sp.sqrt(linear_coeff_alpha**2 - 4 * quad_coeff_alpha * const_coeff_alpha) / ( 2 * quad_coeff_alpha))

#Define all the alpha derivatives with respect to its coefficients
def dalpha_dquad_(p1, p2, p3, p4):
	quad_coeff_alpha = quad_coeff_alpha_(p1, p2, p3, p4)
	linear_coeff_alpha = linear_coeff_alpha_(p1, p2, p3, p4)
	const_coeff_alpha = const_coeff_alpha_(p1, p2, p3, p4)
	return linear_coeff_alpha / (2 * quad_coeff_alpha**2) + np.sqrt(linear_coeff_alpha**2 - 4 * quad_coeff_alpha * const_coeff_alpha) \
 			   / (2 * quad_coeff_alpha**2) + const_coeff_alpha / (quad_coeff_alpha * np.sqrt(linear_coeff_alpha**2 \
 			   - 4 * quad_coeff_alpha * const_coeff_alpha))

def dalpha_dlin_(p1, p2, p3, p4):
	quad_coeff_alpha = quad_coeff_alpha_(p1, p2, p3, p4)
	linear_coeff_alpha = linear_coeff_alpha_(p1, p2, p3, p4)
	const_coeff_alpha = const_coeff_alpha_(p1, p2, p3, p4)
	return - 1 / (2 * quad_coeff_alpha) + linear_coeff_alpha / ( 2 * quad_coeff_alpha * np.sqrt(linear_coeff_alpha**2 \
 			  - 4 * quad_coeff_alpha * const_coeff_alpha))

def dalpha_dconst_(p1, p2, p3, p4):
	quad_coeff_alpha = quad_coeff_alpha_(p1, p2, p3, p4)
	linear_coeff_alpha = linear_coeff_alpha_(p1, p2, p3, p4)
	const_coeff_alpha = const_coeff_alpha_(p1, p2, p3, p4)
	return const_coeff_alpha * np.sqrt(linear_coeff_alpha**2 - 4 * quad_coeff_alpha * const_coeff_alpha)


#Define derivatives of the quadratic delta_alpha coefficient term (a in paper) with respect to the parameters
def dquad_dp1_(p1, p2, p3, p4):
	productl = product_l_(p1, p2, p3, p4)
	productm = product_m_(p1, p2, p3, p4)
	return 2 * precision_matrix_lm * (xi_tensor[radial_bin_l][0][1] * productm[2] + xi_tensor[radial_bin_m][0][2] * productl[1] \
			+ xi_tensor[radial_bin_m][0][1] * productl[2] + xi_tensor[radial_bin_l][0][2] * productm[1])

def dquad_dp2_(p1, p2, p3, p4):
	productl = product_l_(p1, p2, p3, p4)
	productm = product_m_(p1, p2, p3, p4)
	return 2 * precision_matrix_lm * (xi_tensor[radial_bin_l][1][1] * productm[2] + xi_tensor[radial_bin_m][1][2] * productl[1] \
			+ xi_tensor[radial_bin_m][1][1] * productl[2] + xi_tensor[radial_bin_l][1][2] * productm[1])

def dquad_dp3_(p1, p2, p3, p4):
	productl = product_l_(p1, p2, p3, p4)
	productm = product_m_(p1, p2, p3, p4)
	return 2 * precision_matrix_lm * (xi_tensor[radial_bin_l][2][1] * productm[2] + xi_tensor[radial_bin_m][2][2] * productl[1] \
			+ xi_tensor[radial_bin_m][2][1] * productl[2] + xi_tensor[radial_bin_l][2][2] * productm[1])

def dquad_dp4_(p1, p2, p3, p4):
	productl = product_l_(p1, p2, p3, p4)
	productm = product_m_(p1, p2, p3, p4)
	return 2 * precision_matrix_lm * (xi_tensor[radial_bin_l][3][1] * productm[2] + xi_tensor[radial_bin_m][3][2] * productl[1] \
			+ xi_tensor[radial_bin_m][3][1] * productl[2] + xi_tensor[radial_bin_l][3][2] * productm[1])


#Define derivatives of the linear delta_alpha coefficient term (a in paper) with respect to the parameters
def dlin_dp1_(p1, p2, p3, p4):
	productl = product_l_(p1, p2, p3, p4)
	productm = product_m_(p1, p2, p3, p4)
	return precision_matrix_lm * (xi_tensor[radial_bin_l][0][0] * productm[2] + productl[0] * xi_tensor[radial_bin_m][0][2] \
		   + xi_tensor[radial_bin_l][0][1] * productm[0] + productl[1] * xi_tensor[radial_bin_m][0][0] \
		   - data_list[radial_bin_l] * xi_tensor[radial_bin_m][0][1] - data_list[radial_bin_m] * xi_tensor[radial_bin_l][0][1])

def dlin_dp2_(p1, p2, p3, p4):
	productl = product_l_(p1, p2, p3, p4)
	productm = product_m_(p1, p2, p3, p4)
	return precision_matrix_lm * (xi_tensor[radial_bin_l][1][0] * productm[2] + productl[0] * xi_tensor[radial_bin_m][1][2] \
		   + xi_tensor[radial_bin_l][1][1] * productm[0] + productl[1] * xi_tensor[radial_bin_m][1][0] \
		   - data_list[radial_bin_l] * xi_tensor[radial_bin_m][1][1] - data_list[radial_bin_m] * xi_tensor[radial_bin_l][1][1])

def dlin_dp3_(p1, p2, p3, p4):
	productl = product_l_(p1, p2, p3, p4)
	productm = product_m_(p1, p2, p3, p4)
	return precision_matrix_lm * (xi_tensor[radial_bin_l][2][0] * productm[2] + productl[0] * xi_tensor[radial_bin_m][2][2] \
		   + xi_tensor[radial_bin_l][2][1] * productm[0] + productl[1] * xi_tensor[radial_bin_m][2][0] \
		   - data_list[radial_bin_l] * xi_tensor[radial_bin_m][2][1] - data_list[radial_bin_m] * xi_tensor[radial_bin_l][2][1])

def dlin_dp4_(p1, p2, p3, p4):
	productl = product_l_(p1, p2, p3, p4)
	productm = product_m_(p1, p2, p3, p4)
	return precision_matrix_lm * (xi_tensor[radial_bin_l][3][0] * productm[2] + productl[0] * xi_tensor[radial_bin_m][3][2] \
		   + xi_tensor[radial_bin_l][3][1] * productm[0] + productl[1] * xi_tensor[radial_bin_m][3][0] \
		   - data_list[radial_bin_l] * xi_tensor[radial_bin_m][3][1] - data_list[radial_bin_m] * xi_tensor[radial_bin_l][3][1])


#Define derivatives of the linear delta_alpha coefficient term (a in paper) with respect to the parameters
def dconst_dp1_(p1, p2, p3, p4):
	productl = product_l_(p1, p2, p3, p4)
	productm = product_m_(p1, p2, p3, p4)
	return precision_matrix_lm * (xi_tensor[radial_bin_l][0][0] * productm[1] + productl[0] * xi_tensor[radial_bin_m][0][1] \
			 + xi_tensor[radial_bin_l][0][1] * productm[0] + productl[1] * xi_tensor[radial_bin_m][0][0] - data_list[radial_bin_l] \
			 * xi_tensor[radial_bin_m][0][1] - data_list[radial_bin_m] * xi_tensor[radial_bin_l][0][1])

def dconst_dp2_(p1, p2, p3, p4):
	productl = product_l_(p1, p2, p3, p4)
	productm = product_m_(p1, p2, p3, p4)
	return precision_matrix_lm * (xi_tensor[radial_bin_l][1][0] * productm[1] + productl[0] * xi_tensor[radial_bin_m][1][1] \
			 + xi_tensor[radial_bin_l][1][1] * productm[0] + productl[1] * xi_tensor[radial_bin_m][1][0] - data_list[radial_bin_l] \
			 * xi_tensor[radial_bin_m][1][1] - data_list[radial_bin_m] * xi_tensor[radial_bin_l][1][1])

def dconst_dp3_(p1, p2, p3, p4):
	productl = product_l_(p1, p2, p3, p4)
	productm = product_m_(p1, p2, p3, p4)
	return precision_matrix_lm * (xi_tensor[radial_bin_l][2][0] * productm[1] + productl[0] * xi_tensor[radial_bin_m][2][1] \
			 + xi_tensor[radial_bin_l][2][1] * productm[0] + productl[1] * xi_tensor[radial_bin_m][2][0] - data_list[radial_bin_l] \
			 * xi_tensor[radial_bin_m][2][1] - data_list[radial_bin_m] * xi_tensor[radial_bin_l][2][1])

def dconst_dp4_(p1, p2, p3, p4):
	productl = product_l_(p1, p2, p3, p4)
	productm = product_m_(p1, p2, p3, p4)
	return precision_matrix_lm * (xi_tensor[radial_bin_l][3][0] * productm[1] + productl[0] * xi_tensor[radial_bin_m][3][1] \
			 + xi_tensor[radial_bin_l][3][1] * productm[0] + productl[1] * xi_tensor[radial_bin_m][3][0] - data_list[radial_bin_l] \
			 * xi_tensor[radial_bin_m][3][1] - data_list[radial_bin_m] * xi_tensor[radial_bin_l][3][1])


#Define alpah derivatives with respect to the parameters. Given the values of 1 as a temporary holder
def dalpha_dp1_(p1, p2, p3, p4):
	dalpha_dquad = dalpha_dquad_(p1, p2, p3, p4)
	dquad_dp1 = dquad_dp1_(p1, p2, p3, p4)
	dalpha_dlin = dalpha_dlin_(p1, p2, p3, p4)
	dlin_dp1 = dlin_dp1_(p1, p2, p3, p4)
	dalpha_dconst = dalpha_dconst_(p1, p2, p3, p4)
	dconst_dp1 = dconst_dp1_(p1, p2, p3, p4)

	return dalpha_dquad * dquad_dp1 + dalpha_dlin * dlin_dp1 + dalpha_dconst * dconst_dp1

def dalpha_dp2_(p1, p2, p3, p4):
	dalpha_dquad = dalpha_dquad_(p1, p2, p3, p4)
	dquad_dp2 = dquad_dp2_(p1, p2, p3, p4)
	dalpha_dlin = dalpha_dlin_(p1, p2, p3, p4)
	dlin_dp2 = dlin_dp2_(p1, p2, p3, p4)
	dalpha_dconst = dalpha_dconst_(p1, p2, p3, p4)
	dconst_dp2 = dconst_dp2_(p1, p2, p3, p4)

	return dalpha_dquad * dquad_dp2 + dalpha_dlin * dlin_dp2 + dalpha_dconst * dconst_dp2

def dalpha_dp3_(p1, p2, p3, p4):
	dalpha_dquad = dalpha_dquad_(p1, p2, p3, p4)
	dquad_dp3 = dquad_dp3_(p1, p2, p3, p4)
	dalpha_dlin = dalpha_dlin_(p1, p2, p3, p4)
	dlin_dp3 = dlin_dp3_(p1, p2, p3, p4)
	dalpha_dconst = dalpha_dconst_(p1, p2, p3, p4)
	dconst_dp3 = dconst_dp3_(p1, p2, p3, p4)

	return dalpha_dquad * dquad_dp3 + dalpha_dlin * dlin_dp3 + dalpha_dconst * dconst_dp3

def dalpha_dp4_(p1, p2, p3, p4):
	dalpha_dquad = dalpha_dquad_(p1, p2, p3, p4)
	dquad_dp4 = dquad_dp4_(p1, p2, p3, p4)
	dalpha_dlin = dalpha_dlin_(p1, p2, p3, p4)
	dlin_dp4 = dlin_dp4_(p1, p2, p3, p4)
	dalpha_dconst = dalpha_dconst_(p1, p2, p3, p4)
	dconst_dp4 = dconst_dp4_(p1, p2, p3, p4)
	return dalpha_dquad * dquad_dp4 + dalpha_dlin * dlin_dp4 + dalpha_dconst * dconst_dp4


#Model derivatives
def dmodel_l_dp1_(p1, p2, p3, p4):
	delta_alpha = delta_alpha_(p1, p2, p3, p4)
	dalpha_dp1 = dalpha_dp1_(p1, p2, p3, p4)
	productl = product_l_(p1, p2, p3, p4)
	productm = product_m_(p1, p2, p3, p4)

	return xi_tensor[radial_bin_l][0][0] + xi_tensor[radial_bin_l][0][1] * delta_alpha + xi_tensor[radial_bin_l][0][2] * delta_alpha**2 + \
			   dalpha_dp1 * (productl[1] + delta_alpha * productl[2])

def dmodel_m_dp1_(p1, p2, p3, p4):
	delta_alpha = delta_alpha_(p1, p2, p3, p4)
	dalpha_dp1 = dalpha_dp1_(p1, p2, p3, p4)
	productl = product_l_(p1, p2, p3, p4)
	productm = product_m_(p1, p2, p3, p4)

	return xi_tensor[radial_bin_m][0][0] + xi_tensor[radial_bin_m][0][1] * delta_alpha + xi_tensor[radial_bin_m][0][2] * delta_alpha**2 + \
			   dalpha_dp1 * (productm[1] + delta_alpha * productm[2])

def dmodel_l_dp2_(p1, p2, p3, p4):
	delta_alpha = delta_alpha_(p1, p2, p3, p4)
	dalpha_dp2 = dalpha_dp2_(p1, p2, p3, p4)
	productl = product_l_(p1, p2, p3, p4)
	productm = product_m_(p1, p2, p3, p4)

	return xi_tensor[radial_bin_l][1][0] + xi_tensor[radial_bin_l][1][1] * delta_alpha + xi_tensor[radial_bin_l][1][2] * delta_alpha**2 + \
			   dalpha_dp2 * (productl[1] + delta_alpha * productl[2])

def dmodel_m_dp2_(p1, p2, p3, p4):
	delta_alpha = delta_alpha_(p1, p2, p3, p4)
	dalpha_dp2 = dalpha_dp2_(p1, p2, p3, p4)
	productl = product_l_(p1, p2, p3, p4)
	productm = product_m_(p1, p2, p3, p4)

	return xi_tensor[radial_bin_m][1][0] + xi_tensor[radial_bin_m][1][1] * delta_alpha + xi_tensor[radial_bin_m][1][2] * delta_alpha**2 + \
			   dalpha_dp2 * (productm[1] + delta_alpha * productm[2])

def dmodel_l_dp3_(p1, p2, p3, p4):
	delta_alpha = delta_alpha_(p1, p2, p3, p4)
	dalpha_dp3 = dalpha_dp3_(p1, p2, p3, p4)
	productl = product_l_(p1, p2, p3, p4)
	productm = product_m_(p1, p2, p3, p4)

	return xi_tensor[radial_bin_l][2][0] + xi_tensor[radial_bin_l][2][1] * delta_alpha + xi_tensor[radial_bin_l][2][2] * delta_alpha**2 + \
			   dalpha_dp3 * (productl[1] + delta_alpha * productl[2])

def dmodel_m_dp3_(p1, p2, p3, p4):
	delta_alpha = delta_alpha_(p1, p2, p3, p4)
	dalpha_dp3 = dalpha_dp3_(p1, p2, p3, p4)
	productl = product_l_(p1, p2, p3, p4)
	productm = product_m_(p1, p2, p3, p4)

	return xi_tensor[radial_bin_m][2][0] + xi_tensor[radial_bin_m][2][1] * delta_alpha + xi_tensor[radial_bin_m][2][2] * delta_alpha**2 + \
			   dalpha_dp3 * (productm[1] + delta_alpha * productm[2])

def dmodel_l_dp4_(p1, p2, p3, p4):
	delta_alpha = delta_alpha_(p1, p2, p3, p4)
	dalpha_dp4 = dalpha_dp4_(p1, p2, p3, p4)
	productl = product_l_(p1, p2, p3, p4)
	productm = product_m_(p1, p2, p3, p4)

	return xi_tensor[radial_bin_l][3][0] + xi_tensor[radial_bin_l][3][1] * delta_alpha + xi_tensor[radial_bin_l][3][2] * delta_alpha**2 + \
			   dalpha_dp4 * (productl[1] + delta_alpha * productl[2])

def dmodel_m_dp4_(p1, p2, p3, p4):
	delta_alpha = delta_alpha_(p1, p2, p3, p4)
	dalpha_dp4 = dalpha_dp4_(p1, p2, p3, p4)
	productl = product_l_(p1, p2, p3, p4)
	productm = product_m_(p1, p2, p3, p4)

	return xi_tensor[radial_bin_m][3][0] + xi_tensor[radial_bin_m][3][1] * delta_alpha + xi_tensor[radial_bin_m][3][2] * delta_alpha**2 + \
			   dalpha_dp4 * (productm[1] + delta_alpha * productm[2])


#Loglikelihood equations, the derivatives with respect to the parameters
#dLikelihood_dparam = precision_matrix * (-data_l * dmodel_m_dparam - data_m * dmodel_l_dparam + dmodel_l_dparam * model_m + dmodel_m_dparam * model_l)
def dlog_dp1_(p1, p2, p3, p4):
	dmodel_l_dp1 = dmodel_l_dp1_(p1, p2, p3, p4)
	dmodel_m_dp1 = dmodel_m_dp1_(p1, p2, p3, p4)
	model_l = model_l_(p1, p2, p3, p4)
	model_m = model_m_(p1, p2, p3, p4)

	return precision_matrix_lm * (-data_list[radial_bin_l] * dmodel_m_dp1 - data_list[radial_bin_m] * dmodel_l_dp1 + dmodel_l_dp1 * model_m \
		   + dmodel_m_dp1 * model_l)

def dlog_dp2_(p1, p2, p3, p4):
	dmodel_l_dp2 = dmodel_l_dp2_(p1, p2, p3, p4)
	dmodel_m_dp2 = dmodel_m_dp2_(p1, p2, p3, p4)
	model_l = model_l_(p1, p2, p3, p4)
	model_m = model_m_(p1, p2, p3, p4)

	return precision_matrix_lm * (-data_list[radial_bin_l] * dmodel_m_dp2 - data_list[radial_bin_m] * dmodel_l_dp2 + dmodel_l_dp2 * model_m \
		   + dmodel_m_dp2 * model_l)

def dlog_dp3_(p1, p2, p3, p4):
	dmodel_l_dp3 = dmodel_l_dp3_(p1, p2, p3, p4)
	dmodel_m_dp3 = dmodel_m_dp3_(p1, p2, p3, p4)
	model_l = model_l_(p1, p2, p3, p4)
	model_m = model_m_(p1, p2, p3, p4)

	return precision_matrix_lm * (-data_list[radial_bin_l] * dmodel_m_dp3 - data_list[radial_bin_m] * dmodel_l_dp3 + dmodel_l_dp3 * model_m \
		   + dmodel_m_dp3 * model_l)

def dlog_dp4_(p1, p2, p3, p4):
	dmodel_l_dp4 = dmodel_l_dp4_(p1, p2, p3, p4)
	dmodel_m_dp4 = dmodel_m_dp4_(p1, p2, p3, p4)
	model_l = model_l_(p1, p2, p3, p4)
	model_m = model_m_(p1, p2, p3, p4)

	return precision_matrix_lm * (-data_list[radial_bin_l] * dmodel_m_dp4 - data_list[radial_bin_m] * dmodel_l_dp4 + dmodel_l_dp4 * model_m \
		   + dmodel_m_dp4 * model_l)


#Code to solve the system of equations
def functions_(parameters):
	p1, p2, p3, p4 = parameters[:4]

	log1 = dlog_dp1_(p1, p2, p3, p4)
	log2 = dlog_dp2_(p1, p2, p3, p4)
	log3 = dlog_dp3_(p1, p2, p3, p4)
	log4 = dlog_dp4_(p1, p2, p3, p4)

	return(log1, log2, log3, log4)

'''
x0 = np.array([1, 10, 1, 111])
sol = sci.optimize.root(functions_, x0, method='hybr')
'''


'''for i, (redshift, line) in enumerate(zip(z,['-','--'])):
	plt.loglog(kh, pk[i,:], color='k', ls = line)'''
'''
print(z)

plt.xlabel('k/h Mpc');
plt.legend(['z = 0', 'z = .8'], loc='lower left');
plt.title('Matter power at z=%s and z= %s'%tuple(z));
plt.show()
'''
#k, z, power_spectrum = CAMB_General_Code.get_linear_matter_power_spectrum()
'''
sigma = integrate.quad(lambda x: x**2 / (2 * math.pi**2) * ((3 * special.spherical_jn(1, 8*x)) / (8 * x))**2 * pk[0, int(x)], 0, 200)
print(sigma)'''


x_axis = [_ for _ in range(200)]
'''
plt.plot(x, kh)
plt.xlabel("Index")
plt.ylabel("kh")
plt.show()

plt.plot(x, np.log(kh))
plt.xlabel("Index")
plt.ylabel("log(kh)")
plt.show()'''
kh, z, pk = CAMB_General_Code.get_matter_spectrum()

#Made this function as a history to run all the graphs if need to be to check something
def old_graphs():
	plt.plot(kh, pk[0])
	plt.title('pk at z = 0 vs kh')
	plt.xlabel('kh')
	plt.ylabel('pk')
	plt.show()

	plt.plot(np.log(kh), pk[0])
	plt.title('pk at z = 0 vs log(kh)')
	plt.xlabel('log(kh)')
	plt.ylabel('pk')
	plt.show()

	plt.plot(kh, np.log(pk[0]))
	plt.xlabel("log(kh)")
	plt.ylabel("log(pk)")
	plt.title('log(pk) at z = 0')
	plt.show()

	dlogkh = np.gradient(kh)
	vector_want = 1 / (2 * math.pi**2 ) * (kh**2 * pk[0] * ((3 * special.spherical_jn(1, 8*kh)) / (8 * kh))**2) * dlogkh
	#Sigma_8^2
	print("My sigma: ", (np.sum(vector_want)))

	plt.plot(kh, np.log(kh**3 * pk[0] * np.exp(-kh**2)))
	plt.xlabel("log(kh)")
	plt.ylabel("log(kh**3 * pk * exp)")
	plt.title('log(kh**3 * pk * exp) at z=0')
	plt.show()

	#This should be the integral that you put in our paper, Zack
	r = np.arange(1,300)
	dlogkh = kh[1] - kh[0]
	xi_1_paper = [i for i in range(1, 300)]
	for i in range(len(r)):
		xi_1_paper[i] = r[i]**2 * np.sum(1 / (2 * math.pi**2) * kh**3 * special.spherical_jn(0, kh * r[i]) * pk[0]) * dlogkh 

	plt.plot(r, xi_1_paper)
	plt.xlabel("r")
	plt.ylabel("r[i]**2 * xi")
	plt.title('r[i]**2 * xi vs r')
	plt.show()


	#Below is attempting to find derivatives with formula form paper
	dxi_1_paper = [i for i in range(1, 300)]
	for i in range(len(r)):
		dxi_1_paper[i] = -r[i] * np.sum((kh**3 * dlogkh) / (2 * math.pi**2) * kh * special.spherical_jn(1, kh * r[i]) * pk[0] * np.exp(-kh**2))


	spl = Spline(r[1:], xi_1_paper[1:] / r[1:]**2.)

	x = spl(r * 1.001)
	y = spl(r * 0.999)

	dxi_1 = (x - y) / 0.002
	plt.figure()
	plt.plot(r, dxi_1_paper, '-g', label='paper')
	plt.plot(r, dxi_1, label='numerically')
	plt.xlabel("r")
	plt.ylabel("dxi")
	plt.title('dxi / d(alpha)|alpha = 1 {from paper}')
	plt.legend()
	plt.show()

	plt.plot(r, dxi_1)
	plt.title('dxi_1')
	plt.show()

'''
Below is the block of code that will be used to get xi_1
All of these are derived in our paper
'''
r = np.linspace(1., 300., 102)
xi_1 = np.zeros(len(r))
dxi_1 = np.zeros(len(r))
dlogkh = kh[1] - kh[0]

#I have a question about below equation. Zack said that I need to replace every dkh with with kh dlog(kh)
#Before I had kh**3 which was different than what the paper had, but I think it was correcting for the dlog(kh)
#I changed it back to kh**2, pulled the dlog(kh) outside the sum, and left the k resulting in u-sub in the sum (at the end)
for i in range(len(r)):
	xi_1[i] = r[i]**2 * np.sum(1 / (2 * math.pi**2) * kh**2 * special.spherical_jn(0, kh * r[i]) * pk[0] * kh) * dlogkh

plt.plot(r, xi_1)
plt.xlabel("r")
plt.ylabel("r[i]**2 * xi")
plt.title('r[i]**2 * xi vs r')
plt.show()


i = 0
for r_val in r:
	dxi_1[i] = -r[i] * np.sum(kh**2  / (2 * math.pi**2) * kh * special.spherical_jn(1, kh * r_val) * pk[0] * np.exp(-kh**2) * kh) * dlogkh
	i += 1

plt.plot(r, dxi_1)
plt.xlabel("r")
plt.ylabel("dxi_1")
plt.title('dxi_1 vs r')
plt.show()


xi_1_recovered = np.zeros(len(r))
for i in range(len(r)):
	xi_1_recovered[i] = (r[-1] - r[0]) / len(r) * np.sum(dxi_1[0:i])

plt.plot(r, xi_1_recovered)
plt.xlabel("r")
plt.ylabel("xi_1_recovered")
plt.title('xi_1_recovered vs r')
plt.show()

'''
#I'm not sure what this code is doing/ trying to do
def sigma_R_integrand(R, k):
	kr = k * R
	bess = (3. * special.spherical_jn(1, kr))/kr
	return k * k * bess * bess * pk_spline(k)
t0 = time.time()
sig = ((1./(2. * np.pi * np.pi)) * scipy.integrate.quad(lambda k: sigma_R_integrand(0.6, k), 0, np.inf,epsabs=1e2,epsrel=1e2)[0])**0.5
print('quad',time.time()-t0)
'''


'''
#I think something is wrong with what this produces
#This function should give us xi_1, the templates corresponding to linear bias
r, xi_1 = P2xi(kh)(pk[0])
print(xi_1)
print(r.shape)
#print("XI: ", xi_1.shape)

plt.plot(r[1:], xi_1[1:])
plt.xlabel("r")
plt.ylabel("xi_1")
plt.show()

#I think something wrong is happening with this function
r, xi_1 = P2xi(kh)(pk[0])
print(xi_1)
print(r.shape)
#Alex said that spline will allow us to interpolate the data to then take the derivative, numerically
spl = Spline(r, xi_1)

x = spl(r * 1.1)
y = spl(r * 0.9)

xi_1_prime = (x - y) / 0.2

plt.plot(r, xi_1)
plt.xlabel("r")
plt.ylabel("xi_1_prime")
plt.show()

xi_1_dprime = (x + y - 2*xi_1) / 0.1**2
'''
