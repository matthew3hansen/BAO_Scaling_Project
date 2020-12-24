'''
Coding up our solution to the BAO Fast Scaling problem. 
December 2020
Author(s): Matt Hansen
'''
import numpy as np
import sympy as sp
import math

#Define symbols used for the parameters and their vector form.
p1, p2, p3, p4 = sp.symbols('p1, p2, p3, p4')

parameters = np.array([p1, p2, p3, p4])

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

productl = np.matmul(parameters, xi_tensor[radial_bin_l])
productm = np.matmul(parameters, xi_tensor[radial_bin_m])

#These print statements just form a visual aide of what these variables represent. Nothing more, just to check what the functions do
print(xi_tensor[0][2])
print(productl)
print(productm[0])

print(productm[0] * productl[0])

#I'm giving this a value of 1 right now so the code will compile. I do not know how it will look when we have the actual values
#We should talk about how to represent this, I imagine it would be some sort of list or 2d matrix. Either way this will be easy
#to change the code around since the variable is only used in the beginning of each definition and no where else
precision_matrix_lm = 1

quad_coeff_alpha = 2 * precision_matrix_lm * (productl[1] * productm[2] + productl[2] * productm[1])

linear_coeff_alpha = precision_matrix_lm * (productl[0] * productm[1] + productl[2] * productm[0] + 2 * productl[1] * productm[1] - \
					 data_list[radial_bin_l] * productm[2] - data_list[radial_bin_m] * productl[2])

const_coeff_alpha = precision_matrix_lm * (productl[0] * productm[1] + productl[1] * productm[0] - data_list[radial_bin_l] * productm[1] - \
					data_list[radial_bin_m] * productl[1])

#I'm not sure what delta_alpha we want to use
delta_alpha = (-linear_coeff_alpha + sp.sqrt(linear_coeff_alpha**2 - 4 * quad_coeff_alpha * const_coeff_alpha) / ( 2 * quad_coeff_alpha))
delta_alpha_minus = (- linear_coeff_alpha - sp.sqrt(linear_coeff_alpha**2 - 4 * quad_coeff_alpha * const_coeff_alpha) / ( 2 * quad_coeff_alpha))

#Define all the alpha derivatives with respect to its coefficients
dalpha_dquad = linear_coeff_alpha / (2 * quad_coeff_alpha**2) + sp.sqrt(linear_coeff_alpha**2 - 4 * quad_coeff_alpha * const_coeff_alpha) \
 			   / (2 * quad_coeff_alpha**2) + const_coeff_alpha / (quad_coeff_alpha * sp.sqrt(linear_coeff_alpha**2 \
 			   - 4 * quad_coeff_alpha * const_coeff_alpha))

dalpha_dlin = - 1 / (2 * quad_coeff_alpha) + linear_coeff_alpha / ( 2 * quad_coeff_alpha * sp.sqrt(linear_coeff_alpha**2 \
 			  - 4 * quad_coeff_alpha * const_coeff_alpha))

dalpha_dconst = const_coeff_alpha * sp.sqrt(linear_coeff_alpha**2 - 4 * quad_coeff_alpha * const_coeff_alpha)


#Define derivatives of the quadratic delta_alpha coefficient term (a in paper) with respect to the parameters
dquad_dp1 = 2 * precision_matrix_lm * (xi_tensor[radial_bin_l][0][1] * productm[2] + xi_tensor[radial_bin_m][0][2] * productl[1] \
			+ xi_tensor[radial_bin_m][0][1] * productl[2] + xi_tensor[radial_bin_l][0][2] * productm[1])

dquad_dp2 = 2 * precision_matrix_lm * (xi_tensor[radial_bin_l][1][1] * productm[2] + xi_tensor[radial_bin_m][1][2] * productl[1] \
			+ xi_tensor[radial_bin_m][1][1] * productl[2] + xi_tensor[radial_bin_l][1][2] * productm[1])

dquad_dp3 = 2 * precision_matrix_lm * (xi_tensor[radial_bin_l][2][1] * productm[2] + xi_tensor[radial_bin_m][2][2] * productl[1] \
			+ xi_tensor[radial_bin_m][2][1] * productl[2] + xi_tensor[radial_bin_l][2][2] * productm[1])

dquad_dp4 = 2 * precision_matrix_lm * (xi_tensor[radial_bin_l][3][1] * productm[2] + xi_tensor[radial_bin_m][3][2] * productl[1] \
			+ xi_tensor[radial_bin_m][3][1] * productl[2] + xi_tensor[radial_bin_l][3][2] * productm[1])


#Define derivatives of the linear delta_alpha coefficient term (a in paper) with respect to the parameters
dlin_dp1 = precision_matrix_lm * (xi_tensor[radial_bin_l][0][0] * productm[2] + productl[0] * xi_tensor[radial_bin_m][0][2] \
		   xi_tensor[radial_bin_l][0][1] * productm[0] + productl[1] * xi_tensor[radial_bin_m][0][0] \
		   - data_list[radial_bin_l] * xi_tensor[radial_bin_m][0][1] - data_list[radial_bin_m] * xi_tensor[radial_bin_l][0][1])

dlin_dp2 = precision_matrix_lm * (xi_tensor[radial_bin_l][1][0] * productm[2] + productl[0] * xi_tensor[radial_bin_m][1][2] \
		   xi_tensor[radial_bin_l][1][1] * productm[0] + productl[1] * xi_tensor[radial_bin_m][1][0] \
		   - data_list[radial_bin_l] * xi_tensor[radial_bin_m][1][1] - data_list[radial_bin_m] * xi_tensor[radial_bin_l][1][1])

dlin_dp3 = precision_matrix_lm * (xi_tensor[radial_bin_l][2][0] * productm[2] + productl[0] * xi_tensor[radial_bin_m][2][2] \
		   xi_tensor[radial_bin_l][2][1] * productm[0] + productl[1] * xi_tensor[radial_bin_m][2][0] \
		   - data_list[radial_bin_l] * xi_tensor[radial_bin_m][2][1] - data_list[radial_bin_m] * xi_tensor[radial_bin_l][2][1])

dlin_dp4 = precision_matrix_lm * (xi_tensor[radial_bin_l][3][0] * productm[2] + productl[0] * xi_tensor[radial_bin_m][3][2] \
		   xi_tensor[radial_bin_l][3][1] * productm[0] + productl[1] * xi_tensor[radial_bin_m][3][0] \
		   - data_list[radial_bin_l] * xi_tensor[radial_bin_m][3][1] - data_list[radial_bin_m] * xi_tensor[radial_bin_l][3][1])


#Define derivatives of the linear delta_alpha coefficient term (a in paper) with respect to the parameters
dconst_dp1 = precision_matrix_lm * (xi_tensor[radial_bin_l][0][0] * productm[1] + productl[0] * xi_tensor[radial_bin_m][0][1] \
			 + xi_tensor[radial_bin_l][0][1] * productm[0] + productl[1] * xi_tensor[radial_bin_m][0][0] - data_list[radial_bin_l] \
			 * xi_tensor[radial_bin_m][0][1] - data_list[radial_bin_m] * xi_tensor[radial_bin_l][0][1])

dconst_dp2 = precision_matrix_lm * (xi_tensor[radial_bin_l][1][0] * productm[1] + productl[0] * xi_tensor[radial_bin_m][1][1] \
			 + xi_tensor[radial_bin_l][1][1] * productm[0] + productl[1] * xi_tensor[radial_bin_m][1][0] - data_list[radial_bin_l] \
			 * xi_tensor[radial_bin_m][1][1] - data_list[radial_bin_m] * xi_tensor[radial_bin_l][1][1])

dconst_dp3 = precision_matrix_lm * (xi_tensor[radial_bin_l][2][0] * productm[1] + productl[0] * xi_tensor[radial_bin_m][2][1] \
			 + xi_tensor[radial_bin_l][2][1] * productm[0] + productl[1] * xi_tensor[radial_bin_m][2][0] - data_list[radial_bin_l] \
			 * xi_tensor[radial_bin_m][2][1] - data_list[radial_bin_m] * xi_tensor[radial_bin_l][2][1])

dconst_dp4 = precision_matrix_lm * (xi_tensor[radial_bin_l][3][0] * productm[1] + productl[0] * xi_tensor[radial_bin_m][3][1] \
			 + xi_tensor[radial_bin_l][3][1] * productm[0] + productl[1] * xi_tensor[radial_bin_m][3][0] - data_list[radial_bin_l] \
			 * xi_tensor[radial_bin_m][3][1] - data_list[radial_bin_m] * xi_tensor[radial_bin_l][3][1])


#Define alpah derivatives with respect to the parameters. Given the values of 1 as a temporary holder
dalpha_dp1 = dalpha_dquad * dquad_dp1 + dalpha_dlin * dlin_dp1 + dalpha_dconst * dconst_dp1

dalpha_dp2 = dalpha_dquad * dquad_dp2 + dalpha_dlin * dlin_dp2 + dalpha_dconst * dconst_dp2

dalpha_dp3 = dalpha_dquad * dquad_dp3 + dalpha_dlin * dlin_dp3 + dalpha_dconst * dconst_dp3

dalpha_dp4 = dalpha_dquad * dquad_dp4 + dalpha_dlin * dlin_dp4 + dalpha_dconst * dconst_dp4


#Model derivatives
dmodel_l_dp1 = xi_tensor[radial_bin_l][0][0] + xi_tensor[radial_bin_l][0][1] * delta_alpha + xi_tensor[radial_bin_l][0][2] * delta_alpha**2 + \
			   dalpha_dp1 * (productl[1] + delta_alpha * productl[2])

dmodel_m_dp1 = xi_tensor[radial_bin_m][0][0] + xi_tensor[radial_bin_m][0][1] * delta_alpha + xi_tensor[radial_bin_m][0][2] * delta_alpha**2 + \
			   dalpha_dp1 * (productm[1] + delta_alpha * productm[2])

dmodel_l_dp2 = xi_tensor[radial_bin_l][1][0] + xi_tensor[radial_bin_l][1][1] * delta_alpha + xi_tensor[radial_bin_l][1][2] * delta_alpha**2 + \
			   dalpha_dp2 * (productl[1] + delta_alpha * productl[2])

dmodel_m_dp2 = xi_tensor[radial_bin_m][1][0] + xi_tensor[radial_bin_m][1][1] * delta_alpha + xi_tensor[radial_bin_m][1][2] * delta_alpha**2 + \
			   dalpha_dp2 * (productm[1] + delta_alpha * productm[2])

dmodel_l_dp3 = xi_tensor[radial_bin_l][2][0] + xi_tensor[radial_bin_l][2][1] * delta_alpha + xi_tensor[radial_bin_l][2][2] * delta_alpha**2 + \
			   dalpha_dp3 * (productl[1] + delta_alpha * productl[2])

dmodel_m_dp3 = xi_tensor[radial_bin_m][2][0] + xi_tensor[radial_bin_m][2][1] * delta_alpha + xi_tensor[radial_bin_m][2][2] * delta_alpha**2 + \
			   dalpha_dp3 * (productm[1] + delta_alpha * productm[2])

dmodel_l_dp4 = xi_tensor[radial_bin_l][3][0] + xi_tensor[radial_bin_l][3][1] * delta_alpha + xi_tensor[radial_bin_l][3][2] * delta_alpha**2 + \
			   dalpha_dp2 * (productl[1] + delta_alpha * productl[2])

dmodel_m_dp4 = xi_tensor[radial_bin_m][3][0] + xi_tensor[radial_bin_m][3][1] * delta_alpha + xi_tensor[radial_bin_m][3][2] * delta_alpha**2 + \
			   dalpha_dp2 * (productm[1] + delta_alpha * productm[2])

#Loglikelihood equations
#dLikelihood_dparam = precision_matrix * (-data_l * dmodel_m_dparam - data_m * dmodel_l_dparam + dmodel_l_dparam * model_m + dmodel_m_dparam * model_l)

