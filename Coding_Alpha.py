'''
Coding up our solution to the BAO Fast Scaling problem. 
December 2020
Author(s): Matt Hansen
'''
import numpy as np
import sympy as sp
import math

#Define and declare all "constants" and variables
#The xi lists will take the form of [xi, xi', xi''] where each "row" will identify a distinct radial bin
xi1_list = [[2, 3], [1, 2], [2, 4]]
xi2_list = [[2, 3], [1, 2], [2, 4]]
xi3_list = [[2, 3], [1, 2], [2, 4]]
xi4_list = [[2, 3], [1, 2], [2, 4]]


p1, p2, p3, p4 = sp.symbols('p1, p2, p3, p4')

parameters = np.array([p1, p2, p3, p4])
data_list = [12, 12]

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
 Takes the form of [xi_i][xi_i'][xi_i'']
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

print(xi_tensor[0][2])
print(productl)
print(productm[0])

print(productm[0] * productl[0])

#I'm giving this a value of 1 right now so the code will compile. I do not know how it will look when we have the actual values
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
#I think this might be wrong
'''dquad_dp1 = 2 * precision_matrix_lm * (productl[1](xi_tensor[radial_bin_m][0][2] + xi_tensor[radial_bin_m][1][2] + xi_tensor[radial_bin_m][2][2] + \
			xi_tensor[radial_bin_m][3][2]) + productl[2](xi_tensor[radial_bin_m][0][1] + xi_tensor[radial_bin_m][1][1] + xi_tensor[radial_bin_m][2][1] + \
			xi_tensor[radial_bin_m][3][1]) + productm[1](xi_tensor[radial_bin_l][0][2] + xi_tensor[radial_bin_l][1][2] + xi_tensor[radial_bin_l][2][2] + \
			xi_tensor[radial_bin_l][3][2]) + productm[2](xi_tensor[radial_bin_l][0][1] + xi_tensor[radial_bin_l][1][1] + xi_tensor[radial_bin_l][2][1] + \
			xi_tensor[radial_bin_l][3][1]))'''

#Define alpah derivatives with respect to the parameters. Given the values of 1 as a temporary holder
dalpha_dp1 = 1

dalpha_dp2 = 1

dalpha_dp3 = 1

dalpha_dp4 = 1

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

