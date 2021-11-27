'''
Coding up our solution to the BAO Fast Scaling problem. 
Created October 10, 2021
Author(s): Matt Hansen, Alex Krolewski
'''
import numpy as np
import scipy as sp
import BAO_scale_fitting_helper
import time
import timeit
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from matplotlib import pyplot as plt

def fixed_linear_bias(alpha = 1.0):
    helper_object = BAO_scale_fitting_helper.Info(alpha)
        
    helper_object.calc_covariance_matrix()
    
    #difference = helper_object.covariance_matrix - helper_object.covariance_matrix_old
        
    helper_object.calc_CF()
    
    data_list = helper_object.get_data()
    
    covariance_matrix = helper_object.get_covariance_matrix()
    
    xi_IRrs = helper_object.templates()
    
    xi_IRrs_prime = helper_object.templates_deriv()
    
    xi_IRrs_prime2 = helper_object.templates_deriv2()
    
    xi_IRrs_prime3 = helper_object.templates_deriv3()

    precision = np.linalg.inv(covariance_matrix)
    b1 = helper_object.get_biases()
    #start = time.time()
    
    temp = 3 * b1 * np.multiply.outer(xi_IRrs_prime, xi_IRrs_prime2)
    temp1 = temp.transpose()
    temp2 = b1 * np.multiply.outer(xi_IRrs_prime3, xi_IRrs)
    temp3 = temp2.transpose()
    temp4 = -1 * np.multiply.outer(xi_IRrs_prime3, data_list)
    temp5 = temp4.transpose()
    a_vec = (0.5 * b1 * np.multiply(precision, (temp + temp1 + temp2 + temp3 + temp4 + temp5))).sum()
    
    temp1 = b1 * np.multiply.outer(xi_IRrs, xi_IRrs_prime2)
    temp2 = temp1.transpose()
    temp3 = 2 * b1 * np.multiply.outer(xi_IRrs_prime, xi_IRrs_prime)
    temp4 = -1 * np.multiply.outer(data_list, xi_IRrs_prime2)
    temp5 = temp4.transpose()
    b_vec = (b1 * np.multiply(precision, (temp1 + temp2 + temp3 + temp4 + temp5))).sum()
    
    temp6 = b1 * np.multiply.outer(xi_IRrs, xi_IRrs_prime)
    temp7 = temp6.transpose()
    temp8 = -1 * np.multiply.outer(data_list, xi_IRrs_prime)
    temp9 = temp8.transpose()
    c_vec = (b1 * np.multiply(precision, (temp6 + temp7 + temp8 + temp9))).sum()
    
    return ((-1 * b_vec + np.sqrt(b_vec**2. - 4 * a_vec * c_vec)) / (2 * a_vec))
    

def marginal_linear_bias(alpha = 1.0):
    helper_object = BAO_scale_fitting_helper.Info(alpha)
        
    helper_object.calc_covariance_matrix()
    
    #difference = helper_object.covariance_matrix - helper_object.covariance_matrix_old
        
    helper_object.calc_CF()
    
    data_list = helper_object.get_data()
    
    covariance_matrix = helper_object.get_covariance_matrix()
    
    xi_IRrs = helper_object.templates()
    
    xi_IRrs_prime = helper_object.templates_deriv()
    
    xi_IRrs_prime2 = helper_object.templates_deriv2()
    
    xi_IRrs_prime3 = helper_object.templates_deriv3()
    
    '''
    helper1 = BAO_scale_fitting_helper.Info(0.998)
    helper2 = BAO_scale_fitting_helper.Info(1.002)
    helper3 = BAO_scale_fitting_helper.Info(1.)
    
    prime1 = helper1.templates_deriv2()
    prime2 = helper2.templates_deriv2()
    prime3 = helper3.templates_deriv3()
    #print(prime1)
    #print(prime2)
    temp_prime = (prime1 - prime2) / (1.002 - 0.998)
    temp_diff = prime3 - temp_prime
    #print(prime3)
    #print(temp_prime)
    #print(temp_diff)
    #print(temp_diff / prime3)
    '''
    
    precision = np.linalg.inv(covariance_matrix)
    
    timeit
    start = time.time()
    
    temp = 0.5 * np.multiply.outer(xi_IRrs, xi_IRrs_prime2)
    temp1 = temp.transpose()
    temp2 = np.multiply.outer(xi_IRrs_prime, xi_IRrs_prime)
    a_a = (0.5 * np.multiply(precision, (temp + temp1 + temp2))).sum()
    
    temp = np.multiply.outer(xi_IRrs, xi_IRrs_prime)
    temp1 = temp.transpose()
    b_a = (0.5 * np.multiply(precision, (temp + temp1))).sum()
    
    temp = np.multiply.outer(xi_IRrs, xi_IRrs)
    c_a = (0.5 * np.multiply(precision, temp)).sum()
    
    temp = 1.5 * np.multiply.outer(xi_IRrs_prime, xi_IRrs_prime2)
    temp1 = temp.transpose()
    temp2 = 0.5 * np.multiply.outer(xi_IRrs_prime3, xi_IRrs)
    temp3 = temp2.transpose()
    a_da = (0.5 * np.multiply(precision, (temp + temp1 + temp2 + temp3))).sum()
    
    temp = np.multiply.outer(xi_IRrs, xi_IRrs_prime2)
    temp1 = temp.transpose()
    temp2 = 2 * np.multiply.outer(xi_IRrs_prime, xi_IRrs_prime)
    b_da = (0.5 * np.multiply(precision, (temp + temp1 + temp2))).sum()
    
    temp = np.multiply.outer(xi_IRrs, xi_IRrs_prime)
    temp1 = temp.transpose()
    c_da = (0.5 * np.multiply(precision, (temp + temp1))).sum()
    
    temp = -0.5 * np.multiply.outer(data_list, xi_IRrs_prime2)
    temp1 = temp.transpose()
    temp2 = 0.0 #0.5 * np.multiply.outer(xi_IRrs_prime2, G)
    temp3 = 0.0 #temp2.transpose()
    a_b = (-0.5 * np.multiply(precision, (temp + temp1 + temp2 + temp3))).sum()
    
    temp = -1 * np.multiply.outer(data_list, xi_IRrs_prime)
    temp1 = temp.transpose()
    temp2 = 0.0 #np.multiply.outer(xi_IRrs_prime, G)
    temp3 = 0.0 #temp2.transpose()
    b_b = (-0.5 * np.multiply(precision, (temp + temp1 + temp2 + temp3))).sum()
    
    temp = -1 * np.multiply.outer(data_list, xi_IRrs)
    temp1 = temp.transpose()
    temp2 = 0.0 #np.multiply.outer(G, xi_IRrs)
    temp3 = 0.0 #temp2.transpose()
    c_b = (-0.5 * np.multiply(precision, (temp + temp1 + temp2 + temp3))).sum()
    
    temp = - 0.5 * np.multiply.outer(data_list, xi_IRrs_prime3)
    temp1 = temp.transpose()
    temp2 = 0.0 #0.5 * (np.multiply.outer(xi_IRrs_prime3 , G))
    temp3 = 0.0 #temp1.transpose()
    a_db = (-0.5 * np.multiply(precision, (temp + temp1 + temp2 + temp3))).sum()
    
    temp = np.multiply.outer(data_list, xi_IRrs_prime2)
    temp1 = temp.transpose()
    b_db = (0.5 * np.multiply(precision, (temp + temp1))).sum()
    
    temp = np.multiply.outer(data_list, xi_IRrs_prime)
    temp1 = temp.transpose()
    c_db = (0.5 * np.multiply(precision, (temp + temp1))).sum()

    #print('Vectorize: ', time.time() - start)
    #The vectorizable equations are 75 times faster than the for-loop equations
    A = - 0.5 * a_da / c_a + 0.5 * b_a * b_da / c_a**2 - 0.25 * c_da * (2 * b_a**2 / c_a**3 - 2 * a_a / c_a**2) \
           + 0.5 * a_b * c_db / c_a + 0.5 * b_b * b_db / c_a - 0.5 * c_b * b_a * b_db / c_a**2 + 0.25 * c_b * c_db\
           * (2 * b_a**2 / c_a**3 - 2 * a_a / c_a**2) - 0.25 * c_da * (2 * a_b * c_b + b_b**2) / c_a**2 \
           - 0.5 * b_b * c_b * b_da / c_a**2 + b_b * c_b * b_a * c_da / c_a**3 - 0.25 * c_b**2 * a_da / c_a**2 \
           + 0.5 * c_b**2 * b_a * b_da / c_a**3 - 0.125 * c_b**2 * c_da * (6 * b_a**2 / c_a**4 - 4 * a_a / c_a**3)\
           - 0.5 * b_a * b_b * c_db / c_a**2 + 0.5 * c_b * a_db / c_a
    
    B = .5 * b_b * c_db / c_a + .5 * c_b * b_db / c_a - .5 * c_b * b_a * c_db / c_a**2 \
            - .5 * b_b * c_b * c_da / c_a**2 - .25 * c_b**2 * b_da / c_a**2 + .5 * c_b**2 * b_a * c_da / c_a**3 \
            - 0.5 * b_da / c_a + 0.5 * b_a * c_da / c_a**2

    C = .5 * c_b * c_db / c_a - .25 * c_b**2 * c_da / c_a**2 - 0.5 * c_da / c_a
    
    return (-B - np.sqrt(B**2 - 4 * A * C)) / (2 * A)

#marginal_linear_bias()
def delta_alpha(data_list, xi_IRrs, xi_IRrs_prime, xi_IRrs_prime2, xi_IRrs_prime3, precision):
    temp = 0.5 * np.multiply.outer(xi_IRrs, xi_IRrs_prime2)
    temp1 = temp.transpose()
    temp2 = np.multiply.outer(xi_IRrs_prime, xi_IRrs_prime)
    a_a = (0.5 * np.multiply(precision, (temp + temp1 + temp2))).sum()
    
    temp = np.multiply.outer(xi_IRrs, xi_IRrs_prime)
    temp1 = temp.transpose()
    b_a = (0.5 * np.multiply(precision, (temp + temp1))).sum()
    
    temp = np.multiply.outer(xi_IRrs, xi_IRrs)
    c_a = (0.5 * np.multiply(precision, temp)).sum()
    
    temp = 1.5 * np.multiply.outer(xi_IRrs_prime, xi_IRrs_prime2)
    temp1 = temp.transpose()
    temp2 = 0.5 * np.multiply.outer(xi_IRrs_prime3, xi_IRrs)
    temp3 = temp2.transpose()
    a_da = (0.5 * np.multiply(precision, (temp + temp1 + temp2 + temp3))).sum()
    
    temp = np.multiply.outer(xi_IRrs, xi_IRrs_prime2)
    temp1 = temp.transpose()
    temp2 = 2 * np.multiply.outer(xi_IRrs_prime, xi_IRrs_prime)
    b_da = (0.5 * np.multiply(precision, (temp + temp1 + temp2))).sum()
    
    temp = np.multiply.outer(xi_IRrs, xi_IRrs_prime)
    temp1 = temp.transpose()
    c_da = (0.5 * np.multiply(precision, (temp + temp1))).sum()
    
    temp = -0.5 * np.multiply.outer(data_list, xi_IRrs_prime2)
    temp1 = temp.transpose()
    temp2 = 0.0 #0.5 * np.multiply.outer(xi_IRrs_prime2, G)
    temp3 = 0.0 #temp2.transpose()
    a_b = (-0.5 * np.multiply(precision, (temp + temp1 + temp2 + temp3))).sum()
    
    temp = -1 * np.multiply.outer(data_list, xi_IRrs_prime)
    temp1 = temp.transpose()
    temp2 = 0.0 #np.multiply.outer(xi_IRrs_prime, G)
    temp3 = 0.0 #temp2.transpose()
    b_b = (-0.5 * np.multiply(precision, (temp + temp1 + temp2 + temp3))).sum()
    
    temp = -1 * np.multiply.outer(data_list, xi_IRrs)
    temp1 = temp.transpose()
    temp2 = 0.0 #np.multiply.outer(G, xi_IRrs)
    temp3 = 0.0 #temp2.transpose()
    c_b = (-0.5 * np.multiply(precision, (temp + temp1 + temp2 + temp3))).sum()
    
    temp = - 0.5 * np.multiply.outer(data_list, xi_IRrs_prime3)
    temp1 = temp.transpose()
    temp2 = 0.0 #0.5 * (np.multiply.outer(xi_IRrs_prime3 , G))
    temp3 = 0.0 #temp1.transpose()
    a_db = (-0.5 * np.multiply(precision, (temp + temp1 + temp2 + temp3))).sum()
    
    temp = np.multiply.outer(data_list, xi_IRrs_prime2)
    temp1 = temp.transpose()
    b_db = (0.5 * np.multiply(precision, (temp + temp1))).sum()
    
    temp = np.multiply.outer(data_list, xi_IRrs_prime)
    temp1 = temp.transpose()
    c_db = (0.5 * np.multiply(precision, (temp + temp1))).sum()

    #print('Vectorize: ', time.time() - start)
    #The vectorizable equations are 75 times faster than the for-loop equations
    A = - 0.5 * a_da / c_a + 0.5 * b_a * b_da / c_a**2 - 0.25 * c_da * (2 * b_a**2 / c_a**3 - 2 * a_a / c_a**2) \
           + 0.5 * a_b * c_db / c_a + 0.5 * b_b * b_db / c_a - 0.5 * c_b * b_a * b_db / c_a**2 + 0.25 * c_b * c_db \
           * (2 * b_a**2 / c_a**3 - 2 * a_a / c_a**2) - 0.25 * c_da * (2 * a_b * c_b + b_b**2) / c_a**2 \
           - 0.5 * b_b * c_b * b_da / c_a**2 + b_b * c_b * b_a * c_da / c_a**3 - 0.25 * c_b**2 * a_da / c_a**2 \
           + 0.5 * c_b**2 * b_a * b_da / c_a**3 - 0.125 * c_b**2 * c_da * (6 * b_a**2 / c_a**4 - 4 * a_a / c_a**3) \
           - 0.5 * b_a * b_b * c_db / c_a**2 + 0.5 * c_b * a_db / c_a
    
    B = .5 * b_b * c_db / c_a + .5 * c_b * b_db / c_a - .5 * c_b * b_a * c_db / c_a**2 \
            - .5 * b_b * c_b * c_da / c_a**2 - .25 * c_b**2 * b_da / c_a**2 + .5 * c_b**2 * b_a * c_da / c_a**3 \
            - 0.5 * b_da / c_a + 0.5 * b_a * c_da / c_a**2

    C = .5 * c_b * c_db / c_a - .25 * c_b**2 * c_da / c_a**2 - 0.5 * c_da / c_a
    
    value = (-B - np.sqrt(B**2 - 4 * A * C)) / (2 * A)
    return value


alphas = np.linspace(0.96, 1.035, 100)
likelihood = np.zeros(100)

helper_object = BAO_scale_fitting_helper.Info(1.0)
helper_object.calc_covariance_matrix()
helper_object.calc_CF()
data_list = helper_object.get_data()
covariance_matrix = helper_object.get_covariance_matrix()
xi_IRrs = helper_object.templates()
xi_IRrs_prime = helper_object.templates_deriv()
xi_IRrs_prime2 = helper_object.templates_deriv2()
xi_IRrs_prime3 = helper_object.templates_deriv3()
precision = np.linalg.inv(covariance_matrix)
b1 = helper_object.get_biases()

xi_IRrs_splined = Spline(helper_object.r, xi_IRrs)

#Below is the plot the corvariance matrices
plt.imshow(covariance_matrix)
plt.colorbar()
plt.xlabel('Row Index')
plt.ylabel('Column Index')
plt.title('Covariance Matrix')
plt.show()

correlation_matrix = np.zeros_like(covariance_matrix)
fraction_error_matrix = np.zeros_like(covariance_matrix)

for i in range(np.shape(covariance_matrix)[0]):
    for j in range(np.shape(covariance_matrix)[0]):
        correlation_matrix[i, j] = covariance_matrix[i,j]/(covariance_matrix[i,i]*covariance_matrix[j,j])**0.5
        fraction_error_matrix[i, j] = covariance_matrix[i,j]**0.5 / (data_list[i] * data_list[j])**0.5
    
plt.imshow(correlation_matrix)
plt.colorbar()
plt.xlabel('Row Index')
plt.ylabel('Column Index')
plt.title('Correlation Matrix')
plt.show()


fig, ax = plt.subplots()
ax.errorbar(helper_object.r, xi_IRrs, yerr=np.diag(covariance_matrix)**.5)
plt.xlabel('r (Mpc)')
plt.ylabel('Correlation Function')
plt.title('Correlation Function With Error Bars')
plt.show()
print(np.diag(covariance_matrix)**0.5/data_list)

'''
fig, ax = plt.subplots()
ax.errorbar(helper_object.r, xi_IRrs, yerr=xi_IRrs_std)

ax.set_xlabel('r')
ax.set_ylabel('xi_IRrs')
ax.set_title('Correlation Function with error bars')


plt.show()


for x, a in enumerate(alphas):
    for l, l_ in enumerate(helper_object.r):
        for m, m_ in enumerate(helper_object.r):
            likelihood[x] += -0.5 * precision[l][m] * (data_list[l] - b1 * xi_IRrs_splined(l_ / a)) \
                * (data_list[m] - b1 * xi_IRrs_splined(m_ / a))


likelihood_taylor = np.zeros(100)
for x, a in enumerate(alphas):
    model = b1 * (xi_IRrs + xi_IRrs_prime * (a - 1) + xi_IRrs_prime2 * (a - 1)**2 + 1./6 * xi_IRrs_prime3 \
                  * (a - 1)**3.)
    for l in range(30):
        for m in range(30):
            likelihood_taylor[x] += -0.5 * precision[l][m] * (data_list[l] - model[l]) * (data_list[m] \
                                                                                          - model[m])

plt.figure()
plt.plot(alphas, likelihood, label=r'$\mathcal{B}$ * $P_{IR}$(k/alpha)')
plt.plot(alphas, likelihood_taylor, label='Taylor Expansion')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'ln $\mathcal{L}$')
plt.title(r'ln $\mathcal{L}$ vs $\alpha$')
plt.grid()
plt.show()
'''

'''
alpha = 0.99
helper_object = BAO_scale_fitting_helper.Info(a)
helper_object.calc_covariance_matrix()
helper_object.calc_CF()
data_list = helper_object.get_data()
covariance_matrix = helper_object.get_covariance_matrix()
xi_IRrs = helper_object.templates()
xi_IRrs_prime = helper_object.templates_deriv()
xi_IRrs_prime2 = helper_object.templates_deriv2()
xi_IRrs_prime3 = helper_object.templates_deriv3()

model_spline = b1 * xi_IRrs_splined(helper_object.r / alpha)
model_taylor = b1 * (xi_IRrs + xi_IRrs_prime * (alpha - 1) + xi_IRrs_prime2 * (alpha - 1)**2 + 1./6 \
                     * xi_IRrs_prime3 * (alpha - 1)**3.)
plt.figure()
plt.plot(model_spline)
plt.plot(model_taylor)
plt.grid()
plt.show()
'''