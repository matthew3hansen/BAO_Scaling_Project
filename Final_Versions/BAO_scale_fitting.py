'''
Coding up our solution to the BAO Fast Scaling problem. 
Created October 10, 2021
Author(s): Matt Hansen, Alex Krolewski
'''
import numpy as np
import scipy as sp
import BAO_scale_fitting_helper
import time

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

    precision = np.linalg.inv(covariance_matrix)
    b1 = helper_object.get_biases()
    start = time.time()
    
    temp = np.multiply.outer(xi_IRrs_prime, xi_IRrs_prime2)
    a_vec = (2 * b1**2 * np.multiply(precision, (temp + temp.transpose()))).sum()
    
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
    
    #print('vector: ', time.time() - start)
    #The vectorization is 473 times faster than the for loop without the .sum()
    #With the sum, the vectorization method is 100 times faster than the for loop
    #Please let me know if there is a better way then creating all the temp variables!
    '''
    start = time.time()
    a = 0
    b = 0
    c = 0
    for l in range(len(helper_object.r)):
        for m in range(len(helper_object.r)):
            a += 2 * b1**2 * precision[l][m] * (xi_IRrs_prime[l] * xi_IRrs_prime2[m] + xi_IRrs_prime[m] * xi_IRrs_prime2[l])
            
            b += b1 * precision[l][m] * (b1 * xi_IRrs[l] * xi_IRrs_prime2[m] + b1 * xi_IRrs_prime2[l] * xi_IRrs[m] \
                                         + 2 * b1 * xi_IRrs_prime[l] * xi_IRrs_prime[m] - data_list[l] * xi_IRrs_prime2[m] \
                                         - data_list[m] * xi_IRrs_prime2[l])
            c += b1 * precision[l][m] * (b1 * xi_IRrs[l] * xi_IRrs_prime[m] + b1 * xi_IRrs_prime[l] * xi_IRrs[m] \
                                         - data_list[l] * xi_IRrs_prime[m] - data_list[m] * xi_IRrs_prime[l])
    print('loop: ', time.time() - start)
    '''
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
    
    precision = np.linalg.inv(covariance_matrix)
    
    
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
    
    temp = np.multiply.outer(xi_IRrs_prime, xi_IRrs_prime2)
    temp1 = temp.transpose()
    a_da = (0.75 * np.multiply(precision, (temp + temp1))).sum()
    
    temp = np.multiply.outer(xi_IRrs, xi_IRrs_prime2)
    temp1 = temp.transpose()
    temp2 = 2 * np.multiply.outer(xi_IRrs_prime, xi_IRrs_prime)
    b_da = (0.5 * np.multiply(precision, (temp + temp1 + temp2))).sum()
    
    temp = np.multiply.outer(xi_IRrs, xi_IRrs_prime)
    temp1 = temp.transpose()
    c_da = (0.5 * np.multiply(precision, (temp + temp1))).sum()
    
    temp = np.multiply.outer(data_list, xi_IRrs_prime2)
    temp1 = temp.transpose()
    a_b = (0.25 * np.multiply(precision, (temp + temp1))).sum()
    
    temp = np.multiply.outer(data_list, xi_IRrs_prime)
    temp1 = temp.transpose()
    b_b = (0.5 * np.multiply(precision, (temp + temp1))).sum()
    
    temp = np.multiply.outer(data_list, xi_IRrs)
    temp1 = temp.transpose()
    c_b = (0.5 * np.multiply(precision, (temp + temp1))).sum()
    
    temp = np.multiply.outer(data_list, xi_IRrs_prime2)
    temp1 = temp.transpose()
    b_db = (0.5 * np.multiply(precision, (temp + temp1))).sum()
    
    temp = np.multiply.outer(data_list, xi_IRrs_prime)
    temp1 = temp.transpose()
    c_db = (0.5 * np.multiply(precision, (temp + temp1))).sum()

    #print('Vectorize: ', time.time() - start)
    #The vectorizable equations are 75 times faster than the for-loop equations
    
    beta = .5 * a_b * c_db / c_a + .5 * b_b * b_db / c_a - .5 * b_b * b_a * c_db / c_a**2 \
            - .5 * c_b * b_a * b_db / c_a**2 + .5 * c_b * (b_a**2 / c_a**3 - a_a / c_a**2) * c_db \
            - .25 * ((2 * a_b * c_b + b_b**2) / c_a**2) * c_da - .5 * b_b * c_b * b_da / c_a**2 \
            + b_b * c_b * b_a * c_da / c_a**3 - .25 * c_b**2 * a_da / c_a**2 \
            + .5 * c_b**2 * b_a * b_da / c_a**3 - .125 * c_b**2 * (6 * b_a**2 / c_a**4 - 4 * a_a / c_a**3) * c_da \
            - 0.5 * a_da / c_a + 0.5 * b_a * b_da / c_a**2 - 0.25 * ( 2 * b_a**2 / c_a**3 - 2 * a_a / c_a**2 ) * c_da

    gamma = .5 * b_b * c_db / c_a + .5 * c_b * b_db / c_a - .5 * c_b * b_a * c_db / c_a**2 \
            - .5 * b_b * c_b * c_da / c_a**2 - .25 * c_b**2 * b_da / c_a**2 + .5 * c_b**2 * b_a * c_da / c_a**3 \
            - 0.5 * b_da / c_a + 0.5 * b_a * c_da / c_a**2

    delta = .5 * c_b * c_db / c_a - .25 * c_b**2 * c_da / c_a**2 - 0.5 * c_da / c_a
    
    value = (-gamma - np.sqrt(gamma**2 - 4 * beta * delta)) / (2 * beta)
    
    return (time.time() - start)
    #return (-gamma - np.sqrt(gamma**2 - 4 * beta * delta)) / (2 * beta)

times = np.zeros(100)
alphas = np.linspace(0.9, 1.1, 100)
for i in range(100):
    times[i] = marginal_linear_bias(alphas[i])
    print('Time: ', times[i])
    print()

print(np.mean(times))