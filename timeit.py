import BAO_scale_fitting_helper
import numpy as np
import time
import timeit
import random
from matplotlib import pyplot as plt

alpha = 1.0 

helper_object = BAO_scale_fitting_helper.Info(alpha)
        
helper_object.calc_covariance_matrix()

#difference = helper_object.covariance_matrix - helper_object.covariance_matrix_old

helper_object.calc_CF()

data_list = helper_object.get_data()

covariance_matrix = helper_object.get_covariance_matrix()

#print(covariance_matrix == covariance_matrix.transpose())

xi_IRrs = helper_object.templates()

xi_IRrs_prime = helper_object.templates_deriv()

xi_IRrs_prime2 = helper_object.templates_deriv2()

xi_IRrs_prime3 = helper_object.templates_deriv3()

b1 = helper_object.get_biases()

G = helper_object.polynomial_terms()

precision = np.linalg.inv(covariance_matrix)

scriptB = np.linspace(5.8, 5.96, 100)
def likelihood_scriptB():
    likelihood = np.zeros(100)
    for x, b in enumerate(scriptB):
        model = b * xi_IRrs
        for l in range(30):
            for m in range(30):
                likelihood[x] += -0.5 * precision[l][m] * (data_list[l] - model[l]) * (data_list[m] - model[m])
    
    return likelihood

alphas = np.linspace(0.5, 1.5, 100)
def likelihood_alpha():
    likelihood = np.zeros(100)
    for x, a in enumerate(alphas):
        model = b1 * (xi_IRrs + xi_IRrs_prime * (a - 1) + xi_IRrs_prime2 * (a - 1)**2 + 1./6 * xi_IRrs_prime3 * (a - 1)**3.)
        for l in range(30):
            for m in range(30):
                likelihood[x] += -0.5 * precision[l][m] * (data_list[l] - model[l]) * (data_list[m] - model[m])
    
    return likelihood

likelihood = likelihood_scriptB()
plt.figure()
plt.plot(scriptB, likelihood)
plt.xlabel(r'$\mathcal{B}$')
plt.ylabel(r'ln $\mathcal{L}$')
plt.title(r'ln $\mathcal{L}$ vs $\mathcal{B}$')
plt.grid()
plt.show()

def marginalize_delta_alpha():
    '''
    alpha2 = 1.0 + random.uniform(-.1, .1)
    #print(alpha2)
    helper_object2 = BAO_scale_fitting_helper.Info(alpha2)
    helper_object2.calc_CF()
    data_list = helper_object2.get_data()
    '''
    
    start = time.time()
    temp = 0.5 * np.multiply.outer(xi_IRrs, xi_IRrs_prime2)
    temp1 = temp.transpose()
    temp2 = np.multiply.outer(xi_IRrs_prime, xi_IRrs_prime)
    
    a_a = 0.5*(np.multiply(precision, (temp + temp1 + temp2))).sum()
    #zack suggestion
    #bring out .5 in last line above and try np.sum.
    
    
    temp = np.multiply.outer(xi_IRrs, xi_IRrs_prime)
    temp1 = temp.transpose()
    b_a = 0.5*(np.multiply(precision, (temp + temp1))).sum()
    
    temp = np.multiply.outer(xi_IRrs, xi_IRrs)
    c_a = 0.5*(np.multiply(precision, temp)).sum()
    
    temp = 1.5 * np.multiply.outer(xi_IRrs_prime, xi_IRrs_prime2)
    temp1 = temp.transpose()
    temp2 = 0.5 * np.multiply.outer(xi_IRrs_prime3, xi_IRrs)
    temp3 = temp2.transpose()
    a_da = 0.5*(np.multiply(precision, (temp + temp1 + temp2 + temp3))).sum()
    
    temp = np.multiply.outer(xi_IRrs, xi_IRrs_prime2)
    temp1 = temp.transpose()
    temp2 = 2 * np.multiply.outer(xi_IRrs_prime, xi_IRrs_prime)
    b_da = 0.5*(np.multiply(precision, (temp + temp1 + temp2))).sum()
    
    temp = np.multiply.outer(xi_IRrs, xi_IRrs_prime)
    temp1 = temp.transpose()
    c_da = 0.5*(np.multiply(precision, (temp + temp1))).sum()
    
    temp = -0.5 * np.multiply.outer(data_list, xi_IRrs_prime2)
    temp1 = temp.transpose()
    temp2 = 0.0 #0.5 * np.multiply.outer(xi_IRrs_prime2, G)
    temp3 = 0.0 #temp2.transpose()
    a_b = -0.5*(np.multiply(precision, (temp + temp1 + temp2 + temp3))).sum()
    
    temp = -1 * np.multiply.outer(data_list, xi_IRrs_prime)
    temp1 = temp.transpose()
    temp2 = 0.0 #np.multiply.outer(xi_IRrs_prime, G)
    temp3 = 0.0 #temp2.transpose()
    b_b = -0.5*(np.multiply(precision, (temp + temp1 + temp2 + temp3))).sum()
    
    temp = -1 * np.multiply.outer(data_list, xi_IRrs)
    temp1 = temp.transpose()
    temp2 = 0.0 #np.multiply.outer(G, xi_IRrs)
    temp3 = 0.0 #temp2.transpose()
    c_b = -0.5*(np.multiply(precision, (temp + temp1 + temp2 + temp3))).sum()
    
    temp = - 0.5 * np.multiply.outer(data_list, xi_IRrs_prime3)
    temp1 = temp.transpose()
    temp2 = 0.0 #0.5 * (np.multiply.outer(xi_IRrs_prime3 , G))
    temp3 = 0.0 #temp1.transpose()
    a_db = -0.5*(np.multiply(precision, (temp + temp1 + temp2 + temp3))).sum()
    
    temp = np.multiply.outer(data_list, xi_IRrs_prime2)
    temp1 = temp.transpose()
    b_db = 0.5*(np.multiply(precision, (temp + temp1))).sum()
    
    temp = np.multiply.outer(data_list, xi_IRrs_prime)
    temp1 = temp.transpose()
    c_db = 0.5 * (np.multiply(precision, (temp + temp1))).sum()

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
    
    return (-B - np.sqrt(B**2 - 4 * A * C)) / (2 * A)


def linear_delta_alpha():
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