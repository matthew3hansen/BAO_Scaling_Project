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
from matplotlib.ticker import FormatStrFormatter


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
    script_b = helper_object.get_biases()
    #start = time.time()
    
    temp = 3 * script_b * np.multiply.outer(xi_IRrs_prime, xi_IRrs_prime2)
    temp1 = temp.transpose()
    temp2 = script_b * np.multiply.outer(xi_IRrs_prime3, xi_IRrs)
    temp3 = temp2.transpose()
    temp4 = -1 * np.multiply.outer(xi_IRrs_prime3, data_list)
    temp5 = temp4.transpose()
    a_vec = (0.5 * script_b * np.multiply(precision, (temp + temp1 + temp2 + temp3 + temp4 + temp5))).sum()
    
    temp1 = script_b * np.multiply.outer(xi_IRrs, xi_IRrs_prime2)
    temp2 = temp1.transpose()
    temp3 = 2 * script_b * np.multiply.outer(xi_IRrs_prime, xi_IRrs_prime)
    temp4 = -1 * np.multiply.outer(data_list, xi_IRrs_prime2)
    temp5 = temp4.transpose()
    b_vec = (script_b * np.multiply(precision, (temp1 + temp2 + temp3 + temp4 + temp5))).sum()
    
    temp6 = script_b * np.multiply.outer(xi_IRrs, xi_IRrs_prime)
    temp7 = temp6.transpose()
    temp8 = -1 * np.multiply.outer(data_list, xi_IRrs_prime)
    temp9 = temp8.transpose()
    c_vec = (script_b * np.multiply(precision, (temp6 + temp7 + temp8 + temp9))).sum()
    
    return ((-1 * b_vec + np.sqrt(b_vec**2. - 4 * a_vec * c_vec)) / (2 * a_vec))


def fixed_linear_bias_second_order(alpha = 1.0):
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
    script_b = helper_object.get_biases()
    #start = time.time()
    
    #temp = 3 * script_b * np.multiply.outer(xi_IRrs_prime, xi_IRrs_prime2)
    #temp1 = temp.transpose()
    #a_vec = (0.5 * 0.5 * script_b * np.multiply(precision, (temp + temp1))).sum()
    
    temp1 = script_b * np.multiply.outer(xi_IRrs, xi_IRrs_prime2)
    temp2 = temp1.transpose()
    temp3 = 2 * script_b * np.multiply.outer(xi_IRrs_prime, xi_IRrs_prime)
    temp4 = -1 * np.multiply.outer(data_list, xi_IRrs_prime2)
    temp5 = temp4.transpose()
    b_vec = (script_b * np.multiply(precision, (temp1 + temp2 + temp3 + temp4 + temp5))).sum()
    
    temp6 = script_b * np.multiply.outer(xi_IRrs, xi_IRrs_prime)
    temp7 = temp6.transpose()
    temp8 = -1 * np.multiply.outer(data_list, xi_IRrs_prime)
    temp9 = temp8.transpose()
    c_vec = (script_b * np.multiply(precision, (temp6 + temp7 + temp8 + temp9))).sum()
    
    return -c_vec / b_vec
    
    #return ((-1 * b_vec + np.sqrt(b_vec**2. - 4 * a_vec * c_vec)) / (2 * a_vec))

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

def marginal_linear_bias_second_order(alpha = 1.0):
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
    temp2 = 0#0.5 * np.multiply.outer(xi_IRrs_prime3, xi_IRrs)
    temp3 = 0#temp2.transpose()
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
    
    temp = 0#- 0.5 * np.multiply.outer(data_list, xi_IRrs_prime3)
    temp1 = 0#temp.transpose()
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


helper_object = BAO_scale_fitting_helper.Info(1.077)
helper_object.calc_covariance_matrix()
helper_object.calc_CF()
data_list = helper_object.get_data()
covariance_matrix = helper_object.get_covariance_matrix()
xi_IRrs = helper_object.templates()
xi_IRrs_prime = helper_object.templates_deriv()
xi_IRrs_prime2 = helper_object.templates_deriv2()
xi_IRrs_prime3 = helper_object.templates_deriv3()
precision = np.linalg.inv(covariance_matrix)
script_b = helper_object.get_biases()
xi_IRrs_splined = Spline(helper_object.r, xi_IRrs)
spline_arr = np.zeros_like(helper_object.r)


def scaling_plot():
    plt.plot(helper_object.r, (helper_object.r)**2 * script_b * xi_IRrs_splined(helper_object.r), label=r'$\alpha$ = 1', linewidth=3.0)
    plt.plot(helper_object.r, (helper_object.r * 1.1)**2 * script_b * xi_IRrs_splined(1.1 * helper_object.r), 'm--', label=r'$\alpha$ = 1.1', linewidth=3.0)
    plt.plot(helper_object.r, (helper_object.r * 0.9)**2 * script_b * xi_IRrs_splined(0.9 * helper_object.r), 'r.', label=r'$\alpha$ = 0.9', linewidth=3.0)
    #plt.yscale('log')
    plt.xlabel(r'r [$h^{-1}$ Mpc]', fontsize=18)
    plt.ylabel(r'$(\alpha r)^2 \times b^2 \xi$($\alpha$r) [$h^{-2} Mpc^2$]', fontsize=16)
    plt.title(r'Correlation Function for Different $\alpha$', fontsize=18)
    plt.grid()
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    plt.xticks([30, 60, 90, 120, 150, 180])
    plt.xlim(30, 180)
    plt.legend()
    plt.tight_layout()
    #plt.savefig('dilation_BAO_12-08-21.pdf')
    plt.show()  

#scaling_plot()

def covariance_plot():
    #Below is the plot the corvariance matrices
    plt.imshow(covariance_matrix, vmin=-1e-7, extent=[30, 180, 180, 30])
    plt.colorbar()
    plt.xlabel(r'r ($h^{-1}$ Mpc)', fontsize=20)
    plt.ylabel(r'r ($h^{-1}$ Mpc)', fontsize=20)
    plt.title('Covariance Matrix', fontsize=20)
    plt.xticks([30, 60, 90, 120, 150, 180])
    plt.yticks([30, 60, 90, 120, 150, 180])
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    plt.tight_layout()
    plt.savefig('Covariance_matrix_12-08-21.pdf')
    plt.show()
    
#covariance_plot()
    
correlation_matrix = np.zeros_like(covariance_matrix)
fraction_error_matrix = np.zeros_like(covariance_matrix)

for i in range(np.shape(covariance_matrix)[0]):
    for j in range(np.shape(covariance_matrix)[0]):
        correlation_matrix[i, j] = covariance_matrix[i,j]/(covariance_matrix[i,i]*covariance_matrix[j,j])**0.5
        #fraction_error_matrix[i, j] = covariance_matrix[i,j]**0.5 / (data_list[i] * data_list[j])**0.5

def correlation_plot():
    plt.imshow(correlation_matrix, extent=[30, 180, 180, 30])
    plt.colorbar()
    plt.xlabel(r'r ($h^{-1}$ Mpc)', fontsize=20)
    plt.ylabel(r'r ($h^{-1}$ Mpc)', fontsize=20)
    plt.title('Correlation Matrix', fontsize=20)
    plt.xticks([30, 60, 90, 120, 150, 180])
    plt.yticks([30, 60, 90, 120, 150, 180])
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    plt.tight_layout()
    plt.savefig('Correlation_matrix_12-08-21.pdf')
    plt.show()
    
#correlation_plot()

def error_plot():
    plt.errorbar(helper_object.r, (helper_object.r)**2 * script_b * xi_IRrs, yerr= (helper_object.r)**2 * np.diag(covariance_matrix)**.5, linewidth=3.0)
    plt.xlabel(r'r ($h^{-1}$ Mpc)', fontsize=15)
    plt.ylabel(r'$r^2 \times \mathcal{B} \xi$(r)', fontsize=15)
    plt.title('Correlation Function of Mock Data', fontsize=15)
    plt.grid()
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    plt.xticks([30, 60, 90, 120, 150, 180])
    plt.xlim(30, 180)
    plt.tight_layout()
    plt.savefig('correlation_mock_data_12-01-21.pdf')
    plt.show()
    
#error_plot()

def scaling_plot():
    plt.plot(helper_object.r, (helper_object.r)**2 * script_b * xi_IRrs_splined(helper_object.r), color='black', label=r'$\alpha$ = 1', linewidth=3.0)
    plt.plot(helper_object.r, (helper_object.r * 1.1)**2 * script_b * xi_IRrs_splined(1.1 * helper_object.r), 'b--', label=r'$\alpha$ = 1.1', linewidth=3.0)
    plt.plot(helper_object.r, (helper_object.r * 0.9)**2 * script_b * xi_IRrs_splined(0.9 * helper_object.r), '.', color='orange', label=r'$\alpha$ = 0.9', linewidth=3.0)
    #plt.yscale('log')
    plt.xlabel(r'r [$h^{-1}$ Mpc]', fontsize=18)
    plt.ylabel(r'$(\alpha r)^2 \times b^2 \xi$($\alpha$r) [$h^{-2} Mpc^2$]', fontsize=16)
    plt.title(r'Correlation Function for Different $\alpha$', fontsize=18)
    plt.grid()
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    plt.xticks([30, 60, 90, 120, 150, 180])
    plt.xlim(30, 180)
    plt.legend()
    plt.tight_layout()
    #plt.savefig('dilation_BAO_12-08-21.pdf')
    plt.show()  

#scaling_plot()

def likelihood_a_plot(alpha, data_list, xi_IRrs_splined):
    alphas = np.linspace(alpha-0.02, alpha+0.02, 100)
    likelihood = np.zeros(100)
    
    temp1 = script_b * np.multiply.outer(xi_IRrs, xi_IRrs_prime2)
    temp2 = temp1.transpose()
    temp3 = 2 * script_b * np.multiply.outer(xi_IRrs_prime, xi_IRrs_prime)
    temp4 = -1 * np.multiply.outer(data_list, xi_IRrs_prime2)
    temp5 = temp4.transpose()
    b_vec = (script_b * np.multiply(precision, (temp1 + temp2 + temp3 + temp4 + temp5))).sum()
    
    temp6 = script_b * np.multiply.outer(xi_IRrs, xi_IRrs_prime)
    temp7 = temp6.transpose()
    temp8 = -1 * np.multiply.outer(data_list, xi_IRrs_prime)
    temp9 = temp8.transpose()
    c_vec = (script_b * np.multiply(precision, (temp6 + temp7 + temp8 + temp9))).sum()
    
    delta_alpha = alphas - 1
    constant = -0.5 * b_vec * 0.07**2 - c_vec * 0.07
    #plt.plot(alphas, -1 * (0.5 * b_vec * delta_alpha**2 + c_vec * delta_alpha + constant))
    
    
    #likelihood_15 = 
    print(data_list)
    print(script_b * xi_IRrs_splined(1.07 * helper_object.r))
    
    for x, a in enumerate(alphas):
        for l, l_ in enumerate(helper_object.r):
            for m, m_ in enumerate(helper_object.r):
                likelihood[x] += -0.5 * precision[l][m] * (data_list[l] - script_b * xi_IRrs_splined(l_ * a)) \
                                                    * (data_list[m] - script_b * xi_IRrs_splined(m_ * a))
    likelihood_taylor = np.zeros(100)
    likelihood_taylor_second = np.zeros(100)
    for x, a in enumerate(alphas):
        model = script_b * (xi_IRrs + xi_IRrs_prime * (a - 1) + 1. / 2 * xi_IRrs_prime2 * (a - 1)**2 + 1./6 * xi_IRrs_prime3 * (a - 1)**3.)
        model_second = script_b * (xi_IRrs + xi_IRrs_prime * (a - 1) + 1. / 2 *  xi_IRrs_prime2 * (a - 1)**2)
        for l in range(30):
            for m in range(30):
                likelihood_taylor[x] += -0.5 * precision[l][m] * (data_list[l] - model[l]) * (data_list[m] - model[m])
                likelihood_taylor_second[x] += -0.5 * precision[l][m] * (data_list[l] - model_second[l]) * (data_list[m] - model_second[m])
    
    #plt.figure()
    plt.plot(alphas, likelihood, color='black', label=r'Standard Method', linewidth=3.0)
    plt.plot(alphas, likelihood_taylor, '--', label='Taylor Expansion', linewidth=3.0)
    plt.plot(alphas, likelihood_taylor_second, color='orange', linestyle='dotted', label='Taylor Expansion Second-Order', linewidth=3.0)
    plt.xlabel(r'$\alpha$', fontsize=20)
    plt.ylabel(r'ln $\mathcal{L}$', fontsize=20)
    plt.title(r'Likelihood with Fixed $\mathcal{B}$', fontsize=20)
    #plt.xlim(0.99, 1.03)
    #plt.ylim(-30, 1)
    #plt.xticks([0.98, 0.99, 1.0, 1.01, 1.02])
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for label in plt.gca().xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    for label in plt.gca().yaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    #plt.savefig('L_a_12-08-21.pdf')
    plt.show()
    print(alphas[np.argmax(likelihood_taylor)])
    print(alphas[np.argmax(likelihood_taylor_second)])
    
    
    
alphas = [1.08]
for x in alphas:
    helper_object = BAO_scale_fitting_helper.Info(x)
    helper_object.calc_covariance_matrix()
    helper_object.calc_CF()
    data_list = helper_object.get_data()
    covariance_matrix = helper_object.get_covariance_matrix()
    xi_IRrs = helper_object.templates()
    xi_IRrs_prime = helper_object.templates_deriv()
    xi_IRrs_prime2 = helper_object.templates_deriv2()
    xi_IRrs_prime3 = helper_object.templates_deriv3()
    precision = np.linalg.inv(covariance_matrix)
    script_b = helper_object.get_biases()
    
    xi_IRrs_splined = Spline(helper_object.r, xi_IRrs)
    
    likelihood_a_plot(x, data_list, xi_IRrs_splined)

def likelihood_b_plot():
    script_b = np.linspace(5.7, 6.06, 100)
    likelihood = np.zeros(100)
    
    for x, a in enumerate(script_b):
        for l, l_ in enumerate(helper_object.r):
            for m, m_ in enumerate(helper_object.r):
                likelihood[x] += -0.5 * precision[l][m] * (data_list[l] - a * xi_IRrs_splined(l_)) \
                                                    * (data_list[m] - a * xi_IRrs_splined(m_))
    plt.figure()
    plt.plot(script_b, likelihood, linewidth=3.0)
    plt.xlabel(r'$\mathcal{B}$', fontsize=20)
    plt.ylabel(r'ln $\mathcal{L}$', fontsize=20)
    plt.title(r'Likelihood with Fixed $\alpha$', fontsize=20)
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    plt.xticks([5.7, 5.79, 5.88, 5.97, 6.06])
    
    plt.xlim(5.7, 6.06)
    plt.ylim(-8.5, 0.1)
    plt.grid()
    plt.tight_layout()
    #plt.savefig('L_b_12-08-21.pdf')
    plt.show()
    
#likelihood_b_plot()

def likelihood_a_marg_b_plot():
    likelihood = np.zeros(100)
    alphas = np.linspace(0.99, 1.01, 100)
    
    for x, a in enumerate(alphas):
        model = xi_IRrs_splined(a * helper_object.r)
        model_taylor = (xi_IRrs + xi_IRrs_prime * (a - 1) + 1. / 2 * xi_IRrs_prime2 * (a - 1)**2 + 1./6 * xi_IRrs_prime3 * (a - 1)**3.)
        
        a = 0.5 * np.multiply(precision, np.multiply.outer(model, model)).sum()
        #print(a)
        b = 0.5 * np.multiply(precision, (np.multiply.outer(model, data_list) + np.multiply.outer(data_list, model))).sum()
        #print(b)
        c = -0.5 * (np.multiply(precision, np.multiply.outer(data_list, data_list))).sum()
        #print(c)
        prob = np.sqrt(np.pi/a) * np.exp(b**2./(4*a) + c)
        
        for l, l_ in enumerate(helper_object.r):
            for m, m_ in enumerate(helper_object.r):
                likelihood[x] += np.log(prob)
                
    likelihood -= np.max(likelihood)
    plt.figure()
    plt.plot(alphas, likelihood, linewidth=3.0)
    plt.xlabel(r'$\alpha$', fontsize=20)
    plt.ylabel(r'ln $\mathcal{L}$', fontsize=20)
    plt.title(r'Likelihood with Marginalized $\mathcal{B}$', fontsize=18)
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for label in plt.gca().xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    for label in plt.gca().yaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(3,4))
    plt.xlim(0.99, 1.01)
    plt.grid()
    plt.tight_layout()
    #plt.savefig('L_a_marg_b_12-08-21.pdf')
    plt.show()
    
#likelihood_a_marg_b_plot()