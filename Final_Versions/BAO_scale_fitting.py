'''
Coding up our solution to the BAO Fast Scaling problem. 
Created October 10, 2021
Author(s): Matt Hansen, Alex Krolewski
'''
import numpy as np
import BAO_scale_fitting_helper

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
    a = 0
    b = 0
    c = 0
    
    for l in range(len(helper_object.r)):
        for m in range(len(helper_object.r)):
            a += 2 * b1 * precision[l][m] * (xi_IRrs_prime[l] * xi_IRrs_prime2[m] + xi_IRrs_prime[m] * xi_IRrs_prime2[l])
            
            b += b1 * precision[l][m] * (b1 * xi_IRrs[l] * xi_IRrs_prime2[m] + b1 * xi_IRrs_prime2[l] * xi_IRrs[m] \
                                         + 2 * b1 * xi_IRrs_prime[l] * xi_IRrs_prime[m] - data_list[l] * xi_IRrs_prime2[m] \
                                         - data_list[m] * xi_IRrs_prime2[l])
            c += b1 * precision[l][m] * (b1 * xi_IRrs[l] * xi_IRrs_prime[m] + b1 * xi_IRrs_prime[l] * xi_IRrs[m] \
                                         - data_list[l] * xi_IRrs_prime[m] - data_list[m] * xi_IRrs_prime[l])
                
    return ((-1 * b + np.sqrt(b**2. - 4 * a * c)) / (2 * a))
    

def marginalizing_linear_bias(alpha = 1.0):
    helper_object = BAO_scale_fitting_helper.Info(alpha)
        
    helper_object.calc_covariance_matrix()
    
    #difference = helper_object.covariance_matrix - helper_object.covariance_matrix_old
        
    helper_object.calc_CF()
    
    data_list = helper_object.get_data()
    
    covariance_matrix = helper_object.get_covariance_matrix()
    
    xi_IRrs = helper_object.templates()
    
    xi_IRrs_prime = helper_object.templates_deriv()
    
    xi_IRrs_prime2 = helper_object.templates_deriv2()

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
    
    precision = np.linalg.inv(covariance_matrix)
    
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

    return (-gamma - np.sqrt(gamma**2 - 4 * beta * delta)) / (2 * beta) 