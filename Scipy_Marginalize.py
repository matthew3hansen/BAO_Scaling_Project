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
import BAO_scale_fitting_helper
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
import time
import random

alpha = 1.0

helper_object = BAO_scale_fitting_helper.Info(alpha)
        
helper_object.calc_covariance_matrix()

#difference = helper_object.covariance_matrix - helper_object.covariance_matrix_old
    
helper_object.calc_CF()

data_list = helper_object.get_data()

covariance_matrix = helper_object.get_covariance_matrix()


xi_IRrs = helper_object.templates()

precision = np.linalg.inv(covariance_matrix)
b1 = np.sqrt(helper_object.get_biases())
#print(b1)

r = np.linspace(30,180,31)
rbin = 0.5 * (r[1:]+r[:-1])

xi_IRrs_interpolated = Spline(rbin, xi_IRrs)

precision_matrix = np.linalg.inv(covariance_matrix)

def chisq(x):
    start = time.time()
    rs = rbin
    alpha = x
    #b1 = x[1]
    #b1 = 2.46
    model = xi_IRrs_interpolated(alpha * rs)
    
    a = 0.5 * np.multiply(precision, np.multiply.outer(model, model)).sum()
    #print(a)
    b = 0.5 * np.multiply(precision, (np.multiply.outer(model, data_list) + np.multiply.outer(data_list, model))).sum()
    #print(b)
    c = -0.5 * (np.multiply(precision, np.multiply.outer(data_list, data_list))).sum()
    #print(c)
    prob = np.sqrt(np.pi/a) * np.exp(b**2./(4*a) + c)
    #print(prob)
    loglike = -np.log(prob)
    #print(time.time() - start)
    return loglike
    '''
    model = xi_IRrs_interpolated(alpha * rs)
    a = 0.5 * np.dot(model,np.dot(precision, model))
    b = 0.5 * (np.dot(model, np.dot(precision_matrix, data_list)) + np.dot(data_list, np.dot(precision_matrix, model)))
    c = -0.5 * np.dot(data_list, np.dot(precision_matrix, data_list))
    prob = np.sqrt(np.pi/a) * np.exp(b**2./(4*a) + c)
    print(prob)
    loglike = -np.log(prob)
    #print(prob)
    #model = (xi_IRrs_interpolated(alpha * rs) * b1 ** 2 )
    #chisq = np.dot(np.dot((model - data_list),np.linalg.inv(covariance_matrix)), model-data_list)
    print(time.time() - start)
    return loglike
    '''
        
from scipy.optimize import minimize

def time_():
    #start = time.time()
    return minimize(chisq, (0.97))
    #return time.time() - start

#print(time_())

'''
times = np.zeros(10000)
x_s = np.zeros(10000)
for x in range(10000):
    start = time.time()
    obj = time_()
    x_s[x] = obj['x']
    times[x] = time.time() - start
'''

# min_obj['x'] gives us what we want
# alpha 1.00333 vs 1.0
# b1 2.4108 vs 2.4249
# b2 0.781 vs 1.024
# bs -5.88 vs -0.81
# b3nl 3.536 vs 1.424