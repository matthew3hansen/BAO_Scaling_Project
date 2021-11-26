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

alpha = 1.05

helper_object = BAO_scale_fitting_helper.Info(alpha)
        
helper_object.calc_covariance_matrix()

#difference = helper_object.covariance_matrix - helper_object.covariance_matrix_old
    
helper_object.calc_CF()

data_list = helper_object.get_data()

covariance_matrix = helper_object.get_covariance_matrix()

xi_IRrs = helper_object.templates()

start = time.time()
precision = np.linalg.inv(covariance_matrix)
#print(time.time() - start)

b1 = np.sqrt(helper_object.get_biases())
#print(b1)

r = np.linspace(30,180,31)
rbin = 0.5 * (r[1:]+r[:-1])

xi_IRrs_interpolated = Spline(rbin, xi_IRrs)

counter = 0

def chisq(x):
    start = time.time()
    rs = rbin
    alpha = x[0]
    #b1 = x[1]
    model = (xi_IRrs_interpolated(alpha * rs) * b1** 2 )
    chisq = np.dot(np.dot((model - data_list), precision) , model-data_list)
    #print(time.time() - start)
    return chisq
		
from scipy.optimize import minimize

def time_():
    return minimize(chisq, (1.0))

#hello = time_()
#print(time_())

'''
times = np.zeros(10000)
for x in range(10000):
    start = time.time()
    time_()
    times[x] = time.time() - start
'''


