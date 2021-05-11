'''
Coding up our solution to the BAO Fast Scaling problem. 
December 2020
Author(s): Matt Hansen, Alex Krolewski
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
import helper2

def run(alpha=1.):
	helper_object = helper2.Info(alpha)
	helper_object.calc_CF()

	#I don't know how the data vector will look in the end, but right now I'm thinking that it can be a list where each index represents a 
	#radial bin, and its corresponding value is the data at that radial bin
	data_list = helper_object.get_data()

	covariance_matrix = helper_object.get_covariance_matrix()

	xi_IRrs = helper_object.templates()

	xi_IRrs_prime = helper_object.templates_deriv()

	xi_IRrs_prime2 = helper_object.templates_deriv2()
	
	return helper_object.delta_alpha(data_list)