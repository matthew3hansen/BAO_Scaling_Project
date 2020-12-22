#Code to do 2PCF.
#Begun by ZS 24 February 2020 around 3 pm EST.
#This version sent to Matt at 3.12 pm EST on Monday 24 Feb.
#Not super efficient---we could be histogramming on the radius^2, which would be better to avoid the sqrt which is slow.
#also---need a way to avg the histograms. 

#------------------------------------------------------------------
#------------------------------------------------------------------
#Modules.
#------------------------------------------------------------------
import numpy as np
import scipy as sp
import sys
#plotting
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
plt.matplotlib.use("TkAgg")
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
#for getting latex characters in plot titles
from matplotlib import rc
rc('font',size='32',family='serif')
plt.rcParams['pdf.fonttype'] = 42
from pylab import rcParams
rcParams['figure.figsize'] = 12, 10
#plotting a density plot (2-d)
from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from pylab import *
import matplotlib.colors
#interpolation
from scipy import interpolate
from scipy.interpolate import interp1d
#sbfs
from scipy.special import spherical_jn
#quad integration
from scipy.integrate import quad
from scipy.integrate import romb
from scipy.integrate import romberg
from scipy.interpolate import CubicSpline
#high precision quad integration
import mpmath as mp

#------------------------------------------------------------------
#Definitions.
#------------------------------------------------------------------
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

#------------------------------------------------------------------
#Step 1: throw coordinates
#------------------------------------------------------------------
#Let's work in 3-D.

num_points = 5000
x_vec = np.random.random_sample(num_points)
y_vec = np.random.random_sample(num_points)
z_vec = np.random.random_sample(num_points)

#could even rescale to some other box size if I like, how about 1000?
rescale = 100

x_vec *= rescale
y_vec *= rescale
z_vec *= rescale

nbins = 10000

plt.ion()
for i in range(0, num_points):
    #get relative coords.
    x_rel, y_rel, z_rel = x_vec - x_vec[i], y_vec - y_vec[i], z_vec - z_vec[i]
    rad_sq_vec = x_rel*x_rel + y_rel*y_rel + z_rel*z_rel
    #rad_vec = np.sqrt(rad_sq_vec)
    #form local 2PCF estimate about i^th point.
    hist, bin_edges = np.histogram(rad_sq_vec, bins = nbins)
    if (i==0):
        bin_centers = 0.5*(bin_edges[1:] + bin_edges[0:nbins])
        hist_cum = hist
    #print ("i^th point is =", i, "histogram is", hist)
    #plt.plot(bin_centers, hist)
    if (i>0):
        hist_cum += hist

plt.plot(bin_centers, hist_cum)
plt.xlabel("separation")
plt.ylabel("count")
plt.title("Histogram about each point")
plt.show(block=True)

#now average hist_cum down.
hist_cum = hist_cum / num_points


plt.plot(bin_centers, hist_cum)
plt.xlabel("separation")
plt.ylabel("count")
plt.title("Overall averaged histogram")
plt.show(block=True)


dist_sq = np.zeros((num_points, num_points))

#now do this in a triangular way!
for i in range(0, num_points):
    #print ("i = ", i)
    for j in range(i + 1, num_points):
        #print ("j = ", j)
        dist_sq[i, j] = (x_vec[i] - x_vec[j])*(x_vec[i] - x_vec[j]) + (y_vec[i] - y_vec[j])*(y_vec[i] - y_vec[j]) + (z_vec[i] - z_vec[j])*(z_vec[i] - z_vec[j])


#now reshape and then histogram.
dist_sq_1d = dist_sq.reshape(num_points*num_points)

hist_on_dist_sq, bin_edges = np.histogram(dist_sq_1d, bins = nbins)