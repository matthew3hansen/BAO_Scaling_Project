import math
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import random
import matplotlib.pyplot as plt
from sympy import *
from scipy.stats import multivariate_normal


class XYZGrid: 
    def __init__(self, x, y, z=0, data_points=0):
        self.x = x
        self.y = y
        self.z = z
        if(z != 0):
            self.cells = x * y * z
            self.grid = [[[0 for i in range(self.y)] for j in range(self.x)] for _ in range(self.z)]
        else:
            self.cells = x * y
            self.grid = [[0 for i in range(self.y)] for j in range(self.x)]
        
        self.insert_data(data_points)

    def insert_data(self, data_points):
        #Choosing a random 2d cell to add 1 particle to.
        #nvals = 10000
        #could do something like: rvec, cvec = random.randint(0,9,nvals), same for cvec
        for i in range(0, data_points):
            x = random.randint(0,self.x - 1)
            y = random.randint(0,self.y - 1)
            if(self.z != 0):
                z = random.randint(0,self.z - 1)
                self.grid[z][x][y] += 1
            else:
                z = 0
                self.grid[x][y] += 1
        


    def distance_from_origin(self):
        #Set origin at grid[0][0][0]
        if(self.z != 0):
            radial_distance = [0 for i in range(((self.x - 1)**2 + (self.y - 1)**2 + (self.z - 1)**2) + 1)]
        else:
            radial_distance = [0 for i in range((self.x - 1)**2 + (self.y - 1)**2 + 1)]
        #can use np.zeros(number of zeros you want) and then you can even reshape if you want, 
        #say an arry of zeros. np.zeros(100).reshape(10,10) or just np.zeros((10,10))
        count = 0
        for i in range(0, self.x):
            for j in range(0, self.y):
                if(self.z != 0):
                    for k in range(0, self.z):
                        #number_at_radial_distance might be a better name.
                        if((i != 0) or (j != 0) or (k != 0)):
                            radial_distance[i**2 + j**2 + k**2] += self.grid[k][i][j]
                        else:
                            radial_distance[i**2 + j**2 + k**2] = None
                else:
                    #number_at_radial_distance might be a better name.
                    if((i != 0) or (j != 0)):
                        radial_distance[i**2 + j**2] += self.grid[i][j]
                    else:
                        radial_distance[i**2 + j**2] = None
        return radial_distance


    def distance_from_given_point(self, point):
        #Set origin at point
        if(self.z != 0):
            z = point // (self.x * self.y)

        if(self.z != 0):
            x = (point // (self.y)) % self.z
        else:
            x = point // (self.y)

        y = point % (self.y)
        
        radial_distance = [0 for i in range((self.x**2 + self.y**2 + self.z**2) + 1)]
        for i in range(0, self.x):
            for j in range(0, self.y):
                if(self.z != 0):
                    for k in range(0, self.z):
                        #Make sure it does not count itself
                        if((i != x) or (j != y) or (k != z)):
                            distance_val = (i - x)**2 + (j - y)**2 + (k - z)**2
                            radial_distance[distance_val] += self.grid[k][i][j]
                else:
                    #Make sure it does not count itself
                    if((i != x) or (j != y)):
                        distance_val = (i - x)**2 + (j - y)**2
                        radial_distance[distance_val] += self.grid[i][j]

        return radial_distance

def frequency_of_radial_dist(radial_distance_FO, radius):
        frequency_of_entries = 0
        for i in range(0, radius):
                    frequency_of_entries += radial_distance_FO[i]
        return frequency_of_entries

    #Will calulcate the nummber of entries between two given radial distances (creating a shell)
def frequency_of_radial_shells(radial_distance_FO, short_radius, long_radius):
    frequency_of_entries = 0
    for i in range(short_radius, long_radius):
                frequency_of_entries += radial_distance_FO[i]
    return frequency_of_entries


def main():
    xyzgrid = XYZGrid(30, 30)
    xyzgrid.insert_data(100000)

    radial_distance_FO = xyzgrid.distance_from_origin()
    #create a list representing the x-coordinates, which will be the radial distance, will go 1 -> rows^2 + cols^2 inclusive
    #Set that x-coordinate equal to None so it doesn't show up in the plot
    if(xyzgrid.z != 0):
        x_coordinates_for_squared = [None for i in range((xyzgrid.x - 1)**2 + (xyzgrid.y - 1)**2 + (xyzgrid.z - 1)**2 + 1)]
    else:
        x_coordinates_for_squared = [None for i in range((xyzgrid.x - 1)**2 + (xyzgrid.y - 1)**2 + 1)]

    #Since not every available x-coordinate is possible for a data point to be, since only sums of int's squares are valid. 
    #e.g. 3 cannot be a valid x-coordinate
    if(xyzgrid.z != 0):
        for i in range(0,(((xyzgrid.x - 1) * (xyzgrid.y - 1)) * 2 + (xyzgrid.z - 1)**2) + 1):
            #Set a count to keep track if i has already been assigned its value, no need to check past its value!
            count = 0
            for j in range(0,xyzgrid.x):
                for k in range(0,xyzgrid.y):
                    for z in range(0, xyzgrid.z):
                        if(j != 0 and k !=0):
                            if(int(i / (j**2 + k**2 + z**2) == 1)):
                                x_coordinates_for_squared[i] = i
                                count = 1
                                break
                    if(count == 1):
                        break
                if(count == 1):
                        break
    else:
        for i in range(0, ((xyzgrid.x - 1) * (xyzgrid.y - 1) + 1)):
            #Set a count to keep track if i has already been assigned its value, no need to check past its value!
            count = 0
            for j in range(0,xyzgrid.x):
                for k in range(0,xyzgrid.y):
                    if(j != 0 and k !=0):
                        if(int(i / (j**2 + k**2) == 1)):
                            x_coordinates_for_squared[i] = i
                            count = 1
                            break
                if(count == 1):
                    break
    
    #scatter plot of radial distance from origin
    plt.scatter(x_coordinates_for_squared, radial_distance_FO)
    plt.xlabel("radial distance from origin (r^2)")
    plt.ylabel("number of entries at radial distance")
    plt.title("radial distance from origin")
    plt.show()
    
    
    nbins = 1000

    for l in range(0, xyzgrid.cells):
        number_in_each_index = []
        relative_square_dist = xyzgrid.distance_from_given_point(l)
        for i in range(0, len(relative_square_dist)):
            if (relative_square_dist[i] != 0):
                for _ in range(0, relative_square_dist[i]):
                    number_in_each_index.append(i)
        hist, bin_edges = np.histogram(number_in_each_index, bins = nbins)
        if (l == 0):
            bin_centers = 0.5*(bin_edges[1:] + bin_edges[0:nbins])
            hist_cum = hist
        if (l > 0):
            hist_cum += hist

    
    plt.plot(bin_centers, hist_cum)
    plt.xlabel("Separation (r^2)")
    plt.ylabel("Count")
    plt.title("Histogram about each point")
    plt.show(block=True)

    #now average hist_cum down.
    hist_cum = hist_cum / xyzgrid.cells

    plt.plot(bin_centers, hist_cum)
    plt.xlabel("Separation (r^2)")
    plt.ylabel("Count")
    plt.title("Overall averaged histogram")
    plt.show(block=True)
    '''
    levels = np.linspace(0, 10, 40)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    x = np.arange(0, xyzgrid.x, 1)
    y = np.arange(0, xyzgrid.y, 1)
    X, Y = np.meshgrid(x, y)

    # The wireframe
    #ax.plot_wireframe(X, Y, np.array(xyzgrid.grid), rstride= 1, cstride= 1, color='k', alpha=.8)
    # The heatmap
    ax.contourf(X, Y, np.array(xyzgrid.grid), zdir='z', levels=300, alpha=1)
    
    #Code to make a 3D scatter plot
    
    for i in range(xyzgrid.x):
        for j in range(xyzgrid.y):
            ax.scatter(i, j, xyzgrid.grid[i][j], c='k')
    

    ax.set_xlabel('X-Cell')
    ax.set_ylabel('Y-cell')
    ax.set_zlabel('# of Galaxies')
    ax.legend()
    ax.set_title("Frequency at each cell")
    ax.set_xlim3d(0, xyzgrid.x)
    ax.set_ylim3d(0, xyzgrid.y)
    ax.set_zlim3d(50, 150)

    plt.show()
'''

main()