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
import j0j0
from mcfit import P2xi
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from fastpt import *


x_axis = [_ for _ in range(200)]

kh, z, pk = CAMB_General_Code.get_matter_spectrum()

#Made this function as a history to run all the graphs if need to be to check something
def old_graphs():
	plt.plot(kh, pk[0])
	plt.title('pk at z = 0 vs kh')
	plt.xlabel('kh')
	plt.ylabel('pk')
	plt.show()

	plt.plot(np.log(kh), pk[0])
	plt.title('pk at z = 0 vs log(kh)')
	plt.xlabel('log(kh)')
	plt.ylabel('pk')
	plt.show()

	plt.plot(kh, np.log(pk[0]))
	plt.xlabel("log(kh)")
	plt.ylabel("log(pk)")
	plt.title('log(pk) at z = 0')
	plt.show()

	dlogkh = np.gradient(kh)
	vector_want = 1 / (2 * math.pi**2 ) * (kh**2 * pk[0] * ((3 * special.spherical_jn(1, 8*kh)) / (8 * kh))**2) * dlogkh
	#Sigma_8^2
	print("My sigma: ", (np.sum(vector_want)))

	plt.plot(kh, np.log(kh**3 * pk[0] * np.exp(-kh**2)))
	plt.xlabel("log(kh)")
	plt.ylabel("log(kh**3 * pk * exp)")
	plt.title('log(kh**3 * pk * exp) at z=0')
	plt.show()

	#This should be the integral that you put in our paper, Zack
	r = np.arange(1,300)
	dlogkh = kh[1] - kh[0]
	xi_1_paper = [i for i in range(1, 300)]
	for i in range(len(r)):
		xi_1_paper[i] = r[i]**2 * np.sum(1 / (2 * math.pi**2) * kh**3 * special.spherical_jn(0, kh * r[i]) * pk[0]) * dlogkh 

	plt.plot(r, xi_1_paper)
	plt.xlabel("r")
	plt.ylabel("r[i]**2 * xi")
	plt.title('r[i]**2 * xi vs r')
	plt.show()


	#Below is attempting to find derivatives with formula form paper
	dxi_1_paper = [i for i in range(1, 300)]
	for i in range(len(r)):
		dxi_1_paper[i] = -r[i] * np.sum((kh**3 * dlogkh) / (2 * math.pi**2) * kh * special.spherical_jn(1, kh * r[i]) * pk[0] * np.exp(-kh**2))


	spl = Spline(r[1:], xi_1_paper[1:] / r[1:]**2.)

	x = spl(r * 1.001)
	y = spl(r * 0.999)

	dxi_1 = (x - y) / 0.002
	plt.figure()
	plt.plot(r, dxi_1_paper, '-g', label='paper')
	plt.plot(r, dxi_1, label='numerically')
	plt.xlabel("r")
	plt.ylabel("dxi")
	plt.title('dxi / d(alpha)|alpha = 1 {from paper}')
	plt.legend()
	plt.show()

	plt.plot(r, dxi_1)
	plt.title('dxi_1')
	plt.show()

	'''
	Below is the block of code that will be used to get xi_1
	All of these are derived in our paper
	'''
	r = np.linspace(1., 300., 102)
	xi_1 = np.zeros(len(r))
	dxi_1 = np.zeros(len(r))
	dlogkh = kh[1] - kh[0]

	#I have a question about below equation. Zack said that I need to replace every dkh with with kh dlog(kh)
	#Before I had kh**3 which was different than what the paper had, but I think it was correcting for the dlog(kh)
	#I changed it back to kh**2, pulled the dlog(kh) outside the sum, and left the k resulting in u-sub in the sum (at the end)
	for i in range(len(r)):
		xi_1[i] = r[i]**2 * np.sum(1 / (2 * math.pi**2) * kh**2 * special.spherical_jn(0, kh * r[i]) * pk[0] * kh) * dlogkh

	plt.plot(r, xi_1)
	plt.xlabel("r")
	plt.ylabel("r[i]**2 * xi")
	plt.title('r[i]**2 * xi vs r')
	plt.show()


	i = 0
	for r_val in r:
		dxi_1[i] = -r[i] * np.sum(kh**2  / (2 * math.pi**2) * kh * special.spherical_jn(1, kh * r_val) * pk[0] * np.exp(-kh**2) * kh) * dlogkh
		i += 1

	plt.plot(r, dxi_1)
	plt.xlabel("r")
	plt.ylabel("dxi_1")
	plt.title('dxi_1 vs r')
	plt.show()

	#I'm having a hard time trying to recover xi, its not really working out
	xi_1_recovered = np.zeros(len(r))
	for i in range(len(r)):
		xi_1_recovered[i] = (r[-1] - r[0]) / len(r) * np.sum(dxi_1[0:i])

	plt.plot(r, xi_1_recovered)
	plt.xlabel("r")
	plt.ylabel("xi_1_recovered")
	plt.title('xi_1_recovered vs r')
	plt.show();

	plt.plot(kh, pk[0])
	plt.xlabel("kh")
	plt.ylabel("pk")
	plt.title('pk vs kh')
	plt.show()

	'''
	I forget exactly where this code is from, but it reproduces the correct graph now.
	This section gives P(k)_linear and [P_{22} + P_{13}]
	'''
	from time import time

	import fastpt as fpt
	from fastpt import FASTPT

	#Version check
	print('This is FAST-PT version', fpt.__version__)

	# load the data file
	d=np.loadtxt('Pk_test.dat')
	# declare k and the power spectrum
	k=d[:,0]; P=d[:,1]

	# set the parameters for the power spectrum window and
	# Fourier coefficient window
	#P_window=np.array([.2,.2])
	C_window=.75
	#document this better in the user manual

	# padding length
	n_pad=int(0.5*len(k))
	to_do=['all']

	# initialize the FASTPT class
	# including extrapolation to higher and lower k
	# time the operation
	t1 = time()
	fpt = FASTPT(k,to_do=to_do,low_extrap=-5,high_extrap=3,n_pad=n_pad)
	t2 = time()

	# calculate 1loop SPT (and time the operation)
	P_spt = fpt.one_loop_dd(P,C_window=C_window)

	t3=time()
	print('initialization time for', to_do, "%10.3f" %(t2-t1), 's')
	print('one_loop_dd recurring time', "%10.3f" %(t3-t2), 's')

	#calculate tidal torque EE and BB P(k)
	P_IA_tt=fpt.IA_tt(P,C_window=C_window)
	P_IA_ta=fpt.IA_ta(P,C_window=C_window)
	P_IA_mix=fpt.IA_mix(P,C_window=C_window)
	P_RSD=fpt.RSD_components(P,1.0,C_window=C_window)
	P_kPol=fpt.kPol(P,C_window=C_window)
	P_OV=fpt.OV(P,C_window=C_window)
	sig4=fpt.sig4

	# make a plot of 1loop SPT results
	ax=plt.subplot(111)
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_ylabel(r'$P(k)$', size=30)
	ax.set_xlabel(r'$k$', size=30)

	ax.plot(k,P,label='linear')
	ax.plot(k,P_spt[0], label=r'$P_{22}(k) + P_{13}(k)$' )
	#ax.plot(k,P_IA_mix[0])
	#ax.plot(k,-1*P_IA_mix[0],'--')
	#ax.plot(k,P_IA_mix[1])
	#ax.plot(k,-1*P_IA_mix[1],'--')

	plt.legend(loc=3)
	plt.grid()
	plt.show()


'''
The beloe graph is from fastpt-examples.
It produces the correct graphs.
This function gives P_{d1d1}, P_{gg}, and P_{mg}
'''
import fastpt
import fastpt.HT as HT

# import the Core Cosmology Library (CCL) if you have it
try:
    import pyccl as ccl
    have_ccl = True
except:
    have_ccl = False
    print('CCL not found. Steps with CCL will be skipped.')

# If you want to test HT against external Bessel transform code, e.g. mcfit
try:
    from mcfit import P2xi
    have_mcfit = True
except:
    have_mcfit = False
    print('mcfit not found. Steps with mcfit will be skipped.')

## Get from CCL (which runs CLASS by default)
if have_ccl:
    # set two cosmologies
    cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2.1e-9, n_s=0.96)
    cosmo2 = ccl.Cosmology(Omega_c=0.30, Omega_b=0.045, h=0.67, A_s=2.0e-9, n_s=0.96)

    # Get the linear power spectrum at z=0 for our given cosmologies
    # k array to be used for power spectra
    nk = 512
    log10kmin = -5
    log10kmax = 2
    ks = np.logspace(log10kmin,log10kmax,nk)
    pk_lin_z0 = ccl.linear_matter_power(cosmo,ks,1)
    pk_lin_z0_2 = ccl.linear_matter_power(cosmo2,ks,1)

## Or get from pre-computed CAMB run
# This file is in the same examples/ folder
d = np.loadtxt('Pk_test.dat')
k = d[:, 0]
pk = d[:, 1]
p22 = d[:, 2]
p13 = d[:, 3]

#if not have_ccl:
ks = k
pk_lin_z0 = pk
pk_lin_z0_2 = None
    
## Or get from your preferred Boltzmann code

# Note: k needs to be evenly log spaced. FAST-PT will raise an error if it's not.
# We have an issue to add automatic interpolation, but this is not yet implemented.

# Evaluation time scales as roughly N*logN. Tradeoff between time and accuracy in choosing k resolution.
# Currently, k sampling must be done outside of FAST-PT. This feature will also be added.

# Set FAST-PT settings.

# the to_do list sets the k-grid quantities needed in initialization (e.g. the relevant gamma functions)
to_do = ['one_loop_dd', 'dd_bias', 'one_loop_cleft_dd', 'IA_all', 'OV', 'kPol', 'RSD', 'IRres']

pad_factor = 1 # padding the edges with zeros before Pk repeats
n_pad = pad_factor*len(ks)
low_extrap = -5 # Extend Plin to this log10 value if necessary (power law)
high_extrap = 3 # Extend Plin to this log10 value if necessary (power law)
P_window = None # Smooth the input power spectrum edges (typically not needed, especially with zero padding)
C_window = .75 # Smooth the Fourier coefficients of Plin to remove high-frequency noise.

# FAST-PT will parse the full to-do list and only calculate each needed quantity once.
# Ideally, the initialization happens once per likelihood evaluation, or even once per chain.

fpt_obj = FASTPT(ks,to_do=to_do,low_extrap=low_extrap,high_extrap=high_extrap,n_pad=n_pad)

#fpt_obj_temp = fpt.FASTPT(k,to_do=to_do,low_extrap=low_extrap,high_extrap=high_extrap,n_pad=n_pad)

# Parameters for a mock DESI sample at 0.6 < z < 0.8

# First, we need the growth factor and cosmology
def growth_factor(cc,zz):
	'''Retunrs linear growth factor for vector zz'''
	if isinstance(zz,float):
		zz = np.array([zz])
	afid = 1.0/(1.0+zz)
	if isinstance(afid,float):
		afid = np.array([afid])
	zval = 1./np.array(list(map(lambda x: np.logspace(x,0.0,100),np.log10(afid)))).transpose() - 1.0
	#zval = 1/np.logspace(np.log10(afid),0.0,100)-1.0
	Dz   = np.exp(-np.trapz(cc.Om(zval)**0.55,x=np.log(1/(1+zval)),axis=0))
	return(Dz)
	
# round-number cosmology is good enough!
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70.,Om0=0.3)

#zmin and zmax
zmin = 0.6
zmax = 0.8
zbar = 0.5 * (zmin + zmax)

# number density
number_density = 6e-4 # from Rongpu's targeting paper, in (h^1 Mpc)^-3

# Growth factor
growth = growth_factor(cosmo, zbar)

# Bias from 1611.00036
b1 = 1.7/growth

# Higher biases from consistency relations
# from 3.29 in https://arxiv.org/pdf/1611.09787.pdf
delta_cr = 1.686
nu_c = ((b1-1) * delta_cr + 1)**0.5
b2 = 8./21. * (nu_c**2. - 1)/delta_cr + nu_c**2./(delta_cr**2.) * (nu_c**2. - 3.)
# from pg 4 of https://arxiv.org/pdf/2008.05991.pdf
# Note that they are using bias definition appropriate for fastpt (there is a factor of 32/315 that is absorbed into
# b3nl versus the Saito paper that they cite)
bs = -4./7. * (nu_c**2. - 1)/delta_cr
b3nl = b1 - 1.

# Comoving volume and effective volume
v_survey = (4./3.)*np.pi * ((cosmo.comoving_distance(zmax).value*cosmo.h)**3. - (cosmo.comoving_distance(zmin).value*cosmo.h)**3.)

# Need an effective power spectrum
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
pk_spline = Spline(k,pk_lin_z0)
keff = 0.14
mueff = 0.6 # these values are from 1611.00036
Omz = 0.3 * (1+zbar)**3./(0.3 * (1+zbar)**3 + 0.7)
f = Omz ** 0.55
P_eff = (b1 + f * mueff**2.) **2. * growth **2. * pk_spline(keff)
effective_volume = (1 + (1./(number_density * P_eff)))**-2. * v_survey

# Monopole or xi(r)?

# For PT, we need to multiply by the relevant powers of the growth factor.
# For simplicity, we will do this all at z=0, where growth = 1. But we will keep the factors explicit.
g2 = growth**2
g4 = growth**4

## If you have CCL, you could use that here for growth at any redshift.
#if have_ccl:
#    z = 0.0
#    gz = ccl.growth_factor(cosmo,1./(1+z))
#    g2 = gz**2
#    g4 = gz**4

P_bias_E = fpt_obj.one_loop_dd_bias_b3nl(pk_lin_z0,C_window=C_window)

# Output individual terms
Pd1d1 = g2 * pk_lin_z0 + g4 * P_bias_E[0] # could use halofit or emulator instead of 1-loop SPT
Pd1d2 = g4 * P_bias_E[2]
Pd2d2 = g4 * P_bias_E[3]
Pd1s2 = g4 * P_bias_E[4]
Pd2s2 = g4 * P_bias_E[5]
Ps2s2 = g4 * P_bias_E[6]
Pd1p3 = g4 * P_bias_E[8]
s4 =  g4 * P_bias_E[7] # sigma^4 which determines the (non-physical) low-k contributions

P_IRres = g2 * fpt_obj.IRres(pk_lin_z0,C_window=C_window)
# Note that this function needs documentation/validation

r,xi_IRres = HT.k_to_r(ks,P_IRres,1.5,-1.5,.5, (2.*np.pi)**(-1.5))


# Combine for P_gg or P_mg
P_gg = ((b1 * b1) * P_IRres +
        0.5*(b1*b2 * 2) * Pd1d2 +
        0.25*(b2 * b2) * (Pd2d2 - 2.*s4) +
        0.5*(b1 * bs * 2) * Pd1s2 +
        0.25*(b2 * bs * 2) * (Pd2s2 - (4./3.)*s4) +
        0.25*(bs * bs) * (Ps2s2 - (8./9.)*s4) +
        0.5*(b1 * b3nl * 2) * Pd1p3)


#Calculating the CF
r = np.linspace(1., 300., 3000)
xi_gg = np.zeros(len(r))
xi_lin = np.zeros(len(r))
xi_gg_weighted = np.zeros(len(r))
xi_lin_weighted = np.zeros(len(r))
dlogks = ks[1] - ks[0]
dks = np.gradient(ks)

#Below is the code dedicated to adding noise the CF of P_gg, will come through the covariance matrix
#Fitting range 30 - 180
delta_r = 5

for i in range(len(r)):
	xi_gg[i] = np.sum(1 / (2 * math.pi**2) * ks**2 * special.spherical_jn(0, ks * r[i]) * np.exp(-ks**2) * P_gg * dks)
	xi_gg_weighted[i] = r[i]**2 * xi_gg[i]
	xi_lin_weighted[i] = r[i]**2 * np.sum(1 / (2 * math.pi**2) * ks**2 * special.spherical_jn(0, ks * r[i]) * np.exp(-ks**2) * b1**2 * growth**2 * pk_lin_z0 * dks) 
	xi_lin[i] = np.sum(1 / (2 * np.pi**2) * ks**2 * special.spherical_jn(0, ks * r[i]) * np.exp(-ks**2) * b1**2 * growth**2 * pk_lin_z0 * dks) 

#Below are the two graphs that I put in the first draft of the paper
plt.figure()
plt.plot(r, xi_gg_weighted, label=r"$\xi_{\rm gg}$")
plt.plot(r, xi_lin_weighted, color="black", label=r"$\xi_{\rm lin}$")
plt.ylabel(r'$r^2 \xi (r)$')
plt.xlabel(r'$r [\rm Mpc]$')
plt.title("Weighted Galaxy Correlation Function Model")
plt.legend()
plt.show()


xi_lim_1 = np.zeros(len(r))
xi_lim_9 = np.zeros(len(r))
xi_lim_11 = np.zeros(len(r))

for i in range(len(r)):
	xi_lim_1[i] = r[i]**2 * np.sum(1 / (2 * math.pi**2) * ks**2 * special.spherical_jn(0, ks * r[i]) * np.exp(-ks**2) * pk_lin_z0 * ks) * dlogks
	xi_lim_9[i] = r[i]**2 * np.sum(1 / (2 * math.pi**2) * ks**2 * special.spherical_jn(0, 0.9 * ks * r[i]) * np.exp(-ks**2) * pk_lin_z0 * ks) * dlogks
	xi_lim_11[i] = r[i]**2 * np.sum(1 / (2 * math.pi**2) * ks**2 * special.spherical_jn(0, 1.1 * ks * r[i]) * np.exp(-ks**2) * pk_lin_z0 * ks) * dlogks
	
plt.figure()
plt.plot(r, xi_lim_1, color="black", label=r"$\alpha = 1$")
plt.plot(r, xi_lim_9, color="red", label=r"$\alpha = 0.9$")
plt.plot(r, xi_lim_11, color="orange", label=r"$\alpha = 1.1$")
plt.ylabel(r'$r^2 \xi (r)$')
plt.xlabel(r'$r [\rm Mpc]$')
plt.title("Different Scaling Galaxy Correlation Function Model")
plt.legend()
plt.show()

'''
Ps, diag gaussian in FS. K mode is independent in FS, cannot be true in real space. Add noise in k space, then FT in real space
Add noise to P_gg of gaussian times K, and plug in shot noise
'''

#Recalculating the CF
r_bins = np.linspace(30, 180, 31)
print(r_bins)
r = 0.5 * (r_bins[1:] + r_bins[:-1])
print(r)
xi_gg = np.zeros(len(r))
dlogks = ks[1] - ks[0]

for i in range(len(r)):
	xi_gg[i] = np.sum(1 / (2 * math.pi**2) * ks**2 * special.spherical_jn(0, ks * r_bins[i]) * np.exp(-ks**2) * P_gg * np.gradient(ks))

print("xi shape: ", np.shape(xi_gg))

R1, R2 = np.meshgrid(r,r) 

# use b1 ^2 * Pklin in the covariance matrix
j0_return = 2./(effective_volume * number_density * np.pi**2.) * j0j0.rotation_method_bessel_j0j0(ks, b1 ** 2 * pk_lin_z0 * growth **2., R1, R2)
# this one may need to be corrected to include P2, P4, if we are fitting xi0 (or are we fitting xi(r))?
j0_return_pk_sq = 1./(effective_volume * np.pi**2.) * j0j0.rotation_method_bessel_j0j0(ks, (b1 ** 2 * pk_lin_z0 * growth **2.)**2., R1, R2)

dirac_cf_diag = xi_gg/delta_r
dirac_cf_matrix = np.diag(dirac_cf_diag)
dirac_cf_matrix *= 1 / (effective_volume * number_density**2)
dirac_cf_matrix *= 2/(4*np.pi*r**2.)

a = np.zeros((30,30))
np.fill_diagonal(a, 1)
dirac_matrix = a
dirac_matrix *= 1 / (effective_volume * number_density**2)
dirac_matrix /= delta_r
dirac_matrix *= 2/(4*np.pi*r**2.)

covariance_matrix = j0_return + j0_return_pk_sq + dirac_cf_matrix + dirac_matrix

'''
binning all the terms
shot noise = numerical integrals
pk = j1

'''

#Rows will store r values
#Columns will give the k value we integrated over
#ANother check, code this analytically. SLize at r and plot vs k
def get_j0bar(kk, rr):
    nkbins = len(kk)
    half_width = (rr[1] - rr[0])*0.49
    j0_bar = numpy.zeros([len(rr), nkbins])
    for ii, ir in enumerate(rr):
        u = numpy.linspace(ir-half_width, ir+half_width, 100)
        du = u[1] - u[0]
        uv, kv = numpy.meshgrid(u, kk, indexing='ij')
        norm = numpy.sum(uv*uv, axis=0)*du
        ans = numpy.sum(uv*uv*spherical_jn(0, uv*kv), axis=0)*du
        ans /= norm
        j0_bar[ii,:] = ans
    return j0_bar

#Run check on j0 transform
#xi_gg with np.random.multivariate_normal
#Set up tests of the correlation function
#Tests of j0
#Set r' = 0, pk transform recovers linear cf
#Set r2 = 0, make sure it recovers linear CF
#Superimpose linear cf and this to see if they the same

#np.random.multivariate_normal(xi_gg, covariance_matrix)

r = np.linspace(1., 300., 3000)
j0_test = 1 / (2 * np.pi**2) * j0j0.rotation_method_bessel_j0j0(ks, b1**2 * pk_lin_z0 * growth**2., R1, .01)
print(np.shape(j0_test))
print(j0_test[0,:])
plt.figure()
plt.plot(r[200:1300],  xi_lin[200:1300], label=r"$\xi_{\rm lin}$")
plt.plot(R1[0,:], j0_test[0,:], color="black", label="j0_test .01")
plt.ylabel(r'$\xi (r)$')
plt.xlabel(r'$r [\rm Mpc]$')
plt.title("J0 Test")
plt.legend()
plt.show()

plt.figure()
plt.plot(r[200:1300], np.log(xi_lin[200:1300]), label=r"$\xi_{\rm lin}$")
plt.plot(R1[0,:], np.log(j0_test[0,:]), color="black", label="j0_test .01")
plt.ylabel(r'$\xi (r)$')
plt.xlabel(r'$r [\rm Mpc]$')
plt.title("J0 Test")
plt.legend()
plt.show()
