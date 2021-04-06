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

class Info:
	def __init__(self, alpha_):
		self.alpha = alpha_
		import fastpt
		import fastpt.HT as HT

		# import the Core Cosmology Library (CCL) if you have it
		try:
		    import pyccl as ccl
		    have_ccl = True
		except:
		    have_ccl = False

		# If you want to test HT against external Bessel transform code, e.g. mcfit
		try:
		    from mcfit import P2xi
		    have_mcfit = True
		except:
		    have_mcfit = False

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
		self.d = np.loadtxt('Pk_test.dat')
		self.k = self.d[:, 0]
		self.pk = self.d[:, 1]
		self.p22 = self.d[:, 2]
		self.p13 = self.d[:, 3]

		#if not have_ccl:
		self.ks = self.k
		self.pk_lin_z0 = self.pk
		self.pk_lin_z0_2 = None
		    
		## Or get from your preferred Boltzmann code

		# Note: k needs to be evenly log spaced. FAST-PT will raise an error if it's not.
		# We have an issue to add automatic interpolation, but this is not yet implemented.

		# Evaluation time scales as roughly N*logN. Tradeoff between time and accuracy in choosing k resolution.
		# Currently, k sampling must be done outside of FAST-PT. This feature will also be added.

		# Set FAST-PT settings.

		# the to_do list sets the k-grid quantities needed in initialization (e.g. the relevant gamma functions)
		self.to_do = ['one_loop_dd', 'dd_bias', 'one_loop_cleft_dd', 'IA_all', 'OV', 'kPol', 'RSD', 'IRres']

		self.pad_factor = 1 # padding the edges with zeros before Pk repeats
		self.n_pad = self.pad_factor*len(self.ks)
		self.low_extrap = -5 # Extend Plin to this log10 value if necessary (power law)
		self.high_extrap = 3 # Extend Plin to this log10 value if necessary (power law)
		self.P_window = None # Smooth the input power spectrum edges (typically not needed, especially with zero padding)
		self.C_window = .75 # Smooth the Fourier coefficients of Plin to remove high-frequency noise.

		# FAST-PT will parse the full to-do list and only calculate each needed quantity once.
		# Ideally, the initialization happens once per likelihood evaluation, or even once per chain.

		self.fpt_obj = FASTPT(self.ks,to_do=self.to_do,low_extrap=self.low_extrap,high_extrap=self.high_extrap,n_pad=self.n_pad)

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
		self.zmin = 0.6
		self.zmax = 0.8
		self.zbar = 0.5 * (self.zmin + self.zmax)

		# number density
		self.number_density = 6e-4 # from Rongpu's targeting paper, in (h^1 Mpc)^-3

		# Growth factor
		self.growth = growth_factor(cosmo, self.zbar)

		# Bias from 1611.00036
		self.b1 = 1.7/self.growth

		# Higher biases from consistency relations
		# from 3.29 in https://arxiv.org/pdf/1611.09787.pdf
		self.delta_cr = 1.686
		#self.delta_cr = 0.
		self.nu_c = ((self.b1-1) * self.delta_cr + 1)**0.5
		#self.nu_c = 0.
		self.b2 = 8./21. * (self.nu_c**2. - 1)/self.delta_cr + self.nu_c**2./(self.delta_cr**2.) * (self.nu_c**2. - 3.)
		#self.b2 = 0.

		# from pg 4 of https://arxiv.org/pdf/2008.05991.pdf
		# Note that they are using bias definition appropriate for fastpt (there is a factor of 32/315 that is absorbed into
		# b3nl versus the Saito paper that they cite)
		self.bs = -4./7. * (self.nu_c**2. - 1)/self.delta_cr
		#self.bs = 0.
		self.b3nl = self.b1 - 1.
		#self.b3nl = 0.

		# Comoving volume and effective volume
		self.v_survey = (4./3.)*np.pi * ((cosmo.comoving_distance(self.zmax).value*cosmo.h)**3. - (cosmo.comoving_distance(self.zmin).value*cosmo.h)**3.)

		# Need an effective power spectrum
		from scipy.interpolate import InterpolatedUnivariateSpline as Spline
		self.pk_spline = Spline(self.k,self.pk_lin_z0)
		self.keff = 0.14
		self.mueff = 0.6 # these values are from 1611.00036
		self.Omz = 0.3 * (1+self.zbar)**3./(0.3 * (1+self.zbar)**3 + 0.7)
		self.f = self.Omz ** 0.55
		self.P_eff = (self.b1 + self.f * self.mueff**2.) **2. * self.growth **2. * self.pk_spline(self.keff)
		self.effective_volume = (1 + (1./(self.number_density * self.P_eff)))**-2. * self.v_survey

		# Monopole or xi(r)?

		# For PT, we need to multiply by the relevant powers of the growth factor.
		# For simplicity, we will do this all at z=0, where growth = 1. But we will keep the factors explicit.
		self.g2 = self.growth**2
		self.g4 = self.growth**4

		## If you have CCL, you could use that here for growth at any redshift.
		#if have_ccl:
		#    z = 0.0
		#    gz = ccl.growth_factor(cosmo,1./(1+z))
		#    g2 = gz**2
		#    g4 = gz**4

		self.P_bias_E = self.fpt_obj.one_loop_dd_bias_b3nl(self.pk_lin_z0, C_window=self.C_window)

		# Output individual terms
		self.Pd1d1 = self.g2 * self.pk_lin_z0 + self.g4 * self.P_bias_E[0] # could use halofit or emulator instead of 1-loop SPT
		self.Pd1d2 = self.g4 * self.P_bias_E[2]
		self.Pd2d2 = self.g4 * self.P_bias_E[3]
		self.Pd1s2 = self.g4 * self.P_bias_E[4]
		self.Pd2s2 = self.g4 * self.P_bias_E[5]
		self.Ps2s2 = self.g4 * self.P_bias_E[6]
		self.Pd1p3 = self.g4 * self.P_bias_E[8]
		self.s4 =  self.g4 * self.P_bias_E[7] # sigma^4 which determines the (non-physical) low-k contributions

		self.P_IRres = self.g2 * self.fpt_obj.IRres(self.pk_lin_z0, C_window=self.C_window)
		# Note that this function needs documentation/validation

		self.r, self.xi_IRres = HT.k_to_r(self.ks, self.P_IRres,1.5,-1.5,.5, (2.*np.pi)**(-1.5))

		# Combine for P_gg or P_mg
		self.P_gg = ((self.b1 * self.b1) * self.P_IRres +
		        0.5*(self.b1 * self.b2 * 2) * self.Pd1d2 +
		        0.25*(self.b2 * self.b2) * (self.Pd2d2 - 2.*self.s4) +
		        0.5*(self.b1 * self.bs * 2) * self.Pd1s2 +
		        0.25*(self.b2 * self.bs * 2) * (self.Pd2s2 - (4./3.)*self.s4) +
		        0.25*(self.bs * self.bs) * (self.Ps2s2 - (8./9.)*self.s4) +
		        0.5*(self.b1 * self.b3nl * 2) * self.Pd1p3)

		self.x = [_ for _ in range(len(self.P_gg))]
		'''
		Ps, diag gaussian in FS. K mode is independent in FS, cannot be true in real space. Add noise in k space, then FT in real space
		Add noise to P_gg of gaussian times K, and plug in shot noise
		'''

		#Recalculating the CF
		self.r_bins = np.linspace(30, 180, 31)
		self.r = 0.5 * (self.r_bins[1:] + self.r_bins[:-1])
		self.xi_gg = np.zeros(len(self.r))
		self.R1, self.R2 = np.meshgrid(self.r, self.r)
		self.delta_r = 5 
		
	def get_r(self):
		return self.r

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

	def calc_CF(self):
		for i in range(len(self.r)):
			self.xi_gg[i] = np.sum(1 / (2 * math.pi**2) * self.ks**2 * special.spherical_jn(0, self.alpha * self.ks * self.r_bins[i]) * np.exp(-self.ks**2) * self.P_gg * np.gradient(self.ks))

		# use b1 ^2 * Pklin in the covariance matrix
		j0_return = 2./(self.effective_volume * self.number_density * np.pi**2.) * j0j0.rotation_method_bessel_j0j0(self.ks, self.b1 ** 2 * self.pk_lin_z0 * self.growth **2., self.R1, self.R2)
		# this one may need to be corrected to include P2, P4, if we are fitting xi0 (or are we fitting xi(r))?
		j0_return_pk_sq = 1./(self.effective_volume * np.pi**2.) * j0j0.rotation_method_bessel_j0j0(self.ks, (self.b1 ** 2 * self.pk_lin_z0 * self.growth **2.)**2., self.R1, self.R2)

		dirac_cf_diag = self.xi_gg / self.delta_r
		dirac_cf_matrix = np.diag(dirac_cf_diag)
		dirac_cf_matrix *= 1 / (self.effective_volume * self.number_density**2)
		dirac_cf_matrix *= 2/(4 * np.pi * self.r**2.)

		a = np.zeros((30,30))
		np.fill_diagonal(a, 1)
		dirac_matrix = a
		dirac_matrix *= 1 / (self.effective_volume * self.number_density**2)
		dirac_matrix /= self.delta_r
		dirac_matrix *= 2 / (4 * np.pi * self.r**2.)

		self.covariance_matrix = j0_return + j0_return_pk_sq + dirac_cf_matrix + dirac_matrix
		np.random.seed(1234) # for reproducibility
		self.xi_gg_data = np.random.multivariate_normal(self.xi_gg, self.covariance_matrix)

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
	        ans /= norm
	        j0_bar[ii,:] = ans
	    return j0_bar

	#Run check on j0 transform
	#xi_gg with np.random.multivariate_normal
	#Set up tests of the correlation function	        ans = numpy.sum(uv*uv*spherical_jn(0, uv*kv), axis=0)*du
	        
	#Tests of j0
	#Set r' = 0, pk transform recovers linear cf
	#Set r2 = 0, make sure it recovers linear CF
	#Superimpose linear cf and this to see if they the same
	#This adds the noise to xi_gg

	def get_data(self):
		return self.xi_gg_data

	def get_covariance_matrix(self):
		return self.covariance_matrix

	#Calculate the templates
	#Recalculating the CF
	
	def templates(self):
		self.xi_IRrs = np.zeros(len(self.r))

		for i in range(len(self.r)):
			self.xi_IRrs[i] = np.sum(1 / (2 * math.pi**2) * self.ks**2 * special.spherical_jn(0, self.ks * self.r_bins[i]) * np.exp(-self.ks**2) * self.P_IRres * np.gradient(self.ks))
		print('xi: ', self.xi_IRrs[0])
		return self.xi_IRrs

	def templates_deriv(self):
		self.xi_IRrs_prime = np.zeros(len(self.r))

		for i in range(len(self.r)):
			self.xi_IRrs_prime[i] = -self.r_bins[i] * np.sum(1 / (2 * math.pi**2) * self.ks**3 * special.spherical_jn(1, self.ks * self.r_bins[i]) * np.exp(-self.ks**2) * self.P_IRres * np.gradient(self.ks))
		print('xi\': ', self.xi_IRrs_prime[0])
		return self.xi_IRrs_prime

	def templates_deriv2(self):
		self.xi_IRrs_prime2 = np.zeros(len(self.r))

		for i in range(len(self.r)):
			self.xi_IRrs_prime2[i] = self.r_bins[i]**2 * np.sum(1 / (2 * math.pi**2) * self.ks**4 * (special.spherical_jn(2, self.ks * self.r_bins[i]) - (1 / (self.ks * self.r_bins[i])) * special.spherical_jn(1, self.ks * self.r_bins[i])) * np.exp(-self.ks**2) * self.P_IRres * np.gradient(self.ks))
		print('xi\'\': ', self.xi_IRrs_prime2[0])
		return self.xi_IRrs_prime2

	def get_biases(self):
		return (self.b1 * self.b1)