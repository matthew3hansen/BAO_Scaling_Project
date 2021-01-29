
import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

#Assume installed from github using "git clone --recursive https://github.com/cmbant/CAMB.git"
#This file is then in the docs folders
camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))
sys.path.insert(0,camb_path)
import camb
from camb import model, initialpower
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))

def camb_transfer():
    '''
    pars=camb.read_ini(os.path.join(camb_path,'2PCF','CAMB-1.2.0', 'inifiles','planck_2018.ini'))
    pars.set_cosmology(H0=67.77, ombh2=0.022139921, omch2=0.11891102387) # these are the basic cosmo parameters, convenient to change here if needed, i really don't think this line is necessary tbh
    pars.NonLinear=model.NonLinear_none
    pars.WantTransfer = True
    pars.InitPower.set_params(ns=0.9611) # more general params for convenience 
    pars.set_matter_power(redshifts=[0,42], kmax=15, k_per_logint=5000) # these were the different redshifts and k's we needed T_m's for, change if needed
    results=camb.get_transfer_functions(pars)
    trans = results.get_matter_transfer_data()
    print(pars)
    koh_results=trans.transfer_data[0]
    print(len(koh_results))
    return trans.transfer_data[:]
    '''
    #Set up a new set of parameters for CAMB
    pars = camb.CAMBparams()
    #This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
    pars.set_for_lmax(2500, lens_potential_accuracy=0);

    #calculate results for these parameters
    results = camb.get_results(pars)

    #get dictionary of CAMB power spectra
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
    for name in powers: print(name)

    #Now get matter power spectra and sigma8 at redshift 0 and 0.8
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.InitPower.set_params(ns=0.965)
    #Note non-linear corrections couples to smaller scales than you want
    pars.set_matter_power(redshifts=[0., 0.8], kmax=2.0)

    #Linear spectra
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 200)
    s8 = np.array(results.get_sigma8())

    #Non-Linear spectra (Halofit)
    pars.NonLinear = model.NonLinear_both
    results.calc_power_spectra(pars)
    kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 200)

    print(results.get_sigma8())

    for i, (redshift, line) in enumerate(zip(z,['-','--'])):
        plt.loglog(kh, pk[i,:], color='k', ls = line)
        plt.loglog(kh_nonlin, pk_nonlin[i,:], color='r', ls = line)
    plt.xlabel('k/h Mpc');
    plt.legend(['linear','non-linear'], loc='lower left');
    plt.title('Matter power at z=%s and z= %s'%tuple(z));
    plt.show()

#camb_transfer()

def get_matter_spectrum():
    #Set up a new set of parameters for CAMB
    pars = camb.CAMBparams()
    #This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
    pars.set_for_lmax(2500, lens_potential_accuracy=0);

    #calculate results for these parameters
    results = camb.get_results(pars)

    #get dictionary of CAMB power spectra
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
    for name in powers: print(name)

    #Now get matter power spectra and sigma8 at redshift 0 and 0.8
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.InitPower.set_params(ns=0.965)
    #Note non-linear corrections couples to smaller scales than you want
    pars.set_matter_power(redshifts=[0., 0.8], kmax=2.0)

    #Linear spectra
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    s8 = np.array(results.get_sigma8())
    print(s8)
    return results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 200)

def get_linear_matter_power_spectrum():
    #Set up a new set of parameters for CAMB
    pars = camb.CAMBparams()
    #This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
    pars.set_for_lmax(2500, lens_potential_accuracy=0);

    #calculate results for these parameters
    results = camb.get_results(pars)

    #get dictionary of CAMB power spectra
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    for name in powers: print(name)

    #Now get matter power spectra and sigma8 at redshift 0 and 0.8
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.InitPower.set_params(ns=0.965)
    #Note non-linear corrections couples to smaller scales than you want
    pars.set_matter_power(redshifts=[0., 0.8], kmax=2.0)

    #Linear spectra
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    s8 = np.array(results.get_sigma8())
    print(s8)
    return results.get_linear_matter_power_spectrum(var1=None, var2=None, hubble_units=True, k_hunit=True, have_power_spectra=True, params=None, nonlinear=False)
'''
# run CAMB
CAMB_results=camb_transfer()
print("this is what I'm looking for", CAMB_results[6])
deltCDM_results=CAMB_results[1]
deltB_results=CAMB_results[2]
delttot_results=CAMB_results[6,:,0]
koh_results=CAMB_results[0,:,6]

# save
np.save('deltCDM_results_22_July_2020.npy',deltCDM_results)
np.save('deltB_results_22_July_2020.npy',deltB_results)
np.save('delttot_results_22_July_2020.npy', delttot_results)
np.save('koh_results_22_July_2020.npy',koh_results)'''
