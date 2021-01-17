#Author: Nina Brown
import os 
import sys
import numpy as np

#Import CAMB
#Assume installed from github using "git clone --recursive https://github.com/cmbant/CAMB.git"
#This file is then in the docs folders
camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))
sys.path.insert(0,camb_path)
import camb
from camb import model, initialpower
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))

def camb_transfer():
    pars=camb.read_ini(os.path.join(camb_path,'2PCF','CAMB-1.2.0', 'inifiles','planck_2018.ini'))
    pars.set_cosmology(H0=67.77, ombh2=0.022139921, omch2=0.11891102387) # these are the basic cosmo parameters, convenient to change here if needed, i really don't think this line is necessary tbh
    pars.NonLinear=model.NonLinear_none
    pars.WantTransfer = True
    pars.InitPower.set_params(ns=0.9611) # more general params for convenience 
    pars.set_matter_power(redshifts=[0,42,51,59,965,1073,1180], kmax=15, k_per_logint=5000) # these were the different redshifts and k's we needed T_m's for, change if needed
    results=camb.get_transfer_functions(pars)
    trans = results.get_matter_transfer_data()
    print(pars)
    koh_results=trans.transfer_data[0]
    print(len(koh_results))
    return trans.transfer_data[:]

# run CAMB
CAMB_results=camb_transfer()
deltCDM_results=CAMB_results[1]
deltB_results=CAMB_results[2]
delttot_results=CAMB_results[6,:,0]
koh_results=CAMB_results[0,:,6]

# save
np.save('deltCDM_results_22_July_2020.npy',deltCDM_results)
np.save('deltB_results_22_July_2020.npy',deltB_results)
np.save('delttot_results_22_July_2020.npy', delttot_results)
np.save('koh_results_22_July_2020.npy',koh_results)
