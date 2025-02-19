import numpy as np

import matplotlib.pyplot as plt

from LIMxCMBL.cross_spectrum import *
from LIMxCMBL.init import *
from LIMxCMBL.noise import *
from LIMxCMBL.kernels import *

from scipy.signal.windows import dpss

from tqdm import tqdm

import sys
Lambda_idx = np.int32(sys.argv[1])

# CCAT-prime
zmin = 3.5
zmax = 8.1
chimin = ccl.comoving_angular_distance(cosmo, 1/(1+zmin))
chimax = ccl.comoving_angular_distance(cosmo, 1/(1+zmax))

Lambda = Lambdas[Lambda_idx]
log2 = 13

oup_fname = '/scratch/users/delon/LIMxCMBL/eHIeHI/mpmath_zmin_%.5f_zmax_%.5f_Lambda_%.5e_log2_%d.npy'%(zmin, zmax, Lambda, log2)
print(oup_fname)

external_chis   = np.linspace(chimin*(1+1e-8), chimax*(1 - 1e-8), 2**log2)
analytical_diag = np.real(eLOeLO_diag_numpy(a = chimin, 
                                            b = chimax, 
                                            x = external_chis, 
                                            L = Lambda))



mpm_chis_dense = mpm.linspace(chimin*(1+1e-8), chimax*(1 - 1e-8), 2**log2)
mpm_dchi = mpm_chis_dense[1] - mpm_chis_dense[0]

results = f_eHIeHI(chimin = chimin, 
                   chimax = chimax, 
                   dchi = mpm_dchi, 
                   chis = mpm_chis_dense, 
                   Lambda = Lambda)


import pickle
with open(oup_fname, 'wb') as f:
    print(oup_fname)
    pickle.dump(results, f)
