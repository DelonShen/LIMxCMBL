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
log2 = 12
Lambdas = np.logspace(-5, -1, 25)
Lambda = Lambdas[Lambda_idx]

oup_fname = '/scratch/users/delon/LIMxCMBL/eHIeHI/mpmath_zmin_%.5f_zmax_%.5f_Lambda_%.5e.npy'%(zmin, zmax, Lambda)
print(oup_fname)

chimin = ccl.comoving_angular_distance(cosmo, 1/(1+zmin))
chimax = ccl.comoving_angular_distance(cosmo, 1/(1+zmax))


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
