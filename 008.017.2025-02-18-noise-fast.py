import numpy as np

import matplotlib.pyplot as plt

from LIMxCMBL.cross_spectrum import *
from LIMxCMBL.init import *
from LIMxCMBL.noise import *
from LIMxCMBL.kernels import *

from tqdm import tqdm

import sys
Lambda_idx = np.int32(sys.argv[1])

# CCAT-prime
zmin = 3.5
zmax = 8.1
chimin = ccl.comoving_angular_distance(cosmo, 1/(1+zmin))
chimax = ccl.comoving_angular_distance(cosmo, 1/(1+zmax))

Lambda = Lambdas[Lambda_idx]
log2 = 12


oup_fname = '/scratch/users/delon/LIMxCMBL/eHIeHI/scipy_zmin_%.5f_zmax_%.5f_Lambda_%.5e_%d'%(zmin, zmax, Lambda, log2)
print(oup_fname)


external_chis   = np.linspace(chimin*(1+1e-8), chimax*(1 - 1e-8), 2**log2)
analytical_diag = np.real(eLOeLO_diag_numpy(a = chimin, 
                                            b = chimax, 
                                            x = external_chis, 
                                            L = Lambda))

_chi  = external_chis.reshape(-1, 1)
_chip = external_chis.reshape(1, -1)

analytical_eLOeLO = eLOeLO_off_diag_numpy(L = Lambda, 
                                          a = chimin, 
                                          b = chimax, 
                                          x = _chi, 
                                          xp = _chip)
#set the diagonal to the correct quantity
np.fill_diagonal(analytical_eLOeLO, analytical_diag)

fname_LL = oup_fname + '_eLOeLO.npy'
print(fname_LL)
np.save(fname_LL, analytical_eLOeLO)

dchi = np.mean(np.diff(external_chis))
eIeI = f_eIeI(external_chis, dchi = dchi, Lambda = Lambda)
fname_II = oup_fname + '_eIeI.npy'
print(fname_II)
np.save(fname_II, eIeI)

cross = f_cross(chi = _chi,
                chip = _chip,
                Lambda = Lambda)
fname_cross = oup_fname + '_cross.npy'
print(fname_cross)
np.save(fname_cross, cross)
