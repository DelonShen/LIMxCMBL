from LIMxCMBL.cross_spectrum import *
from LIMxCMBL.init import *

import sys
zmin = np.float64(sys.argv[1])
zmax = np.float64(sys.argv[2])


chimin = ccl.comoving_angular_distance(cosmo, 1/(1+zmin))
chimax = ccl.comoving_angular_distance(cosmo, 1/(1+zmax))

#Lambdas = np.logspace(-4, -1, 50)
Lambdas = np.logspace(-5, 0, 50)
print(Lambdas)
chis_resample_len = int(np.log2(len(chis_resample)))
print(chis_resample_len)
window = np.where((chis_resample > chimin) & (chis_resample < chimax))[0]
chis = chis_resample[window]


chis = np.linspace(np.min(chis),
                   np.max(chis),
                   8 * len(chis))


_chi  = chis.reshape(-1, 1)
_chip = chis.reshape(1 ,-1)
dchi = np.mean(np.diff(chis))
print(len(chis))

from tqdm import tqdm

from numba import njit
    
for Lambda in tqdm(Lambdas):
    print(Lambda)
    if(Lambda < 1e-2):
        continue

    f_eIeI = lambda chi, chip, dchi : 1 / (dchi * chi ** 2)
    f_eIeLO = lambda chi, chip : 1/chi**2  * Lambda / np.pi * np.sinc(Lambda * (chi - chip) / np.pi)
    f_eLOeI = lambda chi, chip : 1/chip**2 * Lambda / np.pi * np.sinc(Lambda * (chi - chip) / np.pi)

    eIeI = np.diag(f_eIeI(chis, chis, dchi))
    eIeLO = f_eIeLO(_chi, _chip)
    eLOeI = f_eLOeI(_chi, _chip)

    @njit()
    def integrand(_chib): 
        return (Lambda**2 / np.pi**2 / _chib ** 2 * np.sinc(Lambda * (_chi  - _chib) / np.pi) * np.sinc(Lambda * (_chip - _chib) / np.pi))


    eLOeLO, _ = quad_vec(integrand, chimin, chimax, epsabs = 0.0, epsrel = 1e-3, workers = 32)

    oup_fname = '/scratch/users/delon/LIMxCMBL/eHIeHI/superdense_zmin_%.5f_zmax_%.5f_Lambda_%.5e_chi_sample_2e%d'%(zmin, zmax, Lambda,chis_resample_len)
    np.save(oup_fname + 'eIeI.npy'  , eIeI)
    np.save(oup_fname + 'eIeLO.npy' , eIeLO)
    np.save(oup_fname + 'eLOeI.npy' , eLOeI)
    np.save(oup_fname + 'eLOeLO.npy', eLOeLO)
