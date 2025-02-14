from LIMxCMBL.cross_spectrum import *
from LIMxCMBL.init import *
from LIMxCMBL.noise import *

import sys
zmin = np.float64(sys.argv[1])
zmax = np.float64(sys.argv[2])


chimin = ccl.comoving_angular_distance(cosmo, 1/(1+zmin))
chimax = ccl.comoving_angular_distance(cosmo, 1/(1+zmax))

#Lambdas = np.logspace(-4, -1, 50)
Lambdas = np.logspace(-5, 0, 50)
print(Lambdas)
chis_resample_len = int(np.log2(len(chis_resample)))
window = np.where((chis_resample > chimin) & (chis_resample < chimax))[0]

from tqdm import tqdm

for Lambda in tqdm(Lambdas):
    eIeI, eLOeLO, eIeLO, eLOeI = get_eHIeHI(chimin, chimax, Lambda)
    oup_fname = '/scratch/users/delon/LIMxCMBL/eHIeHI/zmin_%.5f_zmax_%.5f_Lambda_%.5e_chi_sample_2e%d'%(zmin, zmax, Lambda,chis_resample_len)
    np.save(oup_fname + 'eIeI.npy'  , eIeI)
    np.save(oup_fname + 'eIeLO.npy' , eIeLO)
    np.save(oup_fname + 'eLOeI.npy' , eLOeI)
    np.save(oup_fname + 'eLOeLO.npy', eLOeLO)
