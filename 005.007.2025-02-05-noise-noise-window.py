from LIMxCMBL.init import *
from LIMxCMBL.noise import *

import sys
Lambda = np.float64(sys.argv[1]) # 1 / cMpc 
chis_resample_len = int(np.log2(len(chis_resample)))


eIeI, eLOeLO, eIeLO, eLOeI = get_eHIeHI(0, chimax_sample*1.1, Lambda)
oup_fname = '/scratch/users/delon/LIMxCMBL/eHIeHI/Lambda_%.5e_chi_sample_2e%d'%(Lambda,chis_resample_len)
np.save(oup_fname + 'eIeI.npy'  , eIeI)
np.save(oup_fname + 'eIeLO.npy' , eIeLO)
np.save(oup_fname + 'eLOeI.npy' , eLOeI)
np.save(oup_fname + 'eLOeLO.npy', eLOeLO)
