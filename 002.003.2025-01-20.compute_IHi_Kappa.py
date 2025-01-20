from LIMxCMBL.cross_spectrum import *
from LIMxCMBL.init import *
from LIMxCMBL.kernels import get_f_Kkappa, get_f_KI, get_f_KILo, apply_window

import sys
zmin = np.float64(sys.argv[1])
zmax = np.float64(sys.argv[2])
Lambda = np.float64(sys.argv[3]) # 1 / cMpc 
oup_fname = '/scratch/users/delon/LIMxCMBL/IHiKappa/zmin_%.5f_zmax_%.5f_Lambda_%.5f.npy'%(zmin, zmax, Lambda)
print('outputting to', oup_fname)


chimin = ccl.comoving_angular_distance(cosmo, 1/(1+zmin))
chimax = ccl.comoving_angular_distance(cosmo, 1/(1+zmax))

f_Kkappa = get_f_Kkappa()
a_of_chibs = ccl.scale_factor_of_chi(cosmo, chibs)
z_of_chibs = 1/a_of_chibs - 1


#<I Kappa>
f_KLIM   = get_f_KI()
f_KLIM_windowed = apply_window(f_K = f_KLIM,
                               chimin = chimin,
                               chimax = chimax)
Ik = d_delta_integral(f_KLIM_windowed, f_Kkappa)


#<ILo Kappa>
ILok = np.zeros((len(ells), len(chibs)), dtype=np.float128)

from tqdm import trange

for chi_idx in trange(len(chibs)):
    chi = chibs[chi_idx]
    f_KLIMLo   = get_f_KILo(external_chi = chi, Lambda=Lambda)
    f_KLIMLo_windowed = apply_window(f_K = f_KLIMLo,
                                     chimin = chimin,
                                     chimax = chimax)
    ILok[:, chi_idx] = d_chib_integral(f_KLIMLo_windowed, f_Kkappa)


oup_ILo_fname = '/scratch/users/delon/LIMxCMBL/IHiKappa/ILoKappa/zmin_%.5f_zmax_%.5f_Lambda_%.5f.npy'%(zmin, zmax, Lambda)
np.save(oup_ILo_fname, ILok)

IHi_kappa = Ik - ILok
np.save(oup_fname, IHi_kappa)

print('outputted ILoK to', oup_ILo_fname)
print('outputted IHiK to', oup_fname)

