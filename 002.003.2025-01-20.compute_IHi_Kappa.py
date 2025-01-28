from LIMxCMBL.cross_spectrum import *
from LIMxCMBL.init import *
from LIMxCMBL.kernels import get_f_Kkappa, get_f_KI, get_f_KILo, apply_window

import sys
zmin = np.float64(sys.argv[1])
zmax = np.float64(sys.argv[2])
Lambda = np.float64(sys.argv[3]) # 1 / cMpc 
chis_resample_len = int(np.log2(len(chis_resample)))
oup_fname = '/scratch/users/delon/LIMxCMBL/IHiKappa/zmin_%.5f_zmax_%.5f_Lambda_%.5f_chi_sample_2e%d.npy'%(zmin, zmax, Lambda,chis_resample_len)
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


############
#<ILo Kappa>
from tqdm import trange
ILok = np.zeros((len(ells), len(chis_resample)), dtype=np.float32)
external_chis = chis_resample.reshape(-1,1, 1, 1)

for i in trange(len(chis_resample) // 2**3):
    idx_left = i * 2**3
    idx_right = (i+1) * 2 ** 3
    f_KLIMLo = get_f_KILo(external_chi = external_chis[idx_left:idx_right], Lambda=Lambda)
    f_KLIMLo_windowed = apply_window(f_K = f_KLIMLo,
                                     chimin = chimin,
                                     chimax = chimax)
    ILok[:, idx_left:idx_right] = d_chib_integral(f_KLIMLo_windowed, f_Kkappa).T
############



oup_ILo_fname = '/scratch/users/delon/LIMxCMBL/IHiKappa/ILoKappa/zmin_%.5f_zmax_%.5f_Lambda_%.5f.npy'%(zmin, zmax, Lambda)
np.save(oup_ILo_fname, ILok)

IHi_kappa = Ik - ILok
np.save(oup_fname, IHi_kappa)

print('outputted ILoK to', oup_ILo_fname)
print('outputted IHiK to', oup_fname)
