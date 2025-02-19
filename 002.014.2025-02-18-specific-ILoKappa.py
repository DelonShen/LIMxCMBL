from LIMxCMBL.cross_spectrum import *
from LIMxCMBL.init import *
from LIMxCMBL.kernels import get_f_Kkappa, get_f_KI, get_f_KILo, apply_window

from tqdm import trange

import sys
zmin = 3.5
zmax = 8.1
log2 = 13
Lambda_idx = np.int32(sys.argv[1])
Lambda = Lambdas[Lambda_idx]

oup_fname = '/scratch/users/delon/LIMxCMBL/IHiKappa/ILok_zmin_%.5f_zmax_%.5f_Lambda_%.5f_chi_sample_%d.npy'%(zmin, zmax, Lambda, log2)
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

external_chis   = np.linspace(chimin*(1+1e-8), chimax*(1 - 1e-8), 2**log2)

ILok = np.zeros((len(ells), len(external_chis)), dtype=np.float64)
external_chis = external_chis.reshape(-1, 1, 1, 1)

chunk = 10
for i in trange(len(external_chis) // chunk + 1):
    idx_left = i * chunk
    idx_right = min((i+1) * chunk, len(external_chis))
    f_KLIMLo = get_f_KILo(external_chi = external_chis[idx_left:idx_right], Lambda=Lambda)
    f_KLIMLo_windowed = apply_window(f_K = f_KLIMLo,
                                     chimin = chimin,
                                     chimax = chimax)
    ILok[:, idx_left:idx_right] = d_chib_integral(f_KLIMLo_windowed, f_Kkappa).T

np.save(oup_fname, ILok)
print('outputted')
