from LIMxCMBL.cross_spectrum import *
from LIMxCMBL.init import *
from LIMxCMBL.kernels import get_f_Kkappa, get_f_KI, get_f_KILo, apply_window

import sys
zmin = 3.5
zmax = 8.1
log2 = 15

oup_fname = '/scratch/users/delon/LIMxCMBL/IHiKappa/Ik_zmin_%.5f_zmax_%.5f_chi_sample_%d.npy'%(zmin, zmax, log2)
print(oup_fname)


chimin = ccl.comoving_angular_distance(cosmo, 1/(1+zmin))
chimax = ccl.comoving_angular_distance(cosmo, 1/(1+zmax))

f_Kkappa = get_f_Kkappa()
a_of_chibs = ccl.scale_factor_of_chi(cosmo, chibs)
z_of_chibs = 1/a_of_chibs - 1


f_KLIM   = get_f_KI()
f_KLIM_windowed = apply_window(f_K = f_KLIM,
                               chimin = chimin,
                               chimax = chimax)


_deltas = deltas.reshape(1, 1, -1)
_chis   = np.linspace(chimin*(1+1e-8), chimax*(1 - 1e-8), 2**log2)
_chis_rs = _chis.reshape(1, -1, 1)
_minus = _chis_rs * (1 - _deltas)
_plus  = _chis_rs * (1 + _deltas)

f_Kpsi = f_KLIM_windowed
f_Kphi = f_Kkappa

prefactor = 2 / _chis_rs
kernels = (f_Kpsi(_minus) * f_Kphi(_plus) + f_Kpsi(_plus) * f_Kphi(_minus))
inner_integral_resampled = f_inner_integral(_chis)
integrand = prefactor*kernels*inner_integral_resampled*deltas_reshaped
result = trapezoid(y=integrand, x=np.log(deltas), axis = -1)

np.save(oup_fname, result)
print('outputted')
