from LIMxCMBL.init import *
from LIMxCMBL.noise import *

import sys
Lambda_idx = np.int32(sys.argv[1])
factor = 4


# CCAT-prime
zmin = 3.5
zmax = 8.1

Omega_field = 8 * (np.pi/180)**2 #rad^2
Pei = 2.3e4 #Mpc^3 kJy^2 /sr^2 
chimin = ccl.comoving_angular_distance(cosmo, 1/(1+zmin))
chimax = ccl.comoving_angular_distance(cosmo, 1/(1+zmax))


oup_fname = '/scratch/users/delon/LIMxCMBL/eHIeHI/snr_per_mode_mpmath_zmin_%.5f_zmax_%.5f_Lambda_idx_%d_factor_%d'%(zmin, zmax, Lambda_idx, factor)
oup1_fname = oup_fname + '_full.npy'
oup2_fname = oup_fname + '_diag.npy'

print(oup1_fname)
print(oup2_fname)




# get CMB lensing component
from LIMxCMBL.kernels import get_f_Kkappa
f_WkD = get_f_Kkappa()

from LIMxCMBL.cross_spectrum import *
ClKK = d_chib_integral(f_WkD, f_WkD) #[Mpc]^2


# beam=1.4, noise=7
from scipy.interpolate import interp1d
N0_ells = np.logspace(1, np.log10(5000), 500)
with open('LIMxCMBL/N0.npy', 'rb') as f:
    N0 = np.load(f)
    
f_N0 = interp1d(x = N0_ells, y = N0)

    
from  LIMxCMBL.kernels import *
f_KLIM   = get_f_KI()
f_Kkappa = get_f_Kkappa()

f_KLIM_windowed = apply_window(f_K = f_KLIM,
                               chimin = chimin,
                               chimax = chimax)
I_kappa = d_delta_integral(f_KLIM_windowed, f_Kkappa) #[kJy / sr ] [Mpc]



from scipy.integrate import trapezoid, simpson, quad_vec
from scipy.interpolate import interp1d


from tqdm import trange, tqdm

window = np.where((chis_resample > chimin) & (chis_resample < chimax))[0]
chis_resample_len = int(np.log2(len(chis_resample)))
print(chis_resample_len)
Lambdas = np.logspace(-5, 0, 50)
_chis = chis_resample[window]


Lambda = Lambdas[Lambda_idx]
IHi_kappa_fname = '/scratch/users/delon/LIMxCMBL/IHiKappa/zmin_%.5f_zmax_%.5f_Lambda_%.5f_chi_sample_2e%d.npy'%(zmin, zmax, Lambda,chis_resample_len)
IHi_kappa = np.load(IHi_kappa_fname) #[kJy/sr Mpc]
IHi_kappa_windowed = IHi_kappa[:, window].astype(np.float64)

D = np.diag(_chis)
d = D @ IHi_kappa_windowed.T

integrand =  IHi_kappa**2 / (1/chis_resample**2) # [kJy^2/sr^2][Mpc^4]

f_d = interp1d(x = _chis, y = d, axis = 0)

mpm_chis_dense = mpm.linspace(np.min(_chis), 
                  np.max(_chis), 
                  factor * len(_chis))
mpm_dchi = mpm_chis_dense[1] - mpm_chis_dense[0]

oup_fname = '/scratch/users/delon/LIMxCMBL/eHIeHI/mpmath_zmin_%.5f_zmax_%.5f_Lambda_%.5e_factor_%d.npy'%(zmin, zmax, Lambda, factor)
with open(oup_fname, 'rb') as f:
    print(oup_fname)
    results = pickle.load(f)

mpm_eIeI = np.zeros((factor * len(_chis), factor * len(_chis)), 
                    dtype = np.complex128)
mpm_eLOeLO = np.zeros_like(mpm_eIeI, dtype = np.complex128)
mpm_cross = np.zeros_like(mpm_eIeI, dtype = np.complex128)
for i, j, _eIeI, _cross, _eLOeLO in tqdm(results):
    mpm_eIeI[i,j] = _eIeI
    mpm_eLOeLO[i,j] = mpm_eLOeLO[j,i] = _eLOeLO
    mpm_cross[i,j] = mpm_cross[j,i] = _cross
    

mpm_eHIeHI = mpm_eIeI + mpm_eLOeLO - mpm_cross
mpm_eHIeHI = np.real(mpm_eHIeHI)

chis_dense = np.linspace(np.min(_chis), 
                         np.max(_chis), 
                         factor * len(_chis))

D = np.diag(chis_dense)
d_dense = f_d(chis_dense)

L = np.linalg.cholesky(D @ mpm_eHIeHI @ D)
y = np.linalg.solve(L, d_dense)
x = np.linalg.solve(L.T, y)
oup1   = np.einsum('ij, ji->i', d_dense.T, x) / (ClKK + f_N0(ells))
np.save(oup1_fname, oup1)
print('outputted full')

L = np.linalg.cholesky(D @ np.real(mpm_eIeI) @ D)
y = np.linalg.solve(L, d_dense)
x = np.linalg.solve(L.T, y)
oup2 = np.einsum('ij, ji->i', d_dense.T, x) / (ClKK + f_N0(ells))
np.save(oup2_fname, oup2)
print('outputted diag')
