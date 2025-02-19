from LIMxCMBL.init import *
from LIMxCMBL.noise import *
import sys
Lambda_idx = np.int32(sys.argv[1])

Lambda = Lambdas[Lambda_idx]


log2 = 12

# CCAT-prime
zmin = 3.5
zmax = 8.1

Omega_field = 8 * (np.pi/180)**2 #rad^2
Pei = 2.3e4 #Mpc^3 kJy^2 /sr^2 
chimin = ccl.comoving_angular_distance(cosmo, 1/(1+zmin))
chimax = ccl.comoving_angular_distance(cosmo, 1/(1+zmax))

oup_fname = '/scratch/users/delon/LIMxCMBL/eHIeHI/snr_per_mode_scipy_zmin_%.5f_zmax_%.5f_Lambda_idx_%d_log2_%d'%(zmin, zmax, Lambda_idx, log2)
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


# if no high pass IKappa
from  LIMxCMBL.kernels import *
f_KLIM   = get_f_KI()
f_Kkappa = get_f_Kkappa()

f_KLIM_windowed = apply_window(f_K = f_KLIM,
                               chimin = chimin,
                               chimax = chimax)

Ik_fname = '/scratch/users/delon/LIMxCMBL/IHiKappa/Ik_zmin_%.5f_zmax_%.5f_chi_sample_%d.npy'%(zmin, zmax, log2)
I_kappa = np.load(Ik_fname)

external_chis     =  np.linspace(chimin*(1+1e-8), chimax*(1 - 1e-8), 2**log2)
print(external_chis)


#### GET MPM
mpm_external_chis = mpm.linspace(chimin*(1+1e-8), chimax*(1 - 1e-8), 2**log2)

mpm_oup_fname = '/scratch/users/delon/LIMxCMBL/eHIeHI/mpmath_zmin_%.5f_zmax_%.5f_Lambda_%.5e_log2_%d.npy'%(zmin, zmax, Lambda, log2)
print(mpm_oup_fname)
with open(mpm_oup_fname, 'rb') as f:
    print(oup_fname)
    results = pickle.load(f)

mpm_eIeI = np.zeros((2**log2, 2**log2), 
                    dtype = np.complex128)
mpm_eLOeLO = np.zeros_like(mpm_eIeI, dtype = np.complex128)
mpm_cross = np.zeros_like(mpm_eIeI, dtype = np.complex128)
for i, j, _eIeI, _cross, _eLOeLO in tqdm(results):
    mpm_eIeI[i,j] = _eIeI
    mpm_eLOeLO[i,j] = mpm_eLOeLO[j,i] = _eLOeLO
    mpm_cross[i,j] = mpm_cross[j,i] = _cross
 
mpm_eHIeHI = mpm_eIeI + mpm_eLOeLO - mpm_cross
mpm_eHIeHI = np.real(mpm_eHIeHI)
mpm_eIeI = np.real(mpm_eIeI)
###########
print(mpm_eHIeHI)
print(mpm_eIeI)



from scipy.integrate import simpson, trapezoid



ILo_kappa_fname = '/scratch/users/delon/LIMxCMBL/IHiKappa/ILok_zmin_%.5f_zmax_%.5f_Lambda_%.5f_chi_sample_%d.npy'%(zmin, zmax, Lambda, log2)
ILo_kappa = np.load(ILo_kappa_fname)
IHi_kappa = (I_kappa - ILo_kappa).astype(np.float64)
print(IHi_kappa)

D = np.diag(external_chis)
print('computing data vector')
d = D @ IHi_kappa.T
print(d)

#noise_fname_base = '/scratch/users/delon/LIMxCMBL/eHIeHI/scipy_zmin_%.5f_zmax_%.5f_Lambda_%.5e_%d'%(zmin, zmax, Lambda, log2)
#
#eIeI = np.load(noise_fname_base + '_eIeI.npy')
#ecross = np.load(noise_fname_base + '_cross.npy')
#eLOeLO = np.load(noise_fname_base + '_eLOeLO.npy')
#eHIeHI = np.real(eIeI + eLOeLO - ecross)

print('cholesky')
L = np.linalg.cholesky(D @ mpm_eHIeHI @ D)
print('solve')
y = np.linalg.solve(L, d)
x = np.linalg.solve(L.T, y)
oup1   = np.einsum('ij, ji->i', d.T, x) / (ClKK + f_N0(ells))
np.save(oup1_fname, oup1)
print('outputted full')

print('cholesky')
L = np.linalg.cholesky(D @ np.real(mpm_eIeI) @ D)
print('solve')
y = np.linalg.solve(L, d)
x = np.linalg.solve(L.T, y)
oup2 = np.einsum('ij, ji->i', d.T, x) / (ClKK + f_N0(ells))
np.save(oup2_fname, oup2)
print('outputted diag')
