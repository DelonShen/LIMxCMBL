from LIMxCMBL.init import *
from LIMxCMBL.noise import *

import sys
nbins = int(sys.argv[1])

# CCAT-prime
zmin = 3.5
zmax = 8.1

Omega_field = 8 * (np.pi/180)**2 #rad^2
Pei = 2.3e4 #Mpc^3 kJy^2 /sr^2 
chimin = ccl.comoving_angular_distance(cosmo, 1/(1+zmin))
chimax = ccl.comoving_angular_distance(cosmo, 1/(1+zmax))

oup_fname = '/scratch/users/delon/LIMxCMBL/IHiKappa/Ik_zmin_%.5f_zmax_%.5f_quad_nbins_%d.npy'%(zmin, zmax, nbins)
print(oup_fname)


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

    
plt.plot(ells, ClKK)
plt.plot(ells, f_N0(ells))
plt.loglog()

# if no high pass IKappa
from  LIMxCMBL.kernels import *
f_KLIM   = get_f_KI()
f_Kkappa = get_f_Kkappa()

f_KLIM_windowed = apply_window(f_K = f_KLIM,
                               chimin = chimin,
                               chimax = chimax)

_deltas = deltas.reshape(1, -1)

def Ik_integrand(chi):
    _minus = chi * (1 - _deltas)
    _plus  = chi * (1 + _deltas)

    f_Kpsi = f_KLIM_windowed
    f_Kphi = f_Kkappa

    prefactor = 2 / chi
    kernels = (f_Kpsi(_minus) * f_Kphi(_plus) + f_Kpsi(_plus) * f_Kphi(_minus))
    inner_integral_resampled = f_inner_integral(chi)
    integrand = prefactor*kernels*inner_integral_resampled*deltas_reshaped
    return trapezoid(y=integrand, x=np.log(deltas), axis = -1)

chi_bin_edges = np.linspace(chimin*(1+1e-8), chimax*(1 - 1e-8), nbins + 1)
chi_bin_centers = (chi_bin_edges[1:] + chi_bin_edges[:-1])/2
dchi_binned = np.mean(np.diff(chi_bin_edges))

from scipy.integrate import quad_vec
I_kappa = np.zeros((len(ells), nbins), dtype = np.float64)
for i, (left, right) in tqdm(enumerate(zip(chi_bin_edges, 
                                      chi_bin_edges[1:])), total=nbins):
    I_kappa[:,i], _ = quad_vec(Ik_integrand, left, right, epsabs =0, epsrel=1e-8)
    I_kappa[:,i] /= dchi_binned

np.save(oup_fname, I_kappa)
