from LIMxCMBL.init import *
from LIMxCMBL.noise import *

import sys

Lambda_idx = int(sys.argv[1])
nbins = int(sys.argv[2])

Lambda = Lambdas[Lambda_idx]
# CCAT-prime
zmin = 3.5
zmax = 8.1

Omega_field = 8 * (np.pi/180)**2 #rad^2
Pei = 2.3e4 #Mpc^3 kJy^2 /sr^2 
chimin = ccl.comoving_angular_distance(cosmo, 1/(1+zmin))
chimax = ccl.comoving_angular_distance(cosmo, 1/(1+zmax))

oup_fname = '/scratch/users/delon/LIMxCMBL/IHiKappa/ILok_zmin_%.5f_zmax_%.5f_idx_%d_quad_nbins_%d.npy'%(zmin, zmax, Lambda_idx, nbins)
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


def ILOk_integrand(chi):
    external_chis = np.array([chi]).reshape(-1, 1, 1, 1)
    f_KLIMLo = get_f_KILo(external_chi = external_chis, Lambda=Lambda)
    f_KLIMLo_windowed = apply_window(f_K = f_KLIMLo,
                                     chimin = chimin,
                                     chimax = chimax)
    return d_chib_integral(f_KLIMLo_windowed, f_Kkappa)

chi_bin_edges = np.linspace(chimin*(1+1e-8), chimax*(1 - 1e-8), nbins + 1)
chi_bin_centers = (chi_bin_edges[1:] + chi_bin_edges[:-1])/2
dchi_binned = np.mean(np.diff(chi_bin_edges))

from scipy.integrate import quad_vec
ILo_kappa = np.zeros((len(ells), nbins), dtype = np.float64)
for i, (left, right) in tqdm(enumerate(zip(chi_bin_edges, 
                                      chi_bin_edges[1:])), total=nbins):
    ILo_kappa[:,i], _ = quad_vec(ILOk_integrand, left, right, epsabs =0, epsrel=1e-3)
    ILo_kappa[:,i] /= dchi_binned

np.save(oup_fname, ILo_kappa)
print('ouptuted')
