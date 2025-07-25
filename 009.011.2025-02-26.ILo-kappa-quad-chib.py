from LIMxCMBL.init import *
from LIMxCMBL.noise import *

import sys

Lambda_idx = int(sys.argv[1])
n_external = int(sys.argv[2])

Lambda = Lambdas[Lambda_idx]

# CCAT-prime
zmin = 3.5
zmax = 8.1

Omega_field = 8 * (np.pi/180)**2 #rad^2
Pei = 2.3e4 #Mpc^3 kJy^2 /sr^2 
chimin = ccl.comoving_angular_distance(cosmo, 1/(1+zmin))
chimax = ccl.comoving_angular_distance(cosmo, 1/(1+zmax))


oup_fname = '/scratch/users/delon/LIMxCMBL/IHiKappa/ILok_zmin_%.5f_zmax_%.5f_idx_%d_quad_n_ext_%d.npy'%(zmin, zmax, Lambda_idx, n_external)
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


# if no high pass IKappa
from  LIMxCMBL.kernels import *
f_Kkappa = get_f_Kkappa()

external_chis = np.linspace(chimin*(1+1e-8), chimax*(1 - 1e-8), n_external)
print('external chi spacing', np.mean(np.diff(external_chis)))


from scipy.interpolate import interp1d
from scipy.integrate import quad, quad_vec, trapezoid

inner_dkparp_integral = np.load('/oak/stanford/orgs/kipac/users/delon/LIMxCMBL/inner_dkparp_integral.npy')
f_inner_integral = interp1d(x = chibs, y = inner_dkparp_integral.astype(np.float64), axis = 1)

f_KLIMLo = get_f_KILo(external_chi = external_chis.reshape(n_external, 1), Lambda=Lambda)
f_KLIMLo_windowed = apply_window(f_K = f_KLIMLo,
                                 chimin = chimin,
                                 chimax = chimax)

def integrand(_chib):
    plus = _chib*(1+deltas)
    mins = _chib*(1-deltas)

    _interm  = f_KLIMLo_windowed(plus) * f_Kkappa(mins)
    _interm += f_KLIMLo_windowed(mins) * f_Kkappa(plus) 

    _factor = (2 / _chib)
    _factor = _factor * deltas
    _factor = np.einsum('d, ld->ld', _factor, f_inner_integral(_chib))

    _interm  = np.einsum('ed,ld->led', _interm, _factor)

    return trapezoid(x = np.log(deltas), y = _interm, axis=-1)

res, _ = quad_vec(integrand, 10, chimax_sample, epsrel = 1e-3, epsabs =0.0)
np.save(oup_fname, res)
print('outputted')
