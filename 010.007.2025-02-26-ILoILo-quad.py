from LIMxCMBL.init import *
from LIMxCMBL.kernels import *
import sys

Lambda_idx = int(sys.argv[1])
n_external = int(sys.argv[2])


Lambda = Lambdas[Lambda_idx]

# CCAT-prime
zmin = 3.5
zmax = 8.1

oup_fname = '/scratch/users/delon/LIMxCMBL/I_auto/ILo_ILo_zmin_%.1f_zmax_%.1f_Lambda_idx_%d_n_ext_%d_quad.npy'%(zmin, zmax, Lambda_idx, n_external)
print(oup_fname)


Omega_field = 8 * (np.pi/180)**2 #rad^2
Pei = 2.3e4 #Mpc^3 kJy^2 /sr^2 
chimin = ccl.comoving_angular_distance(cosmo, 1/(1+zmin))
chimax = ccl.comoving_angular_distance(cosmo, 1/(1+zmax))

from scipy.interpolate import interp1d
from scipy.integrate import quad, quad_vec, trapezoid

inner_dkparp_integral = np.load('/oak/stanford/orgs/kipac/users/delon/LIMxCMBL/inner_dkparp_integral.npy')
f_inner_integral = interp1d(x = chibs, y = inner_dkparp_integral, axis = 1)

external_chis = np.linspace(chimin*(1+1e-8), chimax*(1 - 1e-8), n_external)
print('external chi spacing', np.mean(np.diff(external_chis)))

ILoILo = np.zeros((len(ells), (n_external), (n_external)), dtype=np.float64)

from tqdm import trange

f_KLIMLo1 = get_f_KILo(external_chi = external_chis.reshape(n_external,1,1), Lambda=Lambda)
f_KLIMLo1_windowed = apply_window(f_K = f_KLIMLo1,
                                 chimin = chimin,
                                 chimax = chimax)

f_KLIMLo2 = get_f_KILo(external_chi = external_chis.reshape(1,n_external, 1), Lambda=Lambda)
f_KLIMLo2_windowed = apply_window(f_K = f_KLIMLo2,
                                 chimin = chimin,
                                 chimax = chimax)


def integrand(_chib):
    plus = _chib*(1+deltas)
    mins = _chib*(1-deltas)

    _interm  = f_KLIMLo1_windowed(plus) * f_KLIMLo2_windowed(mins)
    _interm += f_KLIMLo1_windowed(mins) * f_KLIMLo2_windowed(plus) 

    _factor = (2 / _chib)
    _factor = _factor * deltas
    _factor = np.einsum('d, ld->ld', _factor, f_inner_integral(_chib))

    _interm  = np.einsum('xyd,ld->lxyd', _interm, _factor)

    return trapezoid(x = np.log(deltas), y = _interm, axis=-1)
res, _ = quad_vec(integrand, 10, chimax_sample, epsrel = 1e-3, epsabs =0.0)

np.save(oup_fname, res)
print('outputted')
