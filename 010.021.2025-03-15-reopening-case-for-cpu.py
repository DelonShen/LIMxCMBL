
from LIMxCMBL.init import *
from LIMxCMBL.kernels import *

from scipy.interpolate import interp1d, LinearNDInterpolator
from scipy.integrate import quad, quad_vec, trapezoid

import sys
from os.path import isfile
import jax
import jax.numpy as jnp
from interpax import interp2d
from interpax import interp1d as interp1dx

from jax import config
config.update("jax_enable_x64", True)




Lambda_idx = int(sys.argv[1])
n_external = int(sys.argv[2])
ell_idx = int(sys.argv[3])


Lambda = Lambdas[Lambda_idx]

zmin = float(sys.argv[4])
zmax = float(sys.argv[5])

kernels = {}
kernels['CII'] = np.array(KI)
kernels['CO'] = np.array(KI_CO)
kernels['Lya'] = np.array(KI_Lya)
kernels['HI'] = np.array(KI_HI)


line_str = sys.argv[6]
print(line_str)
_KI = kernels[line_str]

oup_fname = '/scratch/users/delon/LIMxCMBL/I_auto/comb_'
oup_fname += '%s_zmin_%.1f_zmax_%.1f_Lambda_idx_%d_n_ext_%d_l_%d_jax_quad.npy'%(line_str,
                                                                                zmin, zmax, 
                                                                                Lambda_idx, 
                                                                                n_external,
                                                                                ell_idx)


print(oup_fname)

chimin = ccl.comoving_angular_distance(cosmo, 1/(1+zmin))
chimax = ccl.comoving_angular_distance(cosmo, 1/(1+zmax))


external_chis = np.linspace(chimin*(1+1e-8), chimax*(1 - 1e-8), n_external)
print('external chi spacing', np.mean(np.diff(external_chis)))

inner_dkparp_integral = np.load('/oak/stanford/orgs/kipac/users/delon/LIMxCMBL/inner_dkparp_integral.npy')
inner_dkparp_integral = inner_dkparp_integral.astype(np.float64)[ell_idx]



@jax.jit
def f_KILo(chi, external_chi, Lambda):
    return (Lambda / jnp.pi 
            * jnp.interp(x = chi, xp = chis, 
                         fp = _KI, left = 0, right = 0) 
            * jnp.sinc(Lambda * (external_chi - chi) / np.pi))


from tqdm import trange
oup = np.zeros((n_external, n_external))
for chi_idx in trange(n_external):
    chi = external_chis[chi_idx]
    chip = external_chis[chi_idx:]
    
    @jax.jit
    def f_integrand(_chib):
        _delta = jnp.abs(1 - chi/_chib) #(1)
        _delta = jnp.where(_delta < 1e-6, 1e-6, 
                         jnp.where(_delta > 0.7, 0.7, _delta))

        _idx = ((chimin <= 2*_chib - chi) 
                & (2*_chib - chi <= chimax)) #(1)

        cross_integrand = (2 * jnp.interp(x = chi, xp = chis, fp = _KI, left = 0, right = 0) 
                           * interp2d(xq = _chib, yq=jnp.log(_delta), 
                               x = chibs, y = jnp.log(deltas), f=inner_dkparp_integral,
                               method='linear',) 
                           / (_chib**2))

        cross_integrand = jnp.where(_idx,
                                   cross_integrand.reshape(-1, 1)
                                   * f_KILo(2*_chib - chi, 
                                            external_chi = chip,
                                            Lambda=Lambda),
                                    0) #(p)
        
        
        _delta = jnp.abs(1 - chip/_chib) #(p)
        _delta = jnp.where(_delta < 1e-6, 1e-6, 
                         jnp.where(_delta > 0.7, 0.7, _delta))

        _idx = ((chimin <= 2*_chib - chip) 
                & (2*_chib - chip <= chimax)) #(p)


        cross_integrand_2 = (2 * jnp.interp(x = chip, xp = chis, fp = _KI, left = 0, right = 0) 
                           * interp2d(xq = _chib, yq=jnp.log(_delta), 
                               x = chibs, y = jnp.log(deltas), f=inner_dkparp_integral,
                               method='linear',) 
                           / (_chib**2))

        cross_integrand_2 = jnp.where(_idx,
                                   cross_integrand_2
                                   * f_KILo(2*_chib - chip, 
                                            external_chi = chi,
                                            Lambda=Lambda),
                                    0)
        
        cross_integrand += cross_integrand_2
        
        #LoLo
        plus = _chib*(1+deltas)
        mins = _chib*(1-deltas)
        _idxs = (chimin < plus) & (plus < chimax) & (chimin < mins) & (mins < chimax)

        LoLo_integrand  = jnp.where(_idxs,
                                   f_KILo(plus, 
                                          external_chi = chi,
                                          Lambda=Lambda) 
                                    * f_KILo(mins, 
                                             external_chi = chip.reshape(-1, 1),
                                             Lambda=Lambda),
                                   0)
        LoLo_integrand += jnp.where(_idxs,
                                   f_KILo(mins, 
                                          external_chi = chi,
                                          Lambda=Lambda) 
                                    * f_KILo(plus, 
                                             external_chi = chip.reshape(-1, 1),
                                             Lambda=Lambda),0)
        LoLo_integrand *= (2 / _chib) #(p,d)
        LoLo_integrand = jnp.einsum('pd,d->pd', LoLo_integrand, deltas)
        LoLo_integrand = jnp.einsum('pd,d->pd', LoLo_integrand, 
                                    interp1dx(xq = _chib,
                                             x = chibs, f=inner_dkparp_integral,
                                             method='linear',))

        LoLo_integrand = jnp.trapezoid(x = np.log(deltas), y = LoLo_integrand, axis=-1)
        return LoLo_integrand - cross_integrand
#     if(chi_idx==0):
#         print(f_integrand((chimin + chimax)/2)[58])
        

    ret, _, info = quad_vec(f_integrand, 10, chimax_sample,
                            points=[(chimin + chi)/2, (chimax + chi)/2],
                            epsabs=0.0, epsrel=1e-3, full_output=True)
    print(info.success, info.status, info.message)
    print('ninters', info.intervals.shape)

    oup[chi_idx,chi_idx:] = ret

for chi_idx in trange(n_external):
    for chip_idx in range(n_external):
        oup[chip_idx,chi_idx] = oup[chi_idx,chip_idx]
np.save(oup_fname, oup)

print('outputted')
