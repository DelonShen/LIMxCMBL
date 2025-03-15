from LIMxCMBL.init import *
from LIMxCMBL.kernels import *

from scipy.interpolate import interp1d, LinearNDInterpolator
from scipy.integrate import quad, quad_vec, trapezoid

import jax
import jax.numpy as jnp

from interpax import interp2d, interp1d
from quadax import quadcc, quadgk

import sys
from os.path import isfile

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


#if(isfile(oup_fname)):
#    print('already computed')
#    assert(1==0)

print(oup_fname)

chimin = ccl.comoving_angular_distance(cosmo, 1/(1+zmin))
chimax = ccl.comoving_angular_distance(cosmo, 1/(1+zmax))


external_chis = np.linspace(chimin*(1+1e-8), chimax*(1 - 1e-8), n_external)
print('external chi spacing', np.mean(np.diff(external_chis)))

inner_dkparp_integral = np.load('/oak/stanford/orgs/kipac/users/delon/LIMxCMBL/inner_dkparp_integral.npy')
       
@jax.jit
def f_KILo(chi, external_chi, Lambda):
    return (Lambda / jnp.pi 
            * jnp.interp(x = chi, xp = chis, 
                         fp = _KI, left = 0, right = 0) 
            * jnp.sinc(Lambda * (external_chi - chi) / np.pi))

inner_dkparp_integral = jnp.array(inner_dkparp_integral.astype(np.float64))

@jax.jit
def f_integrand(_chib, ell_idx):
    #by construction chimin < exteranl_chis < chimax 
    #I Lo + Lo I
    
    _delta = jnp.abs(1 - external_chis/_chib) #(n_ext)
    _delta = jnp.where(_delta < 1e-6, 1e-6, 
                     jnp.where(_delta > 0.7, 0.7, _delta))
    
    _idx = ((chimin <= 2*_chib - external_chis) 
            & (2*_chib - external_chis <= chimax)) #(n_ext)
    
    cross_integrand = (2 * jnp.interp(x = external_chis, xp = chis, fp = _KI, left = 0, right = 0) 
                       * interp2d(xq = _chib, yq=jnp.log(_delta), 
                           x = chibs, y = jnp.log(deltas), f=inner_dkparp_integral[ell_idx],
                           method='linear',) 
                       / (_chib**2))
    
    cross_integrand = jnp.where(_idx.reshape(-1, 1),
                               cross_integrand.reshape(-1, 1)
                               * f_KILo(2*_chib - external_chis.reshape(-1,1), 
                                        external_chi = external_chis.reshape(1, -1), 
                                        Lambda=Lambda),
                                0)
    cross_integrand = cross_integrand + jnp.moveaxis(cross_integrand, -1, -2) # the two cross terms are just from switching chi and chi'
    #LoLo
    plus = _chib*(1+deltas)
    mins = _chib*(1-deltas)
    _idxs = (chimin < plus) & (plus < chimax) & (chimin < mins) & (mins < chimax)
    
    LoLo_integrand  = jnp.where(_idxs,
                               f_KILo(plus, 
                                      external_chi = external_chis.reshape(-1, 1, 1), 
                                      Lambda=Lambda) 
                                * f_KILo(mins, 
                                         external_chi = external_chis.reshape(1, -1, 1), 
                                         Lambda=Lambda),
                               0)
    LoLo_integrand += jnp.where(_idxs,
                               f_KILo(mins, 
                                      external_chi = external_chis.reshape(-1, 1, 1), 
                                      Lambda=Lambda) 
                                * f_KILo(plus, 
                                         external_chi = external_chis.reshape(1, -1, 1), 
                                         Lambda=Lambda),0)
    LoLo_integrand *= (2 / _chib) #(x,y,d)
    LoLo_integrand = jnp.einsum('xyd,d->xyd', LoLo_integrand, deltas)
    LoLo_integrand = jnp.einsum('xyd,d->xyd', LoLo_integrand, 
                                interp1d(xq = _chib,
                                         x = chibs, f=inner_dkparp_integral[ell_idx],
                                         method='linear',))
    
    LoLo_integrand = jnp.trapezoid(x = np.log(deltas), y = LoLo_integrand, axis=-1)
    return LoLo_integrand - cross_integrand

f_integrand((chimin + chimax)/2, ell_idx = ell_idx)

res, tmp = quadgk(f_integrand, jnp.hstack([10, 
                                           jnp.linspace(chimin, chimax, 500), 
                                           chimax_sample]),
                  epsabs = 0.0, epsrel = 1e-5, 
                  order = 31, max_ninter=600, args=(ell_idx,),
                  full_output = True)

np.save(oup_fname, res)
print('outputted')
