from LIMxCMBL.init import *
from LIMxCMBL.kernels import *

from scipy.interpolate import interp1d, LinearNDInterpolator
from scipy.integrate import quad, quad_vec, trapezoid, cubature

import sys
from os.path import isfile
import jax
import jax.numpy as jnp

from jax import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')




Lambda_idx = 24#int(sys.argv[1])
n_bins = 100#int(sys.argv[2])
ell_idx = 58#int(sys.argv[3])


Lambda = Lambdas[Lambda_idx]

zmin = 2.3#float(sys.argv[4])
zmax = 3.4#float(sys.argv[5])

kernels = {}
kernels['CII'] = np.array(KI)
kernels['CO'] = np.array(KI_CO)
kernels['Lya'] = np.array(KI_Lya)
kernels['HI'] = np.array(KI_HI)


line_str = 'CO'#sys.argv[6]
print(line_str)
_KI = kernels[line_str]

oup_fname = '/scratch/users/delon/LIMxCMBL/I_auto/comb_'
oup_fname += '%s_zmin_%.1f_zmax_%.1f_Lambda_idx_%d_n_b_%d_l_%d_jax_quad.npy'%(line_str,
                                                                                zmin, zmax, 
                                                                                Lambda_idx, 
                                                                                n_bins,
                                                                                ell_idx)


print(oup_fname)

chimin = ccl.comoving_angular_distance(cosmo, 1/(1+zmin))
chimax = ccl.comoving_angular_distance(cosmo, 1/(1+zmax))

inner_dkparp_integral = np.load('/oak/stanford/orgs/kipac/users/delon/LIMxCMBL/inner_dkparp_integral.npy')
inner_dkparp_integral = inner_dkparp_integral.astype(np.float64)[ell_idx]

@jax.jit
def f_KILo(chi, external_chi, Lambda):
    return (Lambda / jnp.pi 
            * jnp.interp(x = chi, xp = chis, 
                         fp = _KI, left = 0, right = 0) 
            * jnp.sinc(Lambda * (external_chi - chi) / np.pi))


chi_bin_edges = np.linspace(chimin*(1+1e-8), chimax*(1 - 1e-8), n_bins + 1)
chi_bin_centers = (chi_bin_edges[1:] + chi_bin_edges[:-1])/2
dchi_binned = np.mean(np.diff(chi_bin_edges))
from interpax import interp2d
from interpax import interp1d as interp1dx

from tqdm import trange, tqdm

@jax.jit
def f_integrand(x):
    '''
    x should be of shape (npoints, ndim)
    output should be of shape (npoints, output_dim_1,...)
    '''
    chi, chip, _chib = x[:,0], x[:,1], x[:,2]

    _delta = jnp.abs(1 - chi/_chib) #(p)
    _delta = jnp.where(_delta < 1e-6, 1e-6, 
                     jnp.where(_delta > 0.7, 0.7, _delta))

    _idx = ((chimin <= 2*_chib - chi) 
            & (2*_chib - chi <= chimax)) #(p)
    cross_integrand = (2 * jnp.interp(x = chi, xp = chis, fp = _KI, left = 0, right = 0) 
                       * interp2d(xq = _chib, yq=jnp.log(_delta), 
                           x = chibs, y = jnp.log(deltas), f=inner_dkparp_integral,
                           method='linear',) 
                       / (_chib**2))
    cross_integrand *= jnp.where(_idx,
                                 f_KILo(2*_chib - chi, 
                                        external_chi = chip,
                                        Lambda=Lambda), 0) #(p)

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

    cross_integrand_2 *= jnp.where(_idx,
                               f_KILo(2*_chib - chip, 
                                        external_chi = chi,
                                        Lambda=Lambda),0)

    cross_integrand += cross_integrand_2

    #LoLo
    plus = _chib.reshape(-1, 1)*(1+deltas.reshape(1, -1))
    mins = _chib.reshape(-1, 1)*(1-deltas.reshape(1, -1))
    _idxs = (chimin < plus) & (plus < chimax) & (chimin < mins) & (mins < chimax)

    LoLo_integrand  = jnp.where(_idxs,
                               f_KILo(plus, 
                                      external_chi = chi.reshape(-1, 1),
                                      Lambda=Lambda) 
                                * f_KILo(mins, 
                                         external_chi = chip.reshape(-1, 1),
                                         Lambda=Lambda),
                               0)
    LoLo_integrand += jnp.where(_idxs,
                               f_KILo(mins, 
                                      external_chi = chi.reshape(-1, 1),
                                      Lambda=Lambda) 
                                * f_KILo(plus, 
                                         external_chi = chip.reshape(-1, 1),
                                         Lambda=Lambda),0)
    LoLo_integrand *= (2 / _chib.reshape(-1, 1)) 
    LoLo_integrand = jnp.einsum('pd,d->pd', LoLo_integrand, deltas)
    LoLo_integrand = jnp.einsum('pd,pd->pd', LoLo_integrand, 
                                interp1dx(xq = _chib,
                                         x = chibs, f=inner_dkparp_integral,
                                         method='linear',))

    LoLo_integrand = jnp.trapezoid(x = np.log(deltas), y = LoLo_integrand, axis=-1)
    return LoLo_integrand - cross_integrand

oup = np.zeros((n_bins, n_bins), dtype=np.float64)

params_list = []
for i in range(n_bins):
    l1, r1 = chi_bin_edges[i], chi_bin_edges[i+1]
    for j in range(i, n_bins):
        l2, r2 = chi_bin_edges[j], chi_bin_edges[j+1]
        params = (i, j, l1, r1, l2, r2)
        params_list.append(params)

def elem(params):
    i, j, l1, r1, l2, r2 = params
    print('cubaturing')
    res = cubature(f_integrand, [l1, l2, chimin], [r1, r2, chimax],
                   atol = 0.0,
                   rtol = 1e-3,)
    
    return (i, j, res.estimate / dchi_binned**2, res)

for params in tqdm(params_list):
    i, j, _oup, res = elem(params)
    oup[i,j] = oup[j,i] = _oup
    print(res.status, '\n nsubs', res.subdivisions)
