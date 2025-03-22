from LIMxCMBL.init import *
from LIMxCMBL.kernels import *

from scipy.interpolate import interp1d, LinearNDInterpolator
from scipy.integrate import quad, quad_vec, trapezoid, qmc_quad

import sys
from os.path import isfile
import jax
import jax.numpy as jnp

from jax import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')




Lambda_idx = int(sys.argv[1])
n_bins = int(sys.argv[2])

idx1 = int(sys.argv[3])
idx2 = int(sys.argv[4])

Lambda = Lambdas[Lambda_idx]

zmin = float(sys.argv[5])
zmax = float(sys.argv[6])

kernels = {}
kernels['CII'] = np.array(KI)
kernels['CO'] = np.array(KI_CO)
kernels['Lya'] = np.array(KI_Lya)
kernels['HI'] = np.array(KI_HI)


line_str = sys.argv[7]
print(line_str)
_KI = kernels[line_str]

oup_fname = '/scratch/users/delon/LIMxCMBL/I_auto/cross_'
oup_fname += '%s_zmin_%.1f_zmax_%.1f_Lambda_idx_%d_n_b_%d_%d_%d_jax_qmc.npy'%(line_str,
                                                                                zmin, zmax, 
                                                                                Lambda_idx, 
                                                                                n_bins,idx1, idx2)


print(oup_fname)

chimin = ccl.comoving_angular_distance(cosmo, 1/(1+zmin))
chimax = ccl.comoving_angular_distance(cosmo, 1/(1+zmax))

inner_dkparp_integral = np.load('/oak/stanford/orgs/kipac/users/delon/LIMxCMBL/inner_dkparp_integral.npy')
inner_dkparp_integral = inner_dkparp_integral.astype(np.float64)
inner_dkparp_integral = np.moveaxis(inner_dkparp_integral, 0, -1)

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
    chi, chip, _chib = x[0], x[1], x[2]
    chi = chi.reshape(-1, 1)
    chip = chip.reshape(-1, 1)
    _chib = _chib.reshape(-1, 1)

    _delta = jnp.abs(1 - chi/_chib) #(p)
    _delta = jnp.where(_delta < 1e-6, 1e-6, 
                     jnp.where(_delta > 0.7, 0.7, _delta))

    _idx = ((chimin <= 2*_chib - chi) 
            & (2*_chib - chi <= chimax)) #(p)
    cross_integrand = (2 * jnp.interp(x = chi, xp = chis, fp = _KI, left = 0, right = 0) 
                       * interp2d(xq = _chib.reshape(-1), yq=jnp.log(_delta).reshape(-1), 
                           x = chibs, y = jnp.log(deltas), f=inner_dkparp_integral,
                           method='linear',) 
                       / (_chib**2))
    cross_integrand *= jnp.where(_idx,
                                 f_KILo(2*_chib - chi, 
                                        external_chi = chip,
                                        Lambda=Lambda), 0) #(p)
    return cross_integrand

cross = np.zeros((n_bins, n_bins), dtype=np.float64)


from scipy.stats import qmc
qrng = qmc.Halton(d = 3)


l1, r1 = chi_bin_edges[idx1], chi_bin_edges[idx1+1]
l2, r2 = chi_bin_edges[idx2], chi_bin_edges[idx2+1]

#because KLIM_LO(2chib - chi) has a window function we can bound
a = np.array([l1, l2, (chimin + l1) / 2])
b = np.array([r1, r2, (chimax + r1) / 2])

def _rng_spawn(rng, n_children):
    bg = rng._bit_generator
    ss = bg._seed_seq
    child_rngs = [np.random.Generator(type(bg)(child_ss))
                  for child_ss in ss.spawn(n_children)]
    return child_rngs


n_estimates = 2**3
n_points = 2**22
estimates = np.zeros((n_estimates, 100))


rngs = _rng_spawn(qrng.rng, n_estimates)

A = np.prod(b - a)
dA = A / n_points

for i in trange(n_estimates):
    sample = qrng.random(n = n_points)
    x = jnp.array(qmc.scale(sample, a, b)).T
    integrands = f_integrand(x)
    estimates[i] = np.sum(integrands * dA, axis = 0)

    qrng = type(qrng)(seed=rngs[i], **qrng._init_quad)
    
integral = np.mean(estimates, axis=0)
standard_error = np.std(estimates, axis = 0, ddof = 1)



np.save(oup_fname, integral/dchi_binned**2)
np.save(oup_fname + 'relerr', standard_error/integral)

print(standard_error/integral)
