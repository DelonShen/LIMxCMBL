import os
os.environ['JAX_PLATFORMS'] = 'cpu'


import jax
import jax.numpy as jnp

from jax import config
config.update("jax_enable_x64", True)

from LIMxCMBL.init import *
from LIMxCMBL.kernels import *
from LIMxCMBL.experiments import *

from scipy.interpolate import interp1d, LinearNDInterpolator
from scipy.integrate import quad, quad_vec, trapezoid, qmc_quad
from scipy.stats import qmc

from tqdm import trange

_tmp_ks = np.logspace(-10, 5, 100000)
_tmp_Pk = np.zeros_like(_tmp_ks)

for k_idx, k in enumerate(_tmp_ks):
    _tmp_Pk[k_idx] = ccl.linear_matter_power(cosmo, k, 1)
    


_ells = jnp.logspace(1, np.log10(5000), 100)
_ells = _ells.reshape(-1,1)

kernels = {}
kernels['CII'] = np.array(KI)
kernels['CO'] = np.array(KI_CO)
kernels['Lya'] = np.array(KI_Lya)
kernels['HI'] = np.array(KI_HI)

import matplotlib.pyplot as plt


#input
import sys
experiment = sys.argv[1]
Lambda_idx = int(sys.argv[2])
chi_idx = int(sys.argv[3])
#


n_bins = 100
n_estimates = 2**3
n_points = 2**13


if(experiment == 'SPHEREx'):
    n_bins = 15
   
Lambda = Lambdas[Lambda_idx]

oup_fname = '/scratch/users/delon/LIMxCMBL/IHiKappa/'
oup_fname += '%s_LIMBER_IHik_idx_%d_chibin_%d_nbins_%d.npy'%(experiment, Lambda_idx, chi_idx, n_bins)

zmin = experiments[experiment]['zmin']
zmax = experiments[experiment]['zmax']

line_str = experiments[experiment]['line_str']


chimin = ccl.comoving_angular_distance(cosmo, 1/(1+zmin))
chimax = ccl.comoving_angular_distance(cosmo, 1/(1+zmax))


chi_bin_edges = np.linspace(chimin*(1+1e-8), chimax*(1 - 1e-8), n_bins + 1)
chi_bin_centers = (chi_bin_edges[1:] + chi_bin_edges[:-1])/2
dchi_binned = np.mean(np.diff(chi_bin_edges))

m_for_LO = 0
if(Lambda_idx > 0):
    while(m_for_LO * 2 * jnp.pi / (chimax - chimin) < Lambda):
        m_for_LO += 1
    m_for_LO -= 1
    print('m for low pass is', m_for_LO)

@jax.jit
def jax_lo_DFT(alpha, m):
    _L = chimax - chimin
    return jnp.where(
        (jnp.abs((jnp.abs(alpha)-_L) / _L) < 1e-5) | (jnp.abs(alpha / _L) < 1e-5),
        (1 + 2 * m) / _L,
        jnp.sin(jnp.pi/_L * alpha * (1 + 2 * m)) / jnp.sin(jnp.pi/_L * alpha) * 1/_L
    )

Lambda = 0.0
if(Lambda_idx >= 0):
    Lambda = Lambdas[Lambda_idx]

__KI = kernels[line_str]
_KI = np.where(((chis >= chimin) & (chis <= chimax)), __KI, 0)

@jax.jit
def f_unfiltered_integrand(chi):
    return (1/chi**2 
            * jnp.interp(x = chi, xp = chis, fp = _KI, left = 0, right = 0)
            * jnp.interp(x = chi, xp = chis, fp = Wk*Dz, left = 0, right = 0)
            * jnp.interp(x = jnp.log((_ells+1/2)/chi), 
                         xp = jnp.log(_tmp_ks), 
                         fp = _tmp_Pk, left = 0, right = 0))

@jax.jit
def _f_filtered_integrand(chi, _chib):
    return (1/_chib**2 
            * jnp.interp(x = _chib, xp = chis, fp = _KI, left = 0, right = 0)
            * jax_lo_DFT(alpha = chi - _chib,  m = m_for_LO)
            * jnp.interp(x = _chib, xp = chis, fp = Wk*Dz, left = 0, right = 0)
            * jnp.interp(x = jnp.log((_ells+1/2)/_chib), 
                         xp = jnp.log(_tmp_ks), 
                         fp = _tmp_Pk, left = 0, right = 0))

@jax.jit
def f_filtered_integrand(x):
    chi, _chib = x[0], x[1]

    chi = chi.reshape(1, -1)
    _chib = _chib.reshape(1, -1)

    return  _f_filtered_integrand(chi=chi, _chib=_chib)

#do integral
qrng = qmc.Halton(d = 2)

l, r = chi_bin_edges[chi_idx], chi_bin_edges[chi_idx+1]

def _rng_spawn(rng, n_children):
    bg = rng._bit_generator
    ss = bg._seed_seq
    child_rngs = [np.random.Generator(type(bg)(child_ss))
                  for child_ss in ss.spawn(n_children)]
    return child_rngs

estimates = np.zeros((n_estimates, 100))

rngs = _rng_spawn(qrng.rng, n_estimates)

for i in range(n_estimates):
    sample = qrng.random(n = n_points)
    sample_bin = sample[:, -1]

    _chis = qmc.scale(jnp.array([sample_bin]), l, r)
    estimates[i] = jnp.mean(f_unfiltered_integrand(_chis), axis = -1)

    if(Lambda_idx > 0):
        a = np.array([l, chimin,])
        b = np.array([r, chimax,])

        #only worry about measure for dchib integral
        #since we want averages in chi bins
        dA = (chimax - chimin) / n_points

        x = jnp.array(qmc.scale(sample, a, b)).T
        estimates[i] -= jnp.sum(f_filtered_integrand(x) * dA, axis = -1)


    qrng = type(qrng)(seed=rngs[i], **qrng._init_quad)
integral = jnp.mean(estimates, axis=0)
standard_error = jnp.std(estimates, axis = 0, ddof = 1)
print(standard_error/integral)
    
jnp.save(oup_fname, integral)
print(oup_fname)
