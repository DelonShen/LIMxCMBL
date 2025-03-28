import sys

Lambda_idx = int(sys.argv[1])
n_bins = int(sys.argv[2])

idx1 = int(sys.argv[3])
idx2 = int(sys.argv[4])


zmin = float(sys.argv[5])
zmax = float(sys.argv[6])

line_str = sys.argv[7]
print(line_str)

#oup_fname = '/scratch/users/delon/LIMxCMBL/I_auto/comb_'
oup_fname = '/sdf/scratch/kipac/delon/I_auto/comb_'
oup_fname += '%s_zmin_%.1f_zmax_%.1f_Lambda_idx_%d_n_b_%d_%d_%d_jax_qmc.npy'%(line_str,
                                                                                zmin, zmax, 
                                                                                Lambda_idx, 
                                                                                n_bins,idx1, idx2)



print(oup_fname)
import os.path
if(os.path.isfile(oup_fname)):
    print('already computed')
    assert(1==0)



from LIMxCMBL.init import *
from LIMxCMBL.kernels import *

from scipy.interpolate import interp1d, LinearNDInterpolator
from scipy.integrate import quad, quad_vec, trapezoid, qmc_quad

from os.path import isfile
import jax
import jax.numpy as jnp

from jax import config
config.update("jax_enable_x64", True)


Lambda = Lambdas[Lambda_idx]


kernels = {}
kernels['CII'] = np.array(KI)
kernels['CO'] = np.array(KI_CO)
kernels['Lya'] = np.array(KI_Lya)
kernels['HI'] = np.array(KI_HI)


_KI = kernels[line_str]


chimin = ccl.comoving_angular_distance(cosmo, 1/(1+zmin))
chimax = ccl.comoving_angular_distance(cosmo, 1/(1+zmax))

inner_dkparp_integral = np.load('inner_dkparp_integral.npy')
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
def f_auto_integrand(chi, chip, _chib):
    #LoLo
    plus = _chib*(1+deltas.reshape(1, -1))
    mins = _chib*(1-deltas.reshape(1, -1))
    _idxs = (chimin < plus) & (plus < chimax) & (chimin < mins) & (mins < chimax)
    LoLo_integrand  = jnp.where(_idxs,
                               f_KILo(plus, 
                                      external_chi = chi,
                                      Lambda=Lambda) 
                                * f_KILo(mins, 
                                         external_chi = chip,
                                         Lambda=Lambda),
                               0)

    LoLo_integrand += jnp.where(_idxs,
                               f_KILo(mins, 
                                      external_chi = chi,
                                      Lambda=Lambda) 
                                * f_KILo(plus, 
                                         external_chi = chip,
                                         Lambda=Lambda),0)
    LoLo_integrand *= (2 / _chib) * deltas.reshape(1, -1)
    LoLo_integrand = jnp.einsum('pd, pdl->pld', LoLo_integrand,
                                interp1dx(xq = _chib.reshape(-1),x = chibs, 
                                f=inner_dkparp_integral,
                                method='linear',))

    LoLo_integrand = jnp.trapezoid(x = np.log(deltas), y = LoLo_integrand, axis=-1)
    return LoLo_integrand


@jax.jit
def f_cross_integrand(chi, chip, _chib):
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
                                        Lambda=Lambda), 0)
    return cross_integrand


@jax.jit 
def f_unfiltered(chi, chip):
    chi = chi.reshape(-1, 1)
    chip = chip.reshape(-1, 1)

    _delta = jnp.abs((chi - chip) / (chi + chip))
    
    ### bound delta
    _delta = jnp.where(_delta > 0.7, 0.7, _delta)
    _delta = jnp.where(_delta < 1e-6, 1e-6, _delta)
    
    _chib  = (chi + chip) / 2

    return (4/(chi + chip)**2 
           * jnp.interp(x = chi, xp = chis, fp = _KI, left = 0, right = 0)
           * jnp.interp(x = chip, xp = chis, fp = _KI, left = 0, right = 0)
           * interp2d(xq = _chib.reshape(-1), yq=jnp.log(_delta).reshape(-1), 
                           x = chibs, y = jnp.log(deltas), f=inner_dkparp_integral,
                           method='linear',))
@jax.jit
def f_integrand(x):
    chi, chip, _chib = x[0], x[1], x[2]
    chi = chi.reshape(-1, 1)
    chip = chip.reshape(-1, 1)
    _chib = _chib.reshape(-1, 1)

    return  f_auto_integrand(chi, chip, _chib) - (f_cross_integrand(chi, chip, _chib) + f_cross_integrand(chip, chi, _chib))




cross = np.zeros((n_bins, n_bins), dtype=np.float64)


from scipy.stats import qmc
qrng = qmc.Halton(d = 3)


l1, r1 = chi_bin_edges[idx1], chi_bin_edges[idx1+1]
l2, r2 = chi_bin_edges[idx2], chi_bin_edges[idx2+1]

def _rng_spawn(rng, n_children):
    bg = rng._bit_generator
    ss = bg._seed_seq
    child_rngs = [np.random.Generator(type(bg)(child_ss))
                  for child_ss in ss.spawn(n_children)]
    return child_rngs


n_estimates = 2**3
n_points = 2**16
estimates = np.zeros((n_estimates, 100))


rngs = _rng_spawn(qrng.rng, n_estimates)

for i in trange(n_estimates):
    sample = qrng.random(n = n_points)
    sample_bin = sample[:, :-1]

    a = [l1, l2]
    b = [r1, r2]

    _chis, _chips = jnp.array(qmc.scale(sample_bin, a, b)).T

    estimates[i] = jnp.mean(f_unfiltered(_chis, _chips), axis = 0)

    edges = np.concatenate(([10], np.linspace(chimin*.9, chimax*1.1, 64), [chimax_sample]))
    for (l3, r3) in zip(edges, edges[1:]):
        a = np.array([l1, l2, l3, ])
        b = np.array([r1, r2, r3, ])

        #only worry about measure for dchib integral
        #since we want averages in chi/chip bins
        A = r3-l3
        dA = A / n_points

        x = jnp.array(qmc.scale(sample, a, b)).T
        estimates[i] += jnp.sum(f_integrand(x) * dA, axis = 0)
        

    qrng = type(qrng)(seed=rngs[i], **qrng._init_quad)
 
integral = jnp.mean(estimates, axis=0)

#I think the error estimate is no longer exact because samples are correlated if we generate once
#and rescale for each bin
standard_error = jnp.std(estimates, axis = 0, ddof = 1)



np.save(oup_fname, integral)
np.save(oup_fname + 'relerr', standard_error/integral)

print(standard_error/integral)
