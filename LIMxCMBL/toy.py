import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import pyccl as ccl 

from tqdm import trange, tqdm
import pickle

# cosmology
h=0.6736
omch2 = 0.1200
ombh2 = 0.02237
Omc = omch2/h**2
Omb = ombh2/h**2
ns = 0.9649
sigma8 =  0.8111

cosmo = ccl.Cosmology(Omega_c=Omc, Omega_b=Omb,
                          h=h, n_s=ns, sigma8=sigma8)

zstar = 1100
cmbk = ccl.CMBLensingTracer(cosmo, z_source=zstar)

# sampling range
zmax_sample = zstar
amax_sample = 1/(zmax_sample+1)
chimax_sample = ccl.comoving_radial_distance(cosmo, amax_sample)

chis = np.linspace(0, chimax_sample, 10**4)
Wk = cmbk.get_kernel(chis)[0]

a_s = ccl.scale_factor_of_chi(cosmo, chis)
zs = 1/a_s - 1

Dz = ccl.growth_factor(cosmo, a_s)


L = chimax_sample
dchi = jnp.mean(jnp.diff(chis))
ks = 2*jnp.pi*jnp.fft.fftfreq(len(chis), d=dchi)
sort_idx = jnp.argsort(ks)

# power spectrum
P1Dk = ccl.linear_matter_power(cosmo, np.abs(ks), 1)
P1Dk[0] = 0.0 #fix mean
P1Dk = jnp.array(P1Dk)

# set up log space k bins
n_k_bins = 50
k_bin_edges = np.hstack([[0], np.logspace(np.log10(8*jnp.pi/L), np.log10(jnp.pi/dchi), n_k_bins)])
k_bin_edges = jnp.array(k_bin_edges)
k_bin_centers = jnp.sqrt(k_bin_edges[1:]*k_bin_edges[:-1])
bin_indices = jnp.digitize((ks), k_bin_edges) - 1

def count_for_bin(bin_idx):
        return jnp.sum(jnp.where(bin_indices == bin_idx, 1, 0))
bin_counts = jax.vmap(count_for_bin)(jnp.arange(n_k_bins))

# get rid of empty bins

empty_bins = (bin_counts == 0)
non_empty_bins = ~empty_bins
new_k_bin_edges = [k_bin_edges[0]]

new_bin_idx = 0

for i in range(n_k_bins):
    if(non_empty_bins[i]):
        new_k_bin_edges.append(k_bin_edges[i+1])
        new_bin_idx += 1
        
k_bin_edges = jnp.array(new_k_bin_edges)
k_bin_centers = jnp.sqrt(k_bin_edges[1:]*k_bin_edges[:-1])

bin_indices = jnp.digitize((ks), k_bin_edges) - 1
n_k_bins = len(k_bin_edges) - 1
bin_counts = jax.vmap(count_for_bin)(jnp.arange(n_k_bins))

# kernels in fourier space
Kkappa_x = jnp.array(Wk*Dz)
Kkappa_k = jnp.fft.fft(Kkappa_x) * dchi #dchi to put numpy convention to my convention

KI = (chis-chis[-1])**2 #just a parabola that resembles our LIMs kernel
KI_k = jnp.fft.fft(KI) * dchi
KIbar = jnp.mean(KI) #factor chosen so that large scale <I kappa> roughly same

# # which filtering to look at for fully projected case
Lambdas = k_bin_edges

@jax.jit
def bin_Ik(bin_idx, spectra):
    mask = (bin_indices == bin_idx)
    return jnp.sum(jnp.where(mask, spectra, 0.0))/jnp.sum(mask)
bin_Ik_vmapped = jax.vmap(bin_Ik, in_axes=(0, None))


masks = jnp.array([(i == bin_indices) for i in range(n_k_bins)])
@jax.jit
def bin_cov(idx0, idx1, _estimated_cov):
    mask = masks[idx0].reshape(-1, 1) * masks[idx1].reshape(1, -1)
    return jnp.sum(jnp.where(mask, _estimated_cov, 0.0)) / jnp.maximum(1, jnp.sum(mask))

bin_cov_vmapped = jax.vmap(
    jax.vmap(
        bin_cov,
        in_axes=(None, 0, None)
    ),
    in_axes=(0, None, None)
)

# Theory spectra
_ks = ks[sort_idx]
_ks = jnp.hstack([_ks, jnp.abs(_ks[0])])

_KI_k = KI_k[sort_idx]
_KI_k = jnp.hstack([_KI_k, jnp.conj(_KI_k[0])])

period = 2 * jnp.pi / dchi

@jax.jit
def f_KI_k(k):
    return jnp.interp(
            x=k,
            xp=_ks,
            fp=_KI_k,
            period=period)



expected_Ik_kappa = 1/L * jnp.sum((f_KI_k(ks.reshape(-1, 1) - ks.reshape(1, -1))
                                   * (Kkappa_k*P1Dk).reshape(1, -1)), 
                                  axis = -1)
expected_Ik_kappa_noLC = KIbar*Kkappa_k*P1Dk

expected_kappa2 = jnp.real(1/L * jnp.sum(Kkappa_k * jnp.conj(Kkappa_k) * P1Dk))
binned_Ik_kappa =      bin_Ik_vmapped(jnp.arange(n_k_bins), expected_Ik_kappa)
binned_Ik_kappa_noLC = bin_Ik_vmapped(jnp.arange(n_k_bins), expected_Ik_kappa_noLC)

# <I I*>
@jax.jit
def compute_element(k, kp):
    result = jnp.sum(f_KI_k(k-ks) * jnp.conj(f_KI_k(kp - ks)) * P1Dk)
    return result / L
    
expected_II = jax.vmap(
    jax.vmap(
        compute_element, 
        in_axes=(None, 0)
    ),
    in_axes=(0, None)
)(ks, ks)
cov = expected_kappa2 * expected_II + expected_Ik_kappa.reshape(-1, 1) * jnp.conj(expected_Ik_kappa).reshape(1, -1)


expected_II_noLC = L * jnp.diag(KIbar**2 * P1Dk)
cov_noLC = (expected_kappa2 * expected_II_noLC 
            + expected_Ik_kappa_noLC.reshape(-1, 1) 
            * jnp.conj(expected_Ik_kappa_noLC).reshape(1, -1))

#expected_binned_II = jnp.load('data/toy_expected_binned_II.npy')
#expected_binned_cov = expected_binned_II * expected_kappa2 + binned_Ik_kappa.reshape(-1, 1) * jnp.conj(binned_Ik_kappa).reshape(1, -1)

expected_binned_II = bin_cov_vmapped(jnp.arange(n_k_bins), jnp.arange(n_k_bins), expected_II)
expected_binned_cov = bin_cov_vmapped(jnp.arange(n_k_bins), jnp.arange(n_k_bins), cov)

expected_binned_II_noLC = bin_cov_vmapped(jnp.arange(n_k_bins), jnp.arange(n_k_bins), expected_II_noLC)
expected_binned_cov_noLC = bin_cov_vmapped(jnp.arange(n_k_bins), jnp.arange(n_k_bins), cov_noLC)


# SNR theory
SNR2s = {'LC':np.zeros(n_k_bins, dtype=np.complex128), 
         'nLC': np.zeros(n_k_bins, dtype=np.complex128),
         'theoryLC':np.zeros(n_k_bins, dtype=np.complex128), 
         'theorynLC': np.zeros(n_k_bins, dtype=np.complex128),
         'proj': np.zeros_like(Lambdas)}



theory_covs = {'LC': expected_binned_cov, 'nLC': expected_binned_cov_noLC}
theory_Ik = {'LC': binned_Ik_kappa, 'nLC': binned_Ik_kappa_noLC}

for _type in ['LC', 'nLC']:
    for kill_idx in trange(n_k_bins):
        #throw out k <= k_bin_edge[kill_idx]
        x,resid,rank,s = jnp.linalg.lstsq(theory_covs[_type][kill_idx:, kill_idx:], 
                                          jnp.conj(theory_Ik[_type][kill_idx:]))
        SNR2s['theory'+_type][kill_idx] = (theory_Ik[_type][kill_idx:] @ x)

# projected
def process_projected(Lambda):
    mask = (jnp.abs(ks) >= Lambda)
    _expected_Ik_projected = 1/L * jnp.sum(jnp.where(mask, 
                                                     (jnp.conj(Kkappa_k) * KI_k * P1Dk), 
                                                     0))
    _expected_II_projected = 1/L * jnp.sum(jnp.where(mask, 
                                                     (jnp.conj(KI_k) * KI_k * P1Dk), 
                                                     0))

    return jnp.array([_expected_Ik_projected, _expected_II_projected])
expected_projected = jnp.zeros((len(Lambdas),2))
expected_projected = jax.vmap(process_projected)(Lambdas)


for LIDX in trange(len(Lambdas)):
    _expected_var_proj = (expected_projected[LIDX,0]*jnp.conj(expected_projected[LIDX,0]) 
                          + expected_kappa2 * expected_projected[LIDX,1])
    SNR2s['proj'][LIDX] = jnp.real(expected_projected[LIDX,0]*jnp.conj(expected_projected[LIDX,0]) / _expected_var_proj)
