import sys
n_runs = int(sys.argv[1])

#tmp patch because I need to prod before binning the II
#and doing things correctly makes things slow
n_runs -= 7560010
n_runs += 2880
#########

oup_mc_fname = '/sdf/scratch/kipac/delon/toy_model_LIMxCMBL/monte_carlo_toy_model_nrun_%d.pkl'%(n_runs)
print(oup_mc_fname)


import matplotlib.pyplot as plt
from LIMxCMBL.toy import *

plt.plot()
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 10,
    "font.family" : "serif",
    'figure.constrained_layout.use':True,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'hatch.linewidth':0.1,
    'figure.figsize': (6, 6/1.618),
    'figure.constrained_layout.use': True,
    'axes.labelsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})


# In[3]:


LC_k = 'magenta'
nLC_color = 'cyan'
nLC_lw = 3
color = {'LC': 'k', 'nLC':'k'}


# In[4]:


fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(3,4.2), dpi=300)
# <Ik> plot
def _plot(x, y, **params):
    axs[0].plot(x, jnp.abs(y/binned_Ik_kappa[0]), **params)
    


_plot(k_bin_centers, binned_Ik_kappa, 
         label=r'\textbf{This Work}', c=LC_k, lw=3)

_plot(k_bin_centers, binned_Ik_kappa_noLC, 
         label='Naive', c=nLC_color, lw=nLC_lw, ls='-')


axs[0].plot([-100, -99, ], [-1000000, -1000000], c='k', ls='--', label=r'\texttt{Monte Carlo}')

axs[0].set_yscale('symlog', linthresh=1e-3)
axs[0].set_ylim(0, 2)

axs[0].set_ylabel(r'$\sim |\langle \textbf{\textsf{LIM}}\times \rm{Lensing}\rangle|$')

# SNR2 Plot
def _plot(x, y, **params):
    axs[1].plot(x, y/SNR2s['theoryLC'][0], **params)

_plot(k_bin_centers, SNR2s['theoryLC'], c=LC_k, lw=3, label=r'\textbf{This Work}')
_plot(k_bin_centers, SNR2s['theorynLC'], c=nLC_color, lw=nLC_lw, label='Naive')

axs[1].errorbar([0, 100], [100, 100], lw=1, c='k', ls='--', label=r'\texttt{Monte Carlo}')

axs[1].set_yscale('symlog', linthresh=1e-2)
axs[1].set_ylim(0, 1e0)

axs[1].set_ylabel(r'$\sim {\sf SNR}^2(\textsf{Filter out}<k)$')
axs[1].set_xlabel(r'$k$ [${\sf Mpc}^{-1}$]')

plt.xscale('asinh', linear_width=1e-3)
plt.xticks([1e-3, 1e-2, 1e-1, 1])
plt.xlim(0, 1.5)

axs[1].legend(frameon=False)


# In[5]:


_idx_bins = jnp.arange(n_k_bins)


# In[6]:
@jax.jit
def get_fields(key):
    white_x = jax.random.normal(key, shape=(len(chis))) / jnp.sqrt(dchi)
    white_k = jnp.fft.fft(white_x) * dchi
    delta_m_k = jnp.sqrt(P1Dk) * white_k
    delta_m_x = jnp.real(jnp.fft.ifft(delta_m_k/dchi))

    kappa = dchi * jnp.sum(Kkappa_x*delta_m_x)
    
    I_x = KI * delta_m_x
    I_k = jnp.fft.fft(I_x) * dchi

    I_x_noLC = KIbar * delta_m_x
    I_k_noLC = jnp.fft.fft(I_x_noLC) * dchi

    IIstar = I_k.reshape(-1, 1) * jnp.conj(I_k).reshape(1, -1)
    IIstar_noLC = I_k_noLC.reshape(-1, 1) * jnp.conj(I_k_noLC).reshape(1, -1)

    

    return (kappa, 
            bin_Ik_vmapped(_idx_bins, I_k), 
            bin_Ik_vmapped(_idx_bins, I_k_noLC), 
            delta_m_k, 
            delta_m_x, 
            bin_cov_vmapped(jnp.arange(n_k_bins), jnp.arange(n_k_bins), IIstar), 
            bin_cov_vmapped(jnp.arange(n_k_bins), jnp.arange(n_k_bins), IIstar_noLC))


@jax.jit
def get_observable(key):
    kappa, I_k, I_k_noLC, delta_m_k, delta_m_x, IIstar, IIstar_noLC = get_fields(key)
    
    return [kappa**2, 
            kappa*I_k, 
            kappa*I_k_noLC, 
            IIstar,
            IIstar_noLC,]

# In[7]:


key = jax.random.key(n_runs)


measured_spectra = {}


process = {
        'kk':     [0, np.array(expected_kappa2), 1],
        'LC_Ik' : [1, np.array(binned_Ik_kappa), n_k_bins],
        'nLC_Ik': [2, np.array(binned_Ik_kappa_noLC), n_k_bins],
        'LC_II':  [3, np.array(expected_binned_II), (n_k_bins, n_k_bins), ],
        'nLC_II': [4, np.array(expected_binned_II_noLC), (n_k_bins, n_k_bins), ],
}

for _type in process:
    measured_spectra[_type] = np.zeros(process[_type][-1], dtype=np.complex128)
    measured_spectra[_type+'_MSE'] = np.zeros(process[_type][-1])


# In[8]:


for run in trange(n_runs):
    new_key, subkey = jax.random.split(key)
    del key
    
    observables = get_observable(subkey)
    del subkey

    key = new_key

    for _type in process:
        _idx, _expected, _shape = process[_type]
        #mean
        measured_spectra[_type] += observables[_idx]
        measured_spectra[_type+'_MSE'] += jnp.real((observables[_idx]-_expected)*jnp.conj(observables[_idx]-_expected))

with open(oup_mc_fname, 'wb') as f:
    pickle.dump(measured_spectra, f)

print('outputted')
