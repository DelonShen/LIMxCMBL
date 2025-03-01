from LIMxCMBL.init import *
from LIMxCMBL.kernels import *
from tqdm import trange
import sys

Lambda_idx = int(sys.argv[1])
n_external = int(sys.argv[2])

Lambda = Lambdas[Lambda_idx]
# CCAT-prime
zmin = 3.5
zmax = 8.1


oup_fname = '/scratch/users/delon/LIMxCMBL/I_auto/I_ILo_zmin_%.1f_zmax_%.1f_Lambda_idx_%d_n_ext_%d_quad.npy'%(zmin, zmax, Lambda_idx, n_external)
print(oup_fname)

Omega_field = 8 * (np.pi/180)**2 #rad^2
Pei = 2.3e4 #Mpc^3 kJy^2 /sr^2 
chimin = ccl.comoving_angular_distance(cosmo, 1/(1+zmin))
chimax = ccl.comoving_angular_distance(cosmo, 1/(1+zmax))

from  LIMxCMBL.kernels import *
f_KLIM   = get_f_KI()

f_KLIM_windowed = apply_window(f_K = f_KLIM,
                               chimin = chimin,
                               chimax = chimax)

from scipy.interpolate import interp1d, interp2d, LinearNDInterpolator

inner_dkparp_integral = np.load('/oak/stanford/orgs/kipac/users/delon/LIMxCMBL/inner_dkparp_integral.npy')

from scipy.integrate import quad_vec

tmp_chibs = []
tmp_log_deltas = []
tmp_fnctn = []
for i in range(len(chibs)):
    for j in range(len(deltas)):
        tmp_chibs += [chibs[i]]
        tmp_log_deltas += [np.log10(deltas[j])]
        tmp_fnctn += [inner_dkparp_integral[:,i,j]]
        
f_inner_integral = LinearNDInterpolator(list(zip(tmp_chibs, tmp_log_deltas)), tmp_fnctn)

external_chis     =  np.linspace(chimin*(1+1e-8), chimax*(1 - 1e-8), n_external)
print('external chi spacing', np.mean(np.diff(external_chis)))
cross = np.zeros((100, n_external, n_external), dtype=np.float64)


for chi_idx in trange(n_external):
    chi = external_chis[chi_idx]
    chip = external_chis
    
    f_KLIMLo = get_f_KILo(external_chi = chip, Lambda=Lambda)

    f_KLIMLo_windowed = apply_window(f_K = f_KLIMLo,
                                     chimin = chimin,
                                     chimax = chimax)

    def f_integrand(_chib):
        _delta = np.abs(1 - chi/_chib)
        _delta = min(0.7, max(1e-6, _delta))
        integrand = 2 * f_KLIM_windowed(chi)
        integrand *= f_inner_integral((_chib, np.log10(_delta)))
        integrand /= (_chib**2)
        kernel = f_KLIMLo_windowed((2*_chib - chi))
        integrand = np.einsum('l,p->lp', integrand, kernel)
        return integrand

    ret, _ = quad_vec(f_integrand, (chimin + chi)/2, (chimax + chi)/2,
            epsabs=0.0, epsrel=1e-3, workers = 32)
    cross[:,chi_idx,:] = ret

cross = cross + np.moveaxis(cross, -1, -2) # the two cross terms are just from switching chi and chi'
np.save(oup_fname, cross)
print('ouptutted')
