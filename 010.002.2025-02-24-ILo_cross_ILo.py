from LIMxCMBL.init import *
from LIMxCMBL.kernels import *
from scipy.signal.windows import dpss
import sys

Lambda_idx = int(sys.argv[1])
n_bins = int(sys.argv[2])
n_external = n_bins
n_chibs = int(sys.argv[3])

Lambda = Lambdas[Lambda_idx]

# CCAT-prime
zmin = 3.5
zmax = 8.1

oup_fname = '/scratch/users/delon/LIMxCMBL/I_auto/ILo_ILo_zmin_%.1f_zmax_%.1f_Lambda_idx_%d_n_ext_%d_n_chib_%d.npy'%(zmin, zmax, Lambda_idx, n_external, n_chibs)
print(oup_fname)


Omega_field = 8 * (np.pi/180)**2 #rad^2
Pei = 2.3e4 #Mpc^3 kJy^2 /sr^2 
chimin = ccl.comoving_angular_distance(cosmo, 1/(1+zmin))
chimax = ccl.comoving_angular_distance(cosmo, 1/(1+zmax))

from scipy.interpolate import interp1d

inner_dkparp_integral = np.load('/oak/stanford/orgs/kipac/users/delon/LIMxCMBL/inner_dkparp_integral.npy')
from scipy.integrate import quad, quad_vec, trapezoid
f_inner_integral = interp1d(x = chibs, y = inner_dkparp_integral, axis = 1)

from tqdm import trange

chi_bin_edges = np.linspace(chimin*(1+1e-8), chimax*(1 - 1e-8), n_bins + 1)
chi_bin_centers = (chi_bin_edges[1:] + chi_bin_edges[:-1])/2

external_chis =  chi_bin_centers

_chibs = np.linspace(10, chimax_sample, n_chibs)
print('chib spacing', (chimax_sample - 10)/((n_chibs - 1)))

interpolated_inner_integral = f_inner_integral(_chibs)

ILoAuto = np.zeros((len(ells), n_external, n_external), dtype=np.float64)

for chi_idx in trange(n_external):
    chi = external_chis[chi_idx]
    f_KLIMLo1 = get_f_KILo(external_chi = chi, Lambda=Lambda)
    f_KLIMLo1_windowed = apply_window(f_K = f_KLIMLo1,
                                     chimin = chimin,
                                     chimax = chimax)

    f_KLIMLo2 = get_f_KILo(external_chi = external_chis.reshape(n_external,1,1), Lambda=Lambda)
    f_KLIMLo2_windowed = apply_window(f_K = f_KLIMLo2,
                                     chimin = chimin,
                                     chimax = chimax)


    plus = np.einsum('b,d->bd',_chibs, (1+deltas))
    mins = np.einsum('b,d->bd',_chibs, (1-deltas))
    ret  = f_KLIMLo1_windowed(plus) * f_KLIMLo2_windowed(mins)
    ret += f_KLIMLo1_windowed(mins) * f_KLIMLo2_windowed(plus) 
    factor = (2 / _chibs)

    # so we can integrate dlog delta
    factor = np.einsum('b,d->bd', factor, deltas)
    factor = np.einsum('bd, lbd->lbd', factor, interpolated_inner_integral)

    ret  = np.einsum('pbd,lbd->lpbd', 
                     ret, factor)

    d_delta_integral = trapezoid(x = np.log(deltas), y = ret, axis=-1)
    ILoAuto[:,chi_idx,:] = trapezoid(x = _chibs, y = d_delta_integral, axis=-1)


np.save(oup_fname, ILoAuto)
print('outputted')
