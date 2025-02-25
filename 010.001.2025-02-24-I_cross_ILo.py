from LIMxCMBL.init import *
from LIMxCMBL.kernels import *
from scipy.signal.windows import dpss
import sys

Lambda_idx = int(sys.argv[1])

n_external = int(sys.argv[2])

n_chibs = int(sys.argv[3])

Lambda = Lambdas[Lambda_idx]
# CCAT-prime
zmin = 3.5
zmax = 8.1

oup_fname = '/scratch/users/delon/LIMxCMBL/I_auto/I_ILo_zmin_%.1f_zmax_%.1f_Lambda_idx_%d_n_ext_%d_n_chib_%d.npy'%(zmin, zmax, Lambda_idx, n_external, n_chibs)
print(oup_fname)

Omega_field = 8 * (np.pi/180)**2 #rad^2
Pei = 2.3e4 #Mpc^3 kJy^2 /sr^2 
chimin = ccl.comoving_angular_distance(cosmo, 1/(1+zmin))
chimax = ccl.comoving_angular_distance(cosmo, 1/(1+zmax))

# get CMB lensing component
from LIMxCMBL.kernels import get_f_Kkappa
f_WkD = get_f_Kkappa()

from LIMxCMBL.cross_spectrum import *
ClKK = d_chib_integral(f_WkD, f_WkD) #[Mpc]^2


# beam=1.4, noise=7
from scipy.interpolate import interp1d
N0_ells = np.logspace(1, np.log10(5000), 500)
with open('LIMxCMBL/N0.npy', 'rb') as f:
    N0 = np.load(f)
    
f_N0 = interp1d(x = N0_ells, y = N0)


# if no high pass IKappa
from  LIMxCMBL.kernels import *
f_KLIM   = get_f_KI()
f_Kkappa = get_f_Kkappa()

f_KLIM_windowed = apply_window(f_K = f_KLIM,
                               chimin = chimin,
                               chimax = chimax)

from scipy.interpolate import interp1d, interp2d, LinearNDInterpolator

inner_dkparp_integral = np.load('/oak/stanford/orgs/kipac/users/delon/LIMxCMBL/inner_dkparp_integral.npy')
from scipy.integrate import quad, quad_vec, trapezoid

tmp_chibs = []
tmp_log_deltas = []
tmp_fnctn = []
for i in range(len(chibs)):
    for j in range(len(deltas)):
        tmp_chibs += [chibs[i]]
        tmp_log_deltas += [np.log10(deltas[j])]
        tmp_fnctn += [inner_dkparp_integral[:,i,j]]
        
f_inner_integral = LinearNDInterpolator(list(zip(tmp_chibs, tmp_log_deltas)), tmp_fnctn)

from tqdm import trange




external_chis     =  np.linspace(chimin*(1+1e-8), chimax*(1 - 1e-8), n_external)
print('external chi spacing', np.mean(np.diff(external_chis)))
cross = np.zeros((100, n_external, n_external), dtype=np.float64)
print('chib spacing', (chimax - chimin)/(2 * (n_chibs - 1)))


for chi_idx in trange(n_external):
    chi = external_chis[chi_idx]
    chip = external_chis
    
    #the KLimLo has a Window(2chib - chi) so we dont need all chibs
    _chibs = np.linspace((chimin + chi)/2, (chimax + chi)/2, n_chibs)

    f_KLIMLo = get_f_KILo(external_chi = chip.reshape(n_external, 1), Lambda=Lambda)
    f_KLIMLo_windowed = apply_window(f_K = f_KLIMLo,
                                     chimin = chimin,
                                     chimax = chimax)

    _deltas = np.abs(1 - chi/_chibs)

    integrand = 2 * f_KLIM_windowed(chi)
    _deltas = np.where(_deltas > 0.7, 0.7, _deltas)
    _deltas = np.where(_deltas < 1e-6, 1e-6, _deltas)
    integrand *= f_inner_integral(list(zip(_chibs, np.log10(_deltas))))
    integrand /= (_chibs**2).reshape(n_chibs, 1)
    kernel = f_KLIMLo_windowed((2*_chibs - chi).reshape(1, n_chibs))
    #b = chib
    #l = ell
    #p = chip
    integrand = np.einsum('bl, pb-> lpb', integrand, kernel)
    cross[:,chi_idx,:] = trapezoid(x = _chibs, y = integrand)

cross = cross + np.moveaxis(cross, -1, -2) # the two cross terms are just from switching chi and chi'
np.save(oup_fname, cross)
print('ouptutted')
