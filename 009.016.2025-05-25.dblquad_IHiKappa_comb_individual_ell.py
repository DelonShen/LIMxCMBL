from LIMxCMBL.init import *
from LIMxCMBL.kernels import *
from LIMxCMBL.noise import *

import sys

Lambda_idx = int(sys.argv[1])
nbins = int(sys.argv[2])
curr_bin = int(sys.argv[3])
Lambda = Lambdas[Lambda_idx]

zmin = float(sys.argv[4])
zmax = float(sys.argv[5])

line_str = sys.argv[6]
ell_idx = int(sys.argv[7])

chimin = ccl.comoving_angular_distance(cosmo, 1/(1+zmin))
chimax = ccl.comoving_angular_distance(cosmo, 1/(1+zmax))
chi_bin_edges = np.linspace(chimin*(1+1e-8), chimax*(1 - 1e-8), nbins + 1)
chi_bin_centers = (chi_bin_edges[1:] + chi_bin_edges[:-1])/2
dchi_binned = np.mean(np.diff(chi_bin_edges))


oup_fname = '/scratch/users/delon/LIMxCMBL/IHiKappa/'
oup_fname += '%s_IHik_zmin_%.1f_zmax_%.1f_idx_%d_dblquad_n_bins_%d_curr_%d_ell_idx_%d.npy'%(line_str,
                                                                                 zmin, 
                                                                                 zmax, 
                                                                                 Lambda_idx, 
                                                                                 nbins, 
                                                                                 curr_bin,
                                                                                           ell_idx)
print(oup_fname)

# get CMB lensing component
from LIMxCMBL.kernels import get_f_Kkappa
f_WkD = get_f_Kkappa()

from LIMxCMBL.cross_spectrum import *
ClKK = d_chib_integral(f_WkD, f_WkD) #[Mpc]^2

kernels = {}
kernels['CII'] = np.array(KI)
kernels['CO'] = np.array(KI_CO)
kernels['Lya'] = np.array(KI_Lya)
kernels['HI'] = np.array(KI_HI)


_KI = kernels[line_str]


from  LIMxCMBL.kernels import *
f_KLIM   = interp1d(x=chis, y = _KI, bounds_error = False, fill_value=0)
f_Kkappa = get_f_Kkappa()

f_KLIM_windowed = apply_window(f_K = f_KLIM,
                               chimin = chimin,
                               chimax = chimax)

from scipy.interpolate import interp1d
from scipy.integrate import quad, quad_vec, trapezoid

inner_dkparp_integral = np.load('/oak/stanford/orgs/kipac/users/delon/LIMxCMBL/inner_dkparp_integral.npy')
inner_dkparp_integral = inner_dkparp_integral.astype(np.float64)


f_inner_integral = interp1d(x = chibs, y = inner_dkparp_integral[ell_idx], axis = 0)

from scipy.interpolate import interp1d, interp2d, LinearNDInterpolator

tmp_chibs = []
tmp_log_deltas = []
tmp_fnctn = []
for i in range(len(chibs)):
    for j in range(len(deltas)):
        tmp_chibs += [chibs[i]]
        tmp_log_deltas += [np.log(deltas[j])]
        tmp_fnctn += [inner_dkparp_integral[ell_idx,i,j]]
        
f_inner_integral_2d = LinearNDInterpolator(list(zip(tmp_chibs, tmp_log_deltas)), tmp_fnctn)

def f_KILo(chi, external_chi, Lambda):
    return (Lambda / np.pi 
            * np.interp(x = chi, xp = chis, 
                         fp = _KI, left = 0, right = 0) 
            * np.sinc(Lambda * (external_chi - chi) / np.pi))
def bin_integrand(_chib, chi):
    
    _curr_KI = 2 * np.interp(x = chi, xp = chis, fp = _KI, left = 0, right = 0)
    
    
    #inner integrand
    #Low passed
    plus = _chib*(1+deltas)
    mins = _chib*(1-deltas)

    _interm  = np.where((plus >= chimin) & (plus <= chimax),
                        f_KILo(plus, external_chi = chi, Lambda=Lambda) 
                        * np.interp(x = mins, xp = chis, fp = Wk * Dz, left = 0, right = 0),
                        0)
    _interm += np.where((mins >= chimin) & (mins <= chimax),
                        f_KILo(mins, external_chi = chi, Lambda=Lambda) 
                        * np.interp(x = plus, xp = chis, fp = Wk * Dz, left = 0, right = 0),
                        0)
    _factor = (2 / _chib)
    _factor = _factor * deltas
    _factor *= f_inner_integral(_chib)

    _interm  *= _factor

    LO_integrand = trapezoid(x = np.log(deltas), y = _interm, axis=-1)

    #unfiltered
    _delta = np.abs(1 - chi / _chib)
    _delta = np.where(_delta < 1e-6, 1e-6,
                     np.where(_delta > 0.7, 
                             0.7,
                             _delta))
    unfiltered_integrand = (_curr_KI 
                            * np.interp(x = 2*_chib - chi, 
                                        xp = chis, fp = Wk * Dz, 
                                        left = 0, right = 0)
                            * f_inner_integral_2d((_chib, np.log(_delta))) 
                            / _chib**2)

    return unfiltered_integrand - LO_integrand

from scipy.integrate import nquad
left = chi_bin_edges[curr_bin]
right = chi_bin_edges[curr_bin + 1]

options={'limit':100000, 'epsabs':0.0, 'epsrel':1e-3}

res, _ = nquad(bin_integrand, [[10,chimax_sample],[left, right]],
               opts=[options,options])

res /= dchi_binned

np.save(oup_fname, res)
print('outputted')
