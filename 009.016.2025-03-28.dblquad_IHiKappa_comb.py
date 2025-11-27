from LIMxCMBL.init import *
from LIMxCMBL.kernels import *
from LIMxCMBL.noise import *

import sys

Lambda_idx = int(sys.argv[1])
nbins = int(sys.argv[2])
curr_bin = int(sys.argv[3])
zmin = float(sys.argv[4])
zmax = float(sys.argv[5])
line_str = sys.argv[6]


Lambda = Lambdas[Lambda_idx]
chimin = ccl.comoving_angular_distance(cosmo, 1/(1+zmin))
chimax = ccl.comoving_angular_distance(cosmo, 1/(1+zmax))

m_for_LO = 0
while(m_for_LO * 2 * np.pi / (chimax - chimin) < Lambda):
    m_for_LO += 1
m_for_LO -= 1
print('m for low pass is', m_for_LO)

chi_bin_edges = np.linspace(chimin*(1+1e-8), chimax*(1 - 1e-8), nbins + 1)
chi_bin_centers = (chi_bin_edges[1:] + chi_bin_edges[:-1])/2
dchi_binned = np.mean(np.diff(chi_bin_edges))


oup_fname = '/scratch/users/delon/LIMxCMBL/IHiKappa/'
oup_fname += '%s_IHik_zmin_%.1f_zmax_%.1f_idx_%d_dblquad_n_bins_%d_curr_%d.npy'%(line_str,
                                                                                 zmin, 
                                                                                 zmax, 
                                                                                 Lambda_idx, 
                                                                                 nbins, 
                                                                                 curr_bin)

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


epsrel = 1e-3

###### SUPERLINEAR
supLin = False
if(len(sys.argv) > 7 and sys.argv[-1] == 'sL'):
    supLin = True

if(supLin):
    tmp_idxs = np.where((chis >= chimin) & (chis <= chimax))
    print('putting in superlinear_evol')
    
    mean_KI = np.mean(_KI[tmp_idxs])
    
    #constraints
    #supLin(chimax) = 0
    #1/L int_chimin^chimax supLin(chi) = mean_KI
    
    _a = (5 * mean_KI)**(1/4)/(chimax - chimin)
    _b = (5 * mean_KI)**(1/4) * chimax / (chimax - chimin)
    _supLin_KI = (_a * chis - _b)**4

    _KI = _supLin_KI
    oup_fname = '/scratch/users/delon/LIMxCMBL/IHiKappa/'
    oup_fname += '%s_IHik_zmin_%.1f_zmax_%.1f_idx_%d_dblquad_n_bins_%d_curr_%d_superlinear.npy'%(line_str,
                                                                                     zmin, 
                                                                                     zmax, 
                                                                                     Lambda_idx, 
                                                                                     nbins, 
                                                                                     curr_bin)

#################

###### NO LC
noLC = False
if(len(sys.argv) > 7 and sys.argv[-1] == 'nLC'):
    noLC = True
    epsrel = 1e-5

if(noLC):
    tmp_idxs = np.where((chis >= chimin) & (chis <= chimax))
    print('putting in no lightcone evolution')
    _KI = np.ones_like(_KI) * np.mean(_KI[tmp_idxs])
    oup_fname = '/scratch/users/delon/LIMxCMBL/IHiKappa/'
    oup_fname += '%s_IHik_zmin_%.1f_zmax_%.1f_idx_%d_dblquad_n_bins_%d_curr_%d_noLC.npy'%(line_str,
                                                                                     zmin, 
                                                                                     zmax, 
                                                                                     Lambda_idx, 
                                                                                     nbins, 
                                                                                     curr_bin)
##################

print(oup_fname)

from  LIMxCMBL.kernels import *
f_KLIM   = interp1d(x=chis, y = _KI, bounds_error = False, fill_value=0)
f_Kkappa = get_f_Kkappa()

from scipy.interpolate import interp1d
from scipy.integrate import quad, quad_vec, trapezoid

inner_dkparp_integral = np.load('/oak/stanford/orgs/kipac/users/delon/LIMxCMBL/inner_dkparp_integral.npy')
inner_dkparp_integral = inner_dkparp_integral.astype(np.float64)

f_inner_integral = interp1d(x = chibs, y = inner_dkparp_integral, axis = 1)

from scipy.interpolate import interp1d, interp2d, LinearNDInterpolator

tmp_chibs = []
tmp_log_deltas = []
tmp_fnctn = []
for i in range(len(chibs)):
    for j in range(len(deltas)):
        tmp_chibs += [chibs[i]]
        tmp_log_deltas += [np.log(deltas[j])]
        tmp_fnctn += [inner_dkparp_integral[:,i,j]]
        
f_inner_integral_2d = LinearNDInterpolator(list(zip(tmp_chibs, tmp_log_deltas)), tmp_fnctn)

from scipy.integrate import quad_vec
left = chi_bin_edges[curr_bin]
right = chi_bin_edges[curr_bin + 1]
print(left, right)


def get_f_KILo(external_chi, chimin, chimax, m):
    return lambda chip : (f_KLIM(chip) 
                          * lo_DFT(alpha = external_chi - chip, 
                                   chimin = chimin,
                                   chimax = chimax,
                                   m = m)
                         )


def bin_integrand(chi):
    f_KLIMLo = get_f_KILo(external_chi = chi, 
                          chimin = chimin,
                          chimax = chimax,
                          m = m_for_LO)
    
    f_KLIMLo_windowed = apply_window(f_K = f_KLIMLo,
                                     chimin = chimin,
                                     chimax = chimax)
    
    _curr_KI = 2 * np.interp(x = chi, xp = chis, fp = _KI, left = 0, right = 0)
    
    def integrand(_chib):
        #Low passed
        plus = _chib*(1+deltas)
        mins = _chib*(1-deltas)

        _interm  = f_KLIMLo_windowed(plus) * f_Kkappa(mins)
        _interm += f_KLIMLo_windowed(mins) * f_Kkappa(plus) 

        _factor = (2 / _chib)
        _factor = _factor * deltas
        _factor = np.einsum('d, ld->ld', _factor, f_inner_integral(_chib))

        _interm  = np.einsum('d,ld->ld', _interm, _factor)

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

    res, _ = quad_vec(integrand, 10, chimax_sample, epsrel = epsrel, epsabs =0.0, 
                      points = (chimin, left, (left+right)/2, right, chimax))
    return res

print('started quadvecs')
res, _ = quad_vec(bin_integrand, left, right, epsabs =0, epsrel=epsrel)
res /= dchi_binned

np.save(oup_fname, res)
print('outputted')
