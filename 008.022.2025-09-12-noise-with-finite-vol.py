from LIMxCMBL.init import *
from LIMxCMBL.noise import *
import scipy
import sys
Lambda_idx = int(sys.argv[1])
nbins = int(sys.argv[2])

Lambda = Lambdas[Lambda_idx]

zmin = float(sys.argv[3])
zmax = float(sys.argv[4])

chimin = ccl.comoving_angular_distance(cosmo, 1/(1+zmin))
chimax = ccl.comoving_angular_distance(cosmo, 1/(1+zmax))


oup_fname = '/scratch/users/delon/LIMxCMBL/eHIeHI/comb_zmin_%.5f_zmax_%.5f_Lambda_idx_%.d_from_quad_nbins_%d.npy'%(zmin, zmax, Lambda_idx, nbins)
print(oup_fname)


chi_bin_edges = np.linspace(chimin*(1+1e-8), chimax*(1 - 1e-8), nbins + 1)
eHIeHI_binned = np.zeros((nbins,nbins), dtype=np.float64)
dchi = np.mean(np.diff(chi_bin_edges))

def get_mathematica(fname, to_scipy=False):
    _f_mathematica = None
    with open(fname, 'r') as f:
        _f_mathematica = f.read()
    _f_mathematica = _f_mathematica.replace('\[Pi]', 'Pi')
    _f_sympy = parse_mathematica(_f_mathematica)
    return _f_sympy

m_for_LO = 0
while(m_for_LO * 2 * np.pi / (chimax - chimin) < Lambda):
    m_for_LO += 1
m_for_LO -= 1

import os
_f_eLOeLO_tot = None
for a in range(-m_for_LO, m_for_LO+1):
    for b in range(-m_for_LO, m_for_LO+1):
        print(a, b)
        _a = np.abs(a)
        _b = np.abs(b)
        _full_fname =  '008.023.mathematica/008.023.eLOeLO_summand_%d_%d_.txt'%(_a, _b)
        _f = get_mathematica(_full_fname)
        if(_f_eLOeLO_tot is None):
            _f_eLOeLO_tot = _f
        else:
            _f_eLOeLO_tot += _f
print('all loaded')

from scipy import special
def SinIntegral(x):
    si, _ = special.sici(x)
    return si

def CosIntegral(x):
    _, ci = special.sici(x)
    return ci

modules = [
        {
            'Ci': CosIntegral,
            'Si': SinIntegral,
            'Cos': np.cos,
            'Sin': np.sin,
            'Log': np.log,
            'Ei': special.expi,
            'I': 1j,},
        'scipy']

f_eLOeLO_tot = lambdify(list(_f_eLOeLO_tot.free_symbols), _f_eLOeLO_tot, modules=modules)

from scipy.integrate import quad, dblquad, nquad

# DFT low pass
def lo_DFT(alpha):
    _L = chimax - chimin
    #can ignore alpha ~ L due to binning
    if(alpha == 0):
        return (1 + 2 * m_for_LO) / (chimax- chimin)
    return np.sin(np.pi/_L * alpha * (1 + 2 * m_for_LO)) / np.sin(np.pi/_L * alpha) * 1/_L

def compute_bin_element(params):
    i, j, l1, r1, l2, r2, dchi, Lambda, chimin, chimax = params
    eIeI = 0
    if i == j:
        eIeI = 1 / dchi**2 * (1 / l1 - 1 / r1)
    
    
    
    def integrand(x, xp):
        _f_cross = lambda chi, chip : (1/chi**2  * lo_DFT(chi-chip) + 1/chip**2  * lo_DFT(chip-chi))
        return f_eLOeLO_tot(chi = x, chip = xp, chimin = chimin, chimax = chimax) - _f_cross(chi = x, chip = xp)    
        #return f_eLOeLO_tot(chimin = chimin, chimax = chimax) - _f_cross(chi = x, chip = xp) #if Lambda_idx = 8


    options1 = {'limit':100000, 'epsabs':0.0, 'epsrel':1e-8, 
                'points':(l1, (l1+r1)/2, r1)}
    options2 = {'limit':100000, 'epsabs':0.0, 'epsrel':1e-8, 
                'points':(l2, (l2+r2)/2, r2)}
    

    LOLO_m_cross, _ = nquad(integrand, [[l1, r1],[l2, r2]],
                   opts=[options1,options2])
    
    
    LOLO_m_cross = LOLO_m_cross / dchi**2
    
    return (i, j, eIeI + LOLO_m_cross)

params_list = []
for i, (l1, r1) in enumerate(zip(chi_bin_edges[:-1], chi_bin_edges[1:])):
    for j, (l2, r2) in enumerate(zip(chi_bin_edges[:-1], chi_bin_edges[1:])):
        if(j < i):
            continue
        params = (i, j, l1, r1, l2, r2, dchi, Lambda, chimin, chimax)
        params_list.append(params)

with Pool(processes=32) as pool:
        results = list(tqdm(
            pool.imap(compute_bin_element, params_list),
            total=len(params_list)
        ))

import pickle

with open(oup_fname+'_results.pkl', 'wb') as f:
    print(oup_fname+'_results.pkl')
    pickle.dump(results, f)
print('outputted')


for i, j, value in results:
    eHIeHI_binned[i, j] = eHIeHI_binned[j, i] = np.real(value)

np.save(oup_fname, eHIeHI_binned)
print('binned eHIeHI outputed to', oup_fname)
