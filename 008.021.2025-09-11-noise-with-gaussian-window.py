from LIMxCMBL.init import *
from LIMxCMBL.noise import *
from scipy.signal.windows import dpss
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

from scipy.integrate import quad, dblquad, nquad


def compute_bin_element(params):
    i, j, l1, r1, l2, r2, dchi, Lambda, chimin, chimax = params
    sigma = 1 / Lambda
    
    eIeI = 0
    if i == j:
        center = (l1 + r1) / 2
        eIeI = f_eIeI(chi=center, dchi=dchi, Lambda=Lambda)
    
    def integrand(x, xp, xb):
        _f_cross = lambda chi, chip : (1/chi**2  * 1 / np.sqrt(2*np.pi*sigma**2)
                                      * np.exp(- (chi - chip)**2 / 2 / sigma**2)
                                      + 1/chip**2  * 1 / np.sqrt(2*np.pi*sigma**2) 
                                      * np.exp(- (chip - chi)**2 / 2 / sigma**2))
        _f_eLOeLO = lambda chi, chip, chib : (1 / (2 * np.pi * sigma**2) / chib**2
                                                    * np.exp(- (chi  - chib)**2 / 2 / sigma**2)
                                                    * np.exp(- (chip - chib)**2 / 2 / sigma**2))

        return _f_eLOeLO(chi = x, chip = xp, chib = xb) - _f_cross(chi = x, chip = xp) / (chimax_sample * 10 - 10)
    
    options1 = {'limit':100000, 'epsabs':0.0, 'epsrel':1e-3, 
                'points':(chimin, l1, (l1+r1)/2, r1, chimax)}
    options2 = {'limit':100000, 'epsabs':0.0, 'epsrel':1e-3, 
                'points':(chimin, l2, (l2+r2)/2, r2, chimax)}
    options3 = {'limit':100000, 'epsabs':0.0, 'epsrel':1e-3, 
                'points':(chimin, l1, l2, (l1+r1)/2, (l2+r2)/2, r1, r2, chimax)}


    LOLO_m_cross, _ = nquad(integrand, [[l1, r1],[l2, r2],[10, chimax_sample * 10]],
                   opts=[options1,options2, options3])
    LOLO_m_cross = LOLO_m_cross / dchi**2
    
    return (i, j, eIeI + LOLO_m_cross)

params_list = []
for i, (l1, r1) in enumerate(zip(chi_bin_edges[:-1], chi_bin_edges[1:])):
    for j, (l2, r2) in enumerate(zip(chi_bin_edges[:-1], chi_bin_edges[1:])):
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
    eHIeHI_binned[i, j] = np.real(value)

np.save(oup_fname, eHIeHI_binned)
print('binned eHIeHI outputed to', oup_fname)
