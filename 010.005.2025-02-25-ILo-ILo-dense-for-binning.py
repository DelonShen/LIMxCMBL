
from LIMxCMBL.init import *
from LIMxCMBL.kernels import *
import sys

Lambda_idx = int(sys.argv[1])
n_external = int(sys.argv[2])
n_chibs = int(sys.argv[3])

Lambda = Lambdas[Lambda_idx]

# CCAT-prime
zmin = 3.5
zmax = 8.1

Omega_field = 8 * (np.pi/180)**2 #rad^2
Pei = 2.3e4 #Mpc^3 kJy^2 /sr^2 
chimin = ccl.comoving_angular_distance(cosmo, 1/(1+zmin))
chimax = ccl.comoving_angular_distance(cosmo, 1/(1+zmax))



oup_fname = '/scratch/users/delon/LIMxCMBL/I_auto/ILo_ILo'
oup_fname += '_zmin_%.1f_zmax_%.1f_Lambda_idx_%d_n_ext_%d_n_chib_%d.npy'%(zmin, 
                                                                          zmax, 
                                                                          Lambda_idx, 
                                                                          n_external, 
                                                                          n_chibs)
print(oup_fname)

from scipy.interpolate import interp1d
from scipy.integrate import quad, quad_vec, trapezoid

inner_dkparp_integral = np.load('/oak/stanford/orgs/kipac/users/delon/LIMxCMBL/inner_dkparp_integral.npy')
f_inner_integral = interp1d(x = chibs, y = inner_dkparp_integral, axis = 1)

external_chis = np.linspace(chimin*(1+1e-8), chimax*(1 - 1e-8), n_external)
print('external chi spacing', np.mean(np.diff(external_chis)))
_chibs = np.linspace(10, chimax_sample, n_chibs)
print('chib spacing', (chimax_sample - 10)/((n_chibs - 1)))

interpolated_inner_integral = f_inner_integral(_chibs).astype(np.float64)
plus = np.einsum('b,d->bd',_chibs, (1+deltas))
mins = np.einsum('b,d->bd',_chibs, (1-deltas))

def compute_elem(params):
    chi_idx, chi, chip_idx, chip = params
    
    f_KLIMLo1 = get_f_KILo(external_chi = chi, Lambda=Lambda)
    f_KLIMLo1_windowed = apply_window(f_K = f_KLIMLo1,
                                     chimin = chimin,
                                     chimax = chimax)
    
    f_KLIMLo2 = get_f_KILo(external_chi = chip, Lambda=Lambda)
    f_KLIMLo2_windowed = apply_window(f_K = f_KLIMLo2,
                                     chimin = chimin,
                                     chimax = chimax)

    

    _interm  = f_KLIMLo1_windowed(plus) * f_KLIMLo2_windowed(mins)
    _interm += f_KLIMLo1_windowed(mins) * f_KLIMLo2_windowed(plus) 

    _factor = (2 / _chibs)
    _factor = np.einsum('b,d->bd', _factor, deltas)
    _factor = np.einsum('bd, lbd->lbd', _factor, interpolated_inner_integral)
    
    _interm  = np.einsum('bd,lbd->lbd', 
                     _interm, _factor)
    
    
    
    d_delta_integral = trapezoid(x = np.log(deltas), y = _interm, axis=-1)

    #(ells)
    ret = trapezoid(x = _chibs, y = d_delta_integral, axis=-1)
    return (chi_idx, chip_idx, ret)

params_list = []
for i in range(n_external):
    chi = external_chis[i]
    for j in range(i, n_external):
        chip = external_chis[j]
        params = (i, chi, j, chip)
        params_list.append(params)


from tqdm import tqdm
from multiprocessing import Pool

with Pool(processes=32) as pool:
        results = list(tqdm(
            pool.imap(compute_elem, params_list),
            total=len(params_list)
        ))


import pickle
with open(oup_fname+'_results.pkl', 'wb') as f:
    print(oup_fname+'_results.pkl')
    pickle.dump(results, f)
print('outputted pickle')


ILoILo = np.zeros((len(ells), (n_external), (n_external)), dtype=np.float64)
for i, j, value in results:
    ILoILo[:,i,j] = ILoILo[:,j,i] = value

np.save(oup_fname, ILoILo)
print('outputted ILoILo')
