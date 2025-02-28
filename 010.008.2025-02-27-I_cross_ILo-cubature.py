from LIMxCMBL.init import *
from LIMxCMBL.kernels import *
from tqdm import trange, tqdm

from multiprocessing import Pool
import sys

Lambda_idx = int(sys.argv[1])
nbins = int(sys.argv[2])



Lambda = Lambdas[Lambda_idx]


# CCAT-prime
zmin = 3.5
zmax = 8.1

Omega_field = 8 * (np.pi/180)**2 #rad^2
Pei = 2.3e4 #Mpc^3 kJy^2 /sr^2 
chimin = ccl.comoving_angular_distance(cosmo, 1/(1+zmin))
chimax = ccl.comoving_angular_distance(cosmo, 1/(1+zmax))

chi_bin_edges = np.linspace(chimin*(1+1e-8), chimax*(1 - 1e-8), nbins + 1)
chi_bin_centers = (chi_bin_edges[1:] + chi_bin_edges[:-1])/2
dchi_binned = np.mean(np.diff(chi_bin_edges))

oup_fname = '/scratch/users/delon/LIMxCMBL/I_auto/I_ILo_zmin_%.1f_zmax_%.1f_Lambda_idx_%d_nbins_%d_cubature.npy'%(zmin, zmax, Lambda_idx, nbins)
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

from scipy.integrate import quad_vec, cubature

tmp_chibs = []
tmp_log_deltas = []
tmp_fnctn = []
for i in range(len(chibs)):
    for j in range(len(deltas)):
        tmp_chibs += [chibs[i]]
        tmp_log_deltas += [np.log10(deltas[j])]
        tmp_fnctn += [inner_dkparp_integral[:,i,j]]
        
f_inner_integral = LinearNDInterpolator(list(zip(tmp_chibs, tmp_log_deltas)), tmp_fnctn)

def integrand(x):
    '''
    x should be of shape (npoints, ndim)
    output should be of shape (npoints, output_dim_1,...)
    '''
    chi, chip, _chib = x[:,0], x[:,1], x[:,2]
    
    f_KLIMLo = get_f_KILo(external_chi = chip, Lambda=Lambda)

    f_KLIMLo_windowed = apply_window(f_K = f_KLIMLo,
                                     chimin = chimin,
                                     chimax = chimax)
    _delta = np.abs(1 - chi/_chib)
    _delta = np.where(_delta < 1e-6, 1e-6,
                     np.where(_delta > 0.7, 0.7,
                              _delta))
    
    integrand = 2 * f_KLIM_windowed(chi)
    integrand = np.einsum('p,pl->pl',
                         integrand,
                         f_inner_integral(list(zip(_chib, np.log10(_delta)))))
    integrand = np.einsum('p,pl->pl', 1/(_chib**2), integrand)
    kernel = f_KLIMLo_windowed((2*_chib - chi))
    return np.einsum('p, pl->pl', kernel, integrand)


cross = np.zeros((100, nbins, nbins), dtype=np.float64)

params_list = []
for i, (l1, r1) in enumerate(zip(chi_bin_edges[:-1], chi_bin_edges[1:])):
    for j, (l2, r2) in enumerate(zip(chi_bin_edges[:-1], chi_bin_edges[1:])):
        params = (i, j, l1, r1, l2, r2)
        params_list.append(params)


def elem(params):
    i, j, l1, r1, l2, r2 = params
    res = cubature(integrand, [l1, l2, (chimin + l1)/2], [r1, r2, (chimax + r1)/2],
                   atol = 0.0,
                   rtol = 1e-3,)
    
    return (i, j, res.estimate / dchi_binned**2)



#for bin1 in trange(nbins):
#    for bin2 in range(nbins):
#        curr_bin = [bin1, bin2]
#        l1, r1 = chi_bin_edges[curr_bin[0]], chi_bin_edges[curr_bin[0]+1]
#        l2, r2 = chi_bin_edges[curr_bin[1]], chi_bin_edges[curr_bin[1]+1]
#        
#        res = cubature(integrand, [l1, l2, 10], [r1, r2, chimax_sample],
#                       atol = 0.0,
#                       rtol = 1e-3,)
#        
#        cross[:,curr_bin[0], curr_bin[1]] = res.estimate

with Pool(processes=32) as pool:
        results = list(tqdm(
            pool.imap(elem, params_list),
            total=len(params_list)
        ))

for i, j, value in results:
    cross[:,i,j] = value

cross = cross + np.moveaxis(cross, -1, -2) # the two cross terms are just from switching chi and chi'
np.save(oup_fname, cross)
print('outputted')
