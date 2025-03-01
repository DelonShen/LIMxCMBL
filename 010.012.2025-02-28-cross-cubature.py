from LIMxCMBL.init import *
from LIMxCMBL.kernels import *
import sys

Lambda_idx = int(sys.argv[1])
nbins = int(sys.argv[2])

Lambda = Lambdas[Lambda_idx]

# CCAT-prime
zmin = 3.5
zmax = 8.1

oup_fname = '/scratch/users/delon/LIMxCMBL/I_auto/comb'
oup_fname += '_zmin_%.1f_zmax_%.1f_Lambda_idx_%d_nb_%d_cubature.npy'%(zmin, zmax, Lambda_idx, nbins)
print(oup_fname)

Omega_field = 8 * (np.pi/180)**2 #rad^2
Pei = 2.3e4 #Mpc^3 kJy^2 /sr^2 
chimin = ccl.comoving_angular_distance(cosmo, 1/(1+zmin))
chimax = ccl.comoving_angular_distance(cosmo, 1/(1+zmax))

from scipy.interpolate import interp1d, LinearNDInterpolator
from scipy.integrate import quad, quad_vec, trapezoid, tplquad, cubature

inner_dkparp_integral = np.load('/oak/stanford/orgs/kipac/users/delon/LIMxCMBL/inner_dkparp_integral.npy')
tmp_chibs = []
tmp_log_deltas = []
tmp_fnctn = []
for i in range(len(chibs)):
    for j in range(len(deltas)):
        tmp_chibs += [chibs[i]]
        tmp_log_deltas += [np.log10(deltas[j])]
        tmp_fnctn += [inner_dkparp_integral[:,i,j]]
        
f_inner_integral = LinearNDInterpolator(list(zip(tmp_chibs, tmp_log_deltas)), tmp_fnctn)

chi_bin_edges = np.linspace(chimin*(1+1e-8), chimax*(1 - 1e-8), nbins + 1)
chi_bin_centers = (chi_bin_edges[1:] + chi_bin_edges[:-1])/2
dchi_binned = np.mean(np.diff(chi_bin_edges))


from tqdm import trange, tqdm

def f_cross_integrand(x):
    '''
    x should be of shape (npoints, ndim)
    output should be of shape (npoints, output_dim_1,...)
    '''
    chi, chip, _chib = x[:,0], x[:,1], x[:,2]
    
    #I Lo
    _delta = np.abs(1 - chi/_chib) #(p)
    _delta = np.where(_delta < 1e-6, 1e-6,
                     np.where(_delta > .7, 0.7, _delta)) #(p)
    
    _idx = ((chimin <= 2*_chib - chi) 
            & (2*_chib - chi <= chimax)
            & (chimin <= chi)
            & (chi <= chimax))
    
    _c1 =  np.where(_idx.reshape(-1, 1),
                    np.einsum('p,pl,p,p->pl', 
                       2 * f_KI1D(chi),
                       f_inner_integral([(_b, np.log10(_d)) for _b, _d in zip(_chib,_delta)]),
                       1/(_chib**2),
                       f_KILo(2*_chib - chi, external_chi = chip, Lambda=Lambda)),
                    0)
        
    #Lo I
    _delta = np.abs(1 - chip/_chib) #(p)
    _delta = np.where(_delta < 1e-6, 1e-6,
                     np.where(_delta > .7, 0.7, _delta)) #(p)
    
    _idx = ((chimin <= 2*_chib - chip) 
            & (2*_chib - chip <= chimax)
            & (chimin <= chip)
            & (chip <= chimax))
    
    _c2 =  np.where(_idx.reshape(-1, 1),
                    np.einsum('p,pl,p,p->pl', 
                       2 * f_KI1D(chip),
                       f_inner_integral([(_b, np.log10(_d)) for _b, _d in zip(_chib,_delta)]),
                       1/(_chib**2),
                       f_KILo(2*_chib - chip, external_chi = chi, Lambda=Lambda)),
                    0)

    return _c1 + _c2

cross = np.zeros((100, nbins, nbins), dtype=np.float64)

params_list = []
for i in range(nbins):
    l1, r1 = chi_bin_edges[i], chi_bin_edges[i+1]
    for j in range(i, nbins):
        l2, r2 = chi_bin_edges[j], chi_bin_edges[j+1]
        params = (i, j, l1, r1, l2, r2)
        params_list.append(params)


for i,j,l1,r1,l2,r2 in tqdm(params_list):
    res = cubature(f_cross_integrand, [l1, l2, chimin], [r1, r2, chimax],
                   atol = 0.0,
                   rtol = 1e-3,)
    cross[:,i,j] = res.estimate / dchi_binned**2

 
#def elem(params):
#    i, j, l1, r1, l2, r2 = params
#    res = cubature(f_cross_integrand, [l1, l2, chimin], [r1, r2, chimax],
#                   atol = 0.0,
#                   rtol = 1e-3,)
#    
#    return (i, j, res.estimate / dchi_binned**2)
#with Pool(processes=32) as pool:
#    results = list(tqdm(
#        pool.imap(elem, params_list),
#        total=len(params_list)
#        ))
#
#for i, j, value in results:
#    cross[:,i,j] = value

np.save(oup_fname, cross)
print('outputted')
