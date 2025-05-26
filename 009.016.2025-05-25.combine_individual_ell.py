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





chimin = ccl.comoving_angular_distance(cosmo, 1/(1+zmin))
chimax = ccl.comoving_angular_distance(cosmo, 1/(1+zmax))
chi_bin_edges = np.linspace(chimin*(1+1e-8), chimax*(1 - 1e-8), nbins + 1)
chi_bin_centers = (chi_bin_edges[1:] + chi_bin_edges[:-1])/2
dchi_binned = np.mean(np.diff(chi_bin_edges))


IHi_kappa = np.zeros((len(ells)))

for ell_idx in range(100):
    oup_fname = '/scratch/users/delon/LIMxCMBL/IHiKappa/'
    oup_fname += '%s_IHik_zmin_%.1f_zmax_%.1f_idx_%d_dblquad_n_bins_%d_curr_%d_ell_idx_%d.npy'%(line_str,
                                                                                     zmin, 
                                                                                     zmax, 
                                                                                     Lambda_idx, 
                                                                                     nbins, 
                                                                                     curr_bin,
                                                                                                ell_idx)

    IHi_kappa[ell_idx] = np.load(oup_fname)

oup_fname = '/scratch/users/delon/LIMxCMBL/IHiKappa/'
oup_fname += '%s_IHik_zmin_%.1f_zmax_%.1f_idx_%d_dblquad_n_bins_%d_curr_%d.npy'%(line_str,
                                                                                 zmin, 
                                                                                 zmax, 
                                                                                 Lambda_idx, 
                                                                                 nbins, 
                                                                                 curr_bin)
np.save(oup_fname, IHi_kappa)
