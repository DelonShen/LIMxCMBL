from LIMxCMBL.init import *
from LIMxCMBL.noise import *

import sys
n_external = 1000

zmin = float(sys.argv[1])
zmax = float(sys.argv[2])

line_str = sys.argv[3]

chimin = ccl.comoving_angular_distance(cosmo, 1/(1+zmin))
chimax = ccl.comoving_angular_distance(cosmo, 1/(1+zmax))

oup_fname = '/scratch/users/delon/LIMxCMBL/IHiKappa/Ik_'
oup_fname +='zmin_%.5f_zmax_%.5f_quad_next_%d.npy'%(zmin, 
                                                    zmax, 
                                                    n_external)

# get CMB lensing component
from LIMxCMBL.kernels import get_f_Kkappa
f_WkD = get_f_Kkappa()

from LIMxCMBL.cross_spectrum import *
ClKK = d_chib_integral(f_WkD, f_WkD) #[Mpc]^2


# beam=1.4, noise=7
from scipy.interpolate import interp1d

# if no high pass IKappa
from  LIMxCMBL.kernels import *
f_Kkappa = get_f_Kkappa()


kernels = {}
kernels['CII'] = np.array(KI)
kernels['CO'] = np.array(KI_CO)
kernels['Lya'] = np.array(KI_Lya)
kernels['HI'] = np.array(KI_HI)


_KI = kernels[line_str]



###### NO LC
noLC = False
if(len(sys.argv) > 4 and sys.argv[-1] == 'nLC'):
    noLC = True

if(noLC):
    tmp_idxs = np.where((chis >= chimin) & (chis <= chimax))
    print('putting in no lightcone evolution')
    _KI = np.ones_like(_KI) * np.mean(_KI[tmp_idxs])
    oup_fname = '/scratch/users/delon/LIMxCMBL/IHiKappa/Ik_'
    oup_fname +='zmin_%.5f_zmax_%.5f_quad_next_%d_noLC.npy'%(zmin, 
                                                        zmax, 
                                                        n_external)
##################

print(oup_fname)

from interpax import interp2d, interp1d
import jax.numpy as jnp
from jax import jit

inner_dkparp_integral = np.load('/oak/stanford/orgs/kipac/users/delon/LIMxCMBL/inner_dkparp_integral.npy')
inner_dkparp_integral = inner_dkparp_integral.astype(np.float64)

from scipy.interpolate import interp1d, interp2d, LinearNDInterpolator

tmp_chibs = []
tmp_log_deltas = []
tmp_fnctn = []
for i in range(len(chibs)):
    for j in range(len(deltas)):
        tmp_chibs += [chibs[i]]
        tmp_log_deltas += [np.log(deltas[j])]
        tmp_fnctn += [inner_dkparp_integral[:,i,j]]
        
f_inner_integral = LinearNDInterpolator(list(zip(tmp_chibs, tmp_log_deltas)), tmp_fnctn)

external_chis = np.linspace(chimin*(1+1e-8), chimax*(1 - 1e-8), n_external)

def integrand(chib):
    _delta = np.abs(1 - external_chis / chib)
    _delta = np.where(_delta < 1e-6, 1e-6,
                     np.where(_delta > 0.7, 
                             0.7,
                             _delta))    
    return np.einsum('x,x,xl->lx',
                     2 * np.interp(x = external_chis, xp = chis, fp = _KI, left = 0, right = 0),
                     np.interp(x = 2*chib - external_chis, xp = chis, fp = Wk * Dz, left = 0, right = 0),
                     f_inner_integral((chib, np.log(_delta)))) / chib**2


from scipy.integrate import quad_vec, dblquad, nquad

I_kappa, _ = quad_vec(integrand, 10, chimax_sample,
                      limit = 11234567,
                      epsabs = 0.0,
                      epsrel=1e-3,
                     workers=32)

np.save(oup_fname, I_kappa)
