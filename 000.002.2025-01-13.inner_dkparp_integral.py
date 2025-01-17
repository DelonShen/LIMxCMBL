import sys

ell_idx = eval(sys.argv[1])
print('ell_idx', ell_idx)

from LIMxCMBL.init import *
from tqdm import trange

ell_curr = ells[ell_idx]
oup_fname = '/scratch/users/delon/LIMxCMBL/dkparp_integral/ell_%.8f.npy'%(ell_curr)
print('outputting to', oup_fname)

print('chib bounds', min(chibs), max(chibs))
print('delta bounds', min(deltas), max(deltas))
_chibs, _deltas = np.meshgrid(chibs, deltas, indexing='ij')
print('oup dimension', _chibs.shape)

_chibs  = np.reshape(_chibs,  (len(chibs) * len(deltas)))
_deltas = np.reshape(_deltas, (len(chibs) * len(deltas)))

print('geometric recalibration')
kperp2s = ell_curr*(ell_curr+1) / (_chibs**2 * (1 - _deltas**2))

from scipy.integrate import quad, quad_vec
from scipy.interpolate import interp1d

for i in range(len(chibs)):
    for j in range(len(deltas)):
        assert(np.abs(kperp2s[i*len(deltas)+j] - ell_curr*(ell_curr+1) / (chibs[i]**2 * (1 - deltas[j]**2))) < 1e-8)

def integrand(kparp):
    return 2/(2*np.pi) * np.cos(kparp * 2 * _chibs * _deltas) * ccl.linear_matter_power(cosmo, np.sqrt(kparp**2 + kperp2s), 1)

print('beginning quad_vec')

res, err = quad_vec(integrand, 0, np.inf,
                  epsabs = 0.0, epsrel=1e-4, limit=1123456, workers=32)

oup = np.reshape(res, (len(chibs), len(deltas)))


np.save(oup_fname, oup)
print('outputted to', oup_fname)
