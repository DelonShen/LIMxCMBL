from LIMxCMBL.init import *
from LIMxCMBL.noise import *
from scipy.signal.windows import dpss
import sys
Lambda_idx = int(sys.argv[1])
nbins = int(sys.argv[2])

Lambda = Lambdas[Lambda_idx]

zmin = 5
zmax = 8

chimin = ccl.comoving_angular_distance(cosmo, 1/(1+zmin))
chimax = ccl.comoving_angular_distance(cosmo, 1/(1+zmax))

# get CMB lensing component
from LIMxCMBL.kernels import get_f_Kkappa
f_WkD = get_f_Kkappa()

from LIMxCMBL.cross_spectrum import *
ClKK = d_chib_integral(f_WkD, f_WkD) #[Mpc]^2


# beam=1.4, noise=7
from scipy.interpolate import interp1d
N0_ells = np.logspace(1, np.log10(5000), 500)
with open('LIMxCMBL/N0.npy', 'rb') as f:
    N0 = np.load(f)
    
f_N0 = interp1d(x = N0_ells, y = N0)


# if no high pass IKappa
from  LIMxCMBL.kernels import *
f_KLIM   = get_f_KI()
f_Kkappa = get_f_Kkappa()

f_KLIM_windowed = apply_window(f_K = f_KLIM,
                               chimin = chimin,
                               chimax = chimax)


oup_fname = '/scratch/users/delon/LIMxCMBL/eHIeHI/mpmath_comb_zmin_%.5f_zmax_%.5f_Lambda_idx_%.d_from_quad_nbins_%d.npy'%(zmin, zmax, Lambda_idx, nbins)
print(oup_fname)


chi_bin_edges = np.linspace(chimin*(1+1e-8), chimax*(1 - 1e-8), nbins + 1)
eHIeHI_binned = np.zeros((nbins,nbins), dtype=np.float64)
dchi = np.mean(np.diff(chi_bin_edges))

from scipy.integrate import quad, dblquad


def compute_bin_element(params):
    i, j, l1, r1, l2, r2, dchi, Lambda, chimin, chimax = params
    
    eIeI = 0
    if i == j:
        center = (l1 + r1) / 2
        eIeI = f_eIeI(chi=center, dchi=dchi, Lambda=Lambda)
    
    integrand = lambda x, xp: (f_eLOeLO(chi=x,
                                        chip=xp,
                                        chimin=chimin,
                                        chimax=chimax,
                                        Lambda=Lambda)
                               - f_cross_mpm(x, xp, Lambda=Lambda))

    LOLO_m_cross = mpm.quad(integrand, [l1,r1], [l2,r2])
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

print(results[0])
import pickle
with open(oup_fname+'_results.pkl', 'wb') as f:
    print(oup_fname+'_results.pkl')
    pickle.dump(results, f)
print('outputted')


for i, j, value in results:
        eHIeHI_binned[i, j] = np.real(value)

np.save(oup_fname, eHIeHI_binned)
print('binned eHIeHI outputed to', oup_fname)
