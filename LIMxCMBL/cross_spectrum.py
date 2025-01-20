from LIMxCMBL.init import *
from scipy.interpolate import interp1d

inner_dkparp_integral = np.load('/oak/stanford/orgs/kipac/users/delon/LIMxCMBL/inner_dkparp_integral.npy')
reversed_inner_dkparp_integral = inner_dkparp_integral[:, :, ::-1]
middle_inner_dkpar_integral = inner_dkparp_integral[:, :, 0:1]


deltas_full = np.concatenate((-deltas[::-1], 
                              [0], 
                              deltas))

from scipy.integrate import quad, quad_vec, trapezoid

chibs_reshaped = chibs.reshape(1, -1, 1)
deltas_reshaped = deltas_full.reshape(1, 1, -1)

psi_args = chibs_reshaped * (1 - deltas_reshaped)
phi_args = chibs_reshaped * (1 + deltas_reshaped)


def d_delta_integral(f_Kpsi, f_Kphi):

    f_psi_term = f_Kpsi(psi_args)
    f_phi_term = f_Kphi(phi_args)


    inner_dkarp_integral_full = np.concatenate(
        [reversed_inner_dkparp_integral, middle_inner_dkpar_integral, inner_dkparp_integral],
        axis=2)


    integrand = (f_psi_term * f_phi_term / chibs_reshaped**2 * 
                2 * chibs_reshaped * inner_dkarp_integral_full)
    
    return trapezoid(y=integrand, x=deltas_full)

def d_chib_integral(f_Kpsi, f_Kphi):
    integrand = d_delta_integral(f_Kpsi, f_Kphi)
    oup = trapezoid(x = chibs, y = integrand)
    return oup


def limber_cross(f_Kpsi, f_Kphi):
    tmp = np.zeros((len(ells), len(chibs)), dtype=np.float128)
    for ell_idx in range(len(ells)):
        ell = ells[ell_idx]
        tmp[ell_idx] = f_Kpsi(chibs) * f_Kphi(chibs) / chibs**2 * ccl.linear_matter_power(cosmo, (ell + 1/2) / chibs,1)

    f_integrand = interp1d(x = chibs, y = tmp)
    oup, _ = quad_vec(f_integrand, min(chibs), max(chibs),
                      epsabs = 0.0, epsrel = 1e-4)

    return oup



#### Make vectorized Pk for vectorized limber
_tmp_ks = np.logspace(-5, 3, 10000)
_tmp_Pk = np.zeros_like(_tmp_ks)

for k_idx, k in enumerate(_tmp_ks):
    _tmp_Pk[k_idx] = ccl.linear_matter_power(cosmo, k, 1)

_f_Pk = interp1d(x = _tmp_ks, y = _tmp_Pk)



def limber_cross_vectorized(f_Kpsi, f_Kphi):
    ells_reshaped = ells.reshape(-1, 1)
    
    k_values = (ells_reshaped + 0.5) / chibs
    
    power_spectrum = _f_Pk(k_values)
    
    f_psi_term = f_Kpsi(chibs)  # (n_chibs,)
    f_phi_term = f_Kphi(chibs)  # (n_chibs,)
    
    integrand = f_psi_term * f_phi_term / chibs**2 * power_spectrum
    
    return trapezoid(x=chibs, y=integrand)
