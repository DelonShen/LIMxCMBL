from LIMxCMBL.init import *
from scipy.interpolate import interp1d

inner_dkparp_integral = np.load('/oak/stanford/orgs/kipac/users/delon/LIMxCMBL/inner_dkparp_integral.npy')
f_inner_integral = interp1d(x = chibs, y = inner_dkparp_integral, axis = 1)


from scipy.integrate import quad, quad_vec, trapezoid

deltas_reshaped = deltas.reshape(1, 1, -1)


#chis_resample = np.linspace(10, chimax_sample, 2**13)
#it turned out this was super expensive for no reason

chis_resample = chibs
chis_reshaped = chis_resample.reshape(1, -1, 1)
minus_args = chis_reshaped * (1 - deltas_reshaped)
plus_args  = chis_reshaped * (1 + deltas_reshaped)

def d_delta_integral(f_Kpsi, f_Kphi):
    prefactor = 2 / chis_reshaped
    kernels = (f_Kpsi(minus_args) * f_Kphi(plus_args) + f_Kpsi(plus_args) * f_Kphi(minus_args))
    inner_integral_resampled = f_inner_integral(chis_resample)
    integrand = prefactor*kernels*inner_integral_resampled*deltas_reshaped
    return trapezoid(y=integrand, x=np.log(deltas), axis = -1)

def d_chib_integral(f_Kpsi, f_Kphi):
    integrand = d_delta_integral(f_Kpsi, f_Kphi)
    oup = trapezoid(x = chis_resample, y = integrand)
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



