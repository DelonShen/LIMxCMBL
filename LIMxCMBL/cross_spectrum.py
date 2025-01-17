from LIMxCMBL.init import *
from scipy.interpolate import interp1d

inner_dkparp_integral = np.load('/oak/stanford/orgs/kipac/users/delon/LIMxCMBL/inner_dkparp_integral.npy')
deltas_full = np.concatenate((-deltas[::-1], 
                              [0], 
                              deltas))

from scipy.integrate import quad, quad_vec, trapezoid

def d_delta_integral(f_Kpsi, f_Kphi):
    oup = np.zeros((len(ells), len(chibs)), dtype = np.float128)

    for ell_idx in range(len(ells)):
        for chib_idx in range(len(chibs)):
            chib = chibs[chib_idx]
            # symmetry wrt delta -> -delta
            inner_dkarp_integral_full = np.concatenate((inner_dkparp_integral[ell_idx,chib_idx][::-1],
                                                    [inner_dkparp_integral[ell_idx,chib_idx,0]],
                                                    inner_dkparp_integral[ell_idx][chib_idx]))

            oup[ell_idx, chib_idx] = trapezoid(y = f_Kpsi(chib*(1-deltas_full)) * f_Kphi(chib*(1+deltas_full)) / chib**2  * 2 * chib * inner_dkarp_integral_full,
                                      x = deltas_full)

    return oup

def d_chib_integral(f_Kpsi, f_Kphi):
    f_dchib_integral = interp1d(x = chibs, y = d_delta_integral(f_Kpsi, f_Kphi))
    oup, _ = quad_vec(f_dchib_integral, min(chibs), max(chibs),
                      epsabs = 0.0, epsrel = 1e-4)
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
