from LIMxCMBL.init import *
from LIMxCMBL.cross_spectrum import *

from scipy.integrate import quad_vec, trapezoid
from tqdm import trange

def get_eHIeHI(chimin, chimax, Lambda):
    """
    No PeI included, have to multiply afterwards
    """
    chis_restricted = chis_resample[np.where((chis_resample >= chimin) & (chis_resample <= chimax))]
    _chis  = chis_restricted.reshape(-1, 1)
    _chips = chis_restricted.reshape(1 ,-1)
    dchi = np.mean(np.diff(chis_restricted))

    eHIeHI = np.zeros((len(chis_restricted), len(chis_restricted)), dtype=np.float64)
#    print('computing eIeI')
    eIeI = np.zeros_like(eHIeHI)
    for chi_idx in range(len(chis_restricted)):
        chi = chis_restricted[chi_idx]
        eIeI[chi_idx, chi_idx] = 1 / (dchi * chi**2) 

#    print('computing eIeLO')
    eIeLO = np.zeros_like(eHIeHI)
    eIeLO = 1/_chis**2 * Lambda / np.pi * np.sinc(Lambda * (_chis - _chips))

#    print('computing eLOeI')
    eLOeI = np.zeros_like(eHIeHI)
    eLOeI = 1/_chips**2 * Lambda / np.pi * np.sinc(Lambda * (_chis - _chips))

#    print('computing eLOeLO')
    def integrand(_chib):
        return Lambda**2 / np.pi**2 / _chib ** 2 * np.sinc(Lambda * (_chis - _chib)) * np.sinc(Lambda * (_chips - _chib))
    eLOeLO, _ = quad_vec(integrand, chimin, chimax, 
                      epsabs = 0.0, epsrel = 1e-3)

    return eIeI, eLOeLO, eIeLO, eLOeI
