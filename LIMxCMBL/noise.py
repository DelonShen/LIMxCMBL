from LIMxCMBL.init import *

from scipy.integrate import quad_vec
from tqdm import trange

def get_eHIeHI(Pei, chimin, chimax, dchi, Lambda, elems = False):
    eHIeHI = np.zeros((len(chibs), len(chibs)), dtype=np.float64)
    print('computing eIeI')
    eIeI = np.zeros_like(eHIeHI)
    for chi_idx in range(len(chibs)):
        chi = chibs[chi_idx]
        if(chi > chimin and chi < chimax):
            eIeI[chi_idx, chi_idx] = 1 / (dchi * chi**2) 
    eIeI *= Pei

    print('computing eIeLO')
    eIeLO = np.zeros_like(eHIeHI)
    for chi_idx in range(len(chibs)):
        chi = chibs[chi_idx]
        if(chi > chimin and chi < chimax):
            for chip_idx in range(len(chibs)):
                chip = chibs[chip_idx]
                eIeLO[chi_idx, chip_idx] = 1/chi**2 * Lambda/np.pi * np.sinc(Lambda * (chi - chip))
    eIeLO *= Pei


    print('computing eLOeI')
    eLOeI = np.zeros_like(eHIeHI)
    for chi_idx in range(len(chibs)):
        chi = chibs[chi_idx]
        for chip_idx in range(len(chibs)):
            chip = chibs[chip_idx]
            if(chip > chimin and chip < chimax):
                eLOeI[chi_idx, chip_idx] = 1/chip**2 * Lambda/np.pi * np.sinc(Lambda * (chip - chi))
    eLOeI *= Pei       

    print('computing eLOeLO')

    def KepsLO(chip, chi):
        return np.sinc(Lambda * (chi - chip)) * Lambda / np.pi
    
    eLOeLO = np.zeros_like(eHIeHI)
    for chi_idx in trange(len(chibs)):
        chi = chibs[chi_idx]
        
        def integrand(chib):
            return 1/chib**2 * KepsLO(chib, chi) * KepsLO(chib, chibs)
            
        eLOeLO[chi_idx], _ = quad_vec(integrand, 
                                         chimin, chimax,
                                         epsabs = 0.0, 
                                         epsrel = 1e-3)
    eLOeLO *= Pei

    if(elems):
        return eIeI, eLOeLO, eIeLO, eLOeI

    return eIeI + eLOeLO - eIeLO - eLOeI
