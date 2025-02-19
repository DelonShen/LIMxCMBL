from LIMxCMBL.init import *
from LIMxCMBL.cross_spectrum import *

from scipy.integrate import quad_vec, trapezoid
from tqdm import trange



from sympy.parsing.mathematica import parse_mathematica
from sympy import lambdify
import mpmath as mpm

from multiprocessing import Pool
from itertools import product
from tqdm import tqdm
import pickle
mpm.mp.dps = 33;

#load into sympy
eLOeLO_diag_mathematica = None
with open('008.008.eLOeLO_diag_mathematica.txt', 'r') as f:
    eLOeLO_diag_mathematica = f.read()
eLOeLO_diag_mathematica = eLOeLO_diag_mathematica.replace('\[Pi]', 'Pi')
eLOeLO_diag_sympy = parse_mathematica(eLOeLO_diag_mathematica)

eLOeLO_off_diag_mathematica = None
with open('008.008.eLOeLO_off_diag_mathematica.txt', 'r') as f:
    eLOeLO_off_diag_mathematica = f.read()
eLOeLO_off_diag_mathematica = eLOeLO_off_diag_mathematica.replace('\[Pi]', 'Pi')
eLOeLO_off_diag_sympy = parse_mathematica(eLOeLO_off_diag_mathematica)

#turn into mpmath function
modules = [
        {
            'Ci': mpm.ci,
            'Si': mpm.si,
            'Cos': mpm.cos,
            'Sin': mpm.sin,
            'Log': mpm.log,
            'I': 1j,},
        'mpmath']


eLOeLO_off_diag_mpmath = lambdify(list(eLOeLO_off_diag_sympy.free_symbols), 
                             eLOeLO_off_diag_sympy, modules=modules)
eLOeLO_diag_mpmath = lambdify(list(eLOeLO_diag_sympy.free_symbols), 
                             eLOeLO_diag_sympy, modules=modules)



#turn into scipy function
from scipy import special
def SinIntegral(x):
    si, _ = special.sici(x)
    return si

def CosIntegral(x):
    _, ci = special.sici(x)
    return ci

modules = [
    {
        'Ci': CosIntegral,
        'Si': SinIntegral,
        'Cos': np.cos,
        'Sin': np.sin,
        'Log': np.log,
        'I': 1j,
    },
    'numpy'
]
eLOeLO_off_diag_numpy = lambdify(list(eLOeLO_off_diag_sympy.free_symbols), 
                             eLOeLO_off_diag_sympy, modules=modules)
eLOeLO_diag_numpy = lambdify(list(eLOeLO_diag_sympy.free_symbols), 
                             eLOeLO_diag_sympy, modules=modules)





f_eIeI = lambda chi, dchi, Lambda : 1 / (dchi * chi ** 2)
f_cross = lambda chi, chip, Lambda : (1/chi**2  * Lambda / np.pi * np.sinc(Lambda * (chi - chip) / np.pi)
                              + 1/chip**2 * Lambda / np.pi * np.sinc(Lambda * (chi - chip) / np.pi))

f_cross_mpm = lambda chi, chip, Lambda : (1/chi**2  * Lambda / mpm.pi * mpm.sinc(Lambda * (chi - chip))
                              + 1/chip**2 * Lambda / mpm.pi * mpm.sinc(Lambda * (chi - chip)))


def compute_elementLOLO(params):
    idx1, idx2, chimin, chimax, chi1, chi2, Lambda = params
    if idx1 == idx2:
        return (idx1, idx2, eLOeLO_diag_mpmath(a=chimin, b=chimax, x=chi1, L=Lambda))
    else:
        return (idx1, idx2, eLOeLO_off_diag_mpmath(a=chimin, b=chimax, x=chi1, xp=chi2, L=Lambda))



def compute_element(params):
    idx1, idx2, chimin, chimax, chi1, chi2, Lambda, dchi = params
    if idx1 == idx2:
        return (idx1, idx2, 
                f_eIeI(chi1, dchi, Lambda), 
                f_cross_mpm(chi1, chi2, Lambda), 
                eLOeLO_diag_mpmath(a=chimin, b=chimax, x=chi1, L=Lambda))
    else:
        return (idx1, idx2, 
                0, 
                f_cross_mpm(chi1, chi2, Lambda), 
                eLOeLO_off_diag_mpmath(a=chimin, b=chimax, x=chi1, xp=chi2, L=Lambda))

def f_eHIeHI(chimin, chimax, dchi, chis, Lambda):
    n = len(chis)
    eIeI = [[0] * n for _ in range(n)]
    cross = [[0] * n for _ in range(n)]
    eLOeLO = [[0] * n for _ in range(n)]
  

    params = [
        (i, j, chimin, chimax, chis[i], chis[j], Lambda, dchi)
        for i in range(n) for j in range(i, n)
    ]
    
    with Pool(processes=32) as pool:
        results = list(tqdm(
            pool.imap(compute_element, params),
            total=len(params),
            desc="Computing matrix elements"
        ))

    return results


def f_eLOeLO(chimin, chimax, chis, Lambda):
    n = len(chis)
    ret = [[0] * n for _ in range(n)]
    

    params = [
        (i, j, chimin, chimax, chis[i], chis[j], Lambda)
        for i, j in product(range(n), range(n))
    ]
    
    with Pool(processes=32) as pool:
        results = list(tqdm(
            pool.imap(compute_elementLOLO, params),
            total=len(params),
            desc="Computing matrix elements"
        ))
    
    for i, j, value in results:
        ret[i][j] = value
    
    return mpm.matrix(ret)

def get_eHIeHI(chimin, chimax, Lambda):
    """
    DEPRECEATED 
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
    eIeLO = 1/_chis**2 * Lambda / np.pi * np.sinc(Lambda * (_chis - _chips) / np.pi)

#    print('computing eLOeI')
    eLOeI = np.zeros_like(eHIeHI)
    eLOeI = 1/_chips**2 * Lambda / np.pi * np.sinc(Lambda * (_chis - _chips) / np.pi)

#    print('computing eLOeLO')
    def integrand(_chib):
        return Lambda**2 / np.pi**2 / _chib ** 2 * np.sinc(Lambda * (_chis - _chib) / np.pi) * np.sinc(Lambda * (_chips - _chib) / np.pi)
    eLOeLO, _ = quad_vec(integrand, chimin, chimax, 
                      epsabs = 0.0, epsrel = 1e-10)

    return eIeI, eLOeLO, eIeLO, eLOeI
