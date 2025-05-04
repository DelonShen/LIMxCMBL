from .init import *
from scipy.interpolate import interp1d


chis = np.linspace(0, chimax_sample, 10**4)
a_s = ccl.scale_factor_of_chi(cosmo, chis)
zs = 1/a_s - 1
Dz = ccl.growth_factor(cosmo, a_s)

##### CMB Lensing
zstar = 1100
cmbk = ccl.CMBLensingTracer(cosmo, z_source=zstar)
Wk = cmbk.get_kernel(chis)[0]

def get_f_Kkappa():
    return interp1d(chis, Wk * Dz, 
                    bounds_error = False,
                    fill_value='extrapolate')


#note that technically zmax_sample should be ~1100 and not 20 as we currently have it
#but this would only be a problem for a kernel that has broad support like CMB lensing
#for LIM and LIMxCMBL, zmax_sample=20 should not be a problem b.c. of the windows + sincs
#here we just compute CL_kappa with Limber for zmax_sample = zstar
_zmax_sample = 1100
_amax_sample = 1/(_zmax_sample+1)
_chimax_sample = ccl.comoving_radial_distance(cosmo, _amax_sample)
_chis = np.linspace(1e-8, _chimax_sample, 10**6)
_a_s = ccl.scale_factor_of_chi(cosmo, _chis)
_zs = 1/_a_s - 1
_Dz = ccl.growth_factor(cosmo, _a_s)
_Wk = cmbk.get_kernel(_chis)[0]

_tmp = np.zeros((len(ells), len(_chis)))
for ell_idx in range(len(ells)):
    _tmp[ell_idx] = (_Wk * _Dz)**2/_chis**2 * ccl.linear_matter_power(cosmo, (ells[ell_idx] + 1/2) / _chis, 1)

from scipy.integrate import trapezoid,simpson
ClKK_perfect = simpson(x=_chis, y=_tmp)

##### LIM from SFR Table
from scipy.integrate import simpson as simps 
from scipy.interpolate import griddata 

### Code from SkyLine

def SFR_Mz_2dinterp(Mvals,zvals):
    '''
    Returns SFR(M,z) interpolated from tables of 1+z, log10(Mhalo/Msun) and 
    log10(SFR / (Msun/yr)), in three columns, where 1+z is the innermost index 
    (the one running fast compared with the mass)
    '''
    x = np.loadtxt('data/UM_sfr.dat')
    zb = np.unique(x[:,0])-1.
    logMb = np.unique(x[:,1])
    
    logSFRb = x[:,2].reshape(len(zb),len(logMb),order='F')

    xx, yy = np.meshgrid(logMb, zb)

    ingrid = np.array([xx, yy]).reshape(2, len(zb)*len(logMb))

    inval = logSFRb.reshape(len(zb)*len(logMb))
    #Assuming in Msun/h units

    Msoverh = Mvals*cosmo['h']

    mzxx, mzyy = np.meshgrid(np.log10(Msoverh), zvals)
    mzgrid = np.array([mzxx, mzyy]).reshape(2, len(zvals)*len(Mvals))
    del mzxx, mzyy
    # grid = np.array([logM, z]) 

    logSFR_interp = griddata(ingrid.T, inval, mzgrid.T, fill_value=-40.)

    
    SFR = 10**logSFR_interp
    
    return SFR

Hzs = cosmo['h']*100*ccl.background.h_over_h0(cosmo, 1./(zs+1))
Dz = ccl.growth_factor(cosmo, 1./(zs+1))
M_min = 1e8
M_max = 1e16
Ms = np.geomspace(M_min, M_max, 100)
sfr_interp = SFR_Mz_2dinterp(Ms, zs)
sfrgrid = sfr_interp.reshape(len(zs), len(Ms))


import astropy.units as u
import astropy.constants as cu



ks = np.geomspace(1e-3, 10, 1000)


pkvec = np.zeros(shape=(len(zs), len(ks)))


#Define critical density for spherical collapse
delta_c = 1.686
#Define time evolution

nuvec = np.zeros(shape=(len(Ms), len(a_s)))


for i, a in enumerate(a_s):
    nuvec[:,i] = delta_c/ccl.sigmaM(cosmo, Ms, a)
###Mass function
hmf = ccl.halos.MassFuncTinker08(mass_def = ccl.halos.MassDef200m)
#Have to call mass function at one redshift at a time... annoying
nm = hmf(cosmo, Ms, 1./(1 + 3))

nmvec = np.zeros(shape=(len(zs), len(Ms)))

sigMvec = np.zeros(shape=(len(zs), len(Ms)))
for i, a in enumerate(a_s):
    nmvec[i] = hmf(cosmo, Ms, a)
    sigMvec[i] = ccl.sigmaM(cosmo, Ms, a)
    pkvec[i] = ccl.linear_matter_power(cosmo, ks, a) * u.Mpc**3


#Integrand for I(z) = rho_L / H(z) 
integrand_Iz = np.einsum('zm, z, zm ->zm', nmvec, 1./Hzs, sfrgrid)

#Integrand for rho_L (M, z) 
#Convert from SFR to L following 2211.04531
#log L_CII = 1.26 * log SFR + 7.1
#units of SFR are Msun/yr and L_CII are Lsun.
alpha = 1.26
beta = 7.1
Lgrid = 10**(alpha * np.log10(sfrgrid) + beta) * u.Lsun
integrand_L = np.einsum('zm, zm ->zm', nmvec, Lgrid)

#Integrand for the variance of the luminosity which is Lgrid**2
integrand_var = np.einsum('zm, zm ->zm', nmvec, Lgrid**2)


###### Borrowed from CCL -- Calculates Tinker10 bias as a function of nu. 
dcrit = 1.686

nuvec = dcrit/sigMvec

B = 0.183
b = 1.5
c = 2.4

ld = np.log10(200)
xp = np.exp(-(4./ld)**4.)
A = 1.0 + 0.24 * ld * xp
C = 0.019 + 0.107 * ld + 0.19*xp
aa = 0.44 * ld - 0.88
dc = 1.*dcrit
nupa = nuvec**aa

bvec = 1 - A * nupa / (nupa + dc**aa) + (
            B * nuvec**b + C * nuvec**c)
########


integrand_bias = np.einsum('zm, zm, zm -> zm', nmvec, Lgrid, bvec)



Lz = simps(y=integrand_L, x=np.log10(Ms), axis=-1) * u.Lsun / (u.Mpc)**3


bL_avg = simps(y=integrand_bias, x=np.log10(Ms), axis=-1)*(u.Lsun / (u.Mpc)**3)/ Lz


nu_CII = cu.c / (158*u.um)
Hzbit = cu.c / (4*np.pi * u.sr * (Hzs*u.km/u.s/u.Mpc) * nu_CII)

KI = Dz*(bL_avg * Lz * Hzbit).to(u.kJy/u.sr)
_KI = np.array(KI)

f_KI = interp1d(chis, KI, bounds_error = False, fill_value=0)


def f_KI1D(chi):
    return np.interp(x = chi, xp = chis, fp = _KI, left = 0, right = 0)


def get_f_KI():
    return f_KI

def get_window(chimin, chimax):
    _window = np.zeros_like(chis)
    _window[(chis > chimin) & (chis < chimax)] = 1
    return interp1d(chis, _window, fill_value = 'extrapolate')

def apply_window(f_K, chimin, chimax):
    f_window = get_window(chimin, chimax)
    return lambda chi : f_K(chi) * f_window(chi)


# the extra factor of pi is to follow numpy's sinc convention


import numba
import math
@numba.vectorize(['f8(f8)','f4(f4)'])
def sinc(x):
    if x == 0:
        return 1.0
    else:
        return math.sin(x*math.pi) / (x*math.pi)

def f_KILo(chi, external_chi, Lambda):
    return (Lambda / np.pi * np.interp(x = chi, xp = chis, fp = _KI, left = 0, right = 0) * sinc(Lambda * (external_chi - chi) / np.pi))


def get_f_KILo(external_chi, Lambda):
    prefactor = Lambda / np.pi #units 1/cMpc
    return lambda chi : prefactor * f_KI(chi) * np.sinc(Lambda * (external_chi - chi) / np.pi)

#Dongwoo 2111.05931
logC = 10.63
AC0 = -2.85
BC0 = -0.42
logMpMsun = 12.3
MpMsun = 10**(logMpMsun)
MpMsunph = MpMsun * h

pseudo_LC0 = 10**logC/((Ms / MpMsunph)**AC0 + (Ms / MpMsunph)**BC0)

#focus on J=1->0
LCO = 4.9e-5 * pseudo_LC0 * u.Lsun
CO_integrand_bias = np.einsum('zm, m, zm -> zm', nmvec, LCO, bvec)
integrand_LCO = np.einsum('zm, m ->zm', nmvec, LCO)
L_COz = simps(y=integrand_LCO, x=np.log10(Ms), axis=-1) * u.Lsun / (u.Mpc)**3
bL_CO_avg = simps(y=CO_integrand_bias, x=np.log10(Ms), axis=-1)*(u.Lsun / (u.Mpc)**3)/ L_COz


nu_CO = 1.1527e11 * u.Hz
Hzbit_CO = cu.c / (4*np.pi * u.sr * (Hzs*u.km/u.s/u.Mpc) * nu_CO)

KI_CO = Dz*(L_COz * bL_CO_avg * Hzbit_CO).to(u.kJy/u.sr)

#LyAlpha
#basically 2212.08056 which follows 1809.04550 
f_esc = np.einsum('z, zm->zm',
                  1/np.sqrt(1 + np.exp(-1.6*zs + 5)),
                  (0.18 + 0.82/(1 + 0.8 * sfrgrid**0.875))**2)
L_Lya = 1.6e42 * f_esc * sfrgrid * u.erg / u.s
L_Lya = L_Lya.to(u.Lsun)

LYa_integrand_bias = np.einsum('zm, zm, zm -> zm', nmvec, L_Lya, bvec)
integrand_LYa = np.einsum('zm, zm ->zm', nmvec, L_Lya)

L_Lya_z = simps(y=integrand_LYa, x=np.log10(Ms), axis=-1) * u.Lsun / (u.Mpc)**3
bL_Lya_avg = simps(y=LYa_integrand_bias, x=np.log10(Ms), axis=-1)*(u.Lsun / (u.Mpc)**3)/ L_Lya_z


nu_Lya = cu.c/(121.6 * u.nm)
Hzbit_Lya = cu.c / (4*np.pi * u.sr * (Hzs*u.km/u.s/u.Mpc) * nu_Lya)
KI_Lya = Dz*(L_Lya_z * bL_Lya_avg * Hzbit_Lya).to(u.kJy/u.sr)


#HI
#2212.08056 which follows 1405.1452, 1804.09180

##from Table 1 FOF of 1804.09180
_M0s = np.array([4.3e10, 1.5e10, 1.3e10, 2.9e9, 1.4e9, 1.9e9]) #Msol / h
_M0s /= h #Msol
f_M0 = interp1d(x=[0,1,2,3,4,5], y = _M0s, bounds_error = False)

_Mms = np.array([2e12, 6e11, 3.6e11, 6.7e10, 2.1e10, 2e10]) #Msol / h
_Mms /= h #Msol
f_Mm = interp1d(x=[0,1,2,3,4,5], y = _Mms, bounds_error = False)

_alphaMs = np.array([0.24, 0.53, 0.6, 0.76, 0.79, 0.74])
f_alphaM = interp1d(x=[0,1,2,3,4,5], y = _alphaMs, bounds_error = False)


betaM = 0.35
_tmp_zs = zs.reshape(-1, 1)
_tmp_Ms = Ms.reshape(1, -1)
MHI = np.where(_tmp_zs <= 5,
               f_M0(_tmp_zs)
               * (_tmp_Ms / f_Mm(_tmp_zs)) ** (f_alphaM(_tmp_zs))
               * np.exp(-(f_Mm(_tmp_zs) / _tmp_Ms)**betaM),
               0) * u.Msun

A10=2.869e-15*u.s**(-1) #spontaneous emission coefficient
nu_HI = cu.c / (21 * u.cm)
HI_coeff=((3/4)*A10*cu.h*nu_HI/cu.m_p).to(u.Lsun/u.Msun)
L_HI = (HI_coeff * MHI).to(u.Lsun)

HI_integrand_bias = np.einsum('zm, zm, zm -> zm', nmvec, L_HI, bvec)
integrand_HI = np.einsum('zm, zm ->zm', nmvec, L_HI)

L_HI_z = simps(y=integrand_HI, x=np.log10(Ms), axis=-1) * u.Lsun / (u.Mpc)**3
#bL_HI_avg = simps(y=HI_integrand_bias, x=np.log10(Ms), axis=-1)*(u.Lsun / (u.Mpc)**3)/ L_HI_z


Hzbit_HI = cu.c / (4*np.pi * u.sr * (Hzs*u.km/u.s/u.Mpc) * nu_HI)
KI_HI = Dz*(simps(y=HI_integrand_bias, x=np.log10(Ms), axis=-1)*(u.Lsun / (u.Mpc)**3)
            * Hzbit_HI).to(u.kJy/u.sr)



