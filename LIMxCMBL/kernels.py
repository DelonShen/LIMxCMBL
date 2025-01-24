from .init import *
from scipy.interpolate import interp1d


chis = np.linspace(0, chimax, 11234)
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




##### LIM from SFR Table
from scipy.integrate import simps 
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



Lz = simps(integrand_L, np.log10(Ms), axis=-1) * u.Lsun / (u.Mpc)**3


bL_avg = simps(integrand_bias, np.log10(Ms), axis=-1)*(u.Lsun / (u.Mpc)**3)/ Lz


nu_CII = cu.c / (158*u.um)
Hzbit = cu.c / (4*np.pi * u.sr * (Hzs*u.km/u.s/u.Mpc) * nu_CII)

KI = Dz*(bL_avg * Lz * Hzbit).to(u.kJy/u.sr)

def get_f_KI():
    return interp1d(chis, KI, 
                    bounds_error = False,
                    fill_value='extrapolate')


def apply_window(f_K, chimin, chimax):
    _window = np.zeros_like(chis)
    _window[(chis > chimin) & (chis < chimax)] = 1
    return interp1d(chis, f_K(chis) * _window,
                    bounds_error = False,
                    fill_value='extrapolate')

def get_f_KILo(external_chi, Lambda):
    prefactor = Lambda / np.pi #units 1/cMpc
    return interp1d(chis, prefactor*KI*np.sinc(Lambda*(external_chi - chis)),
                    bounds_error = False,
                    fill_value='extrapolate')


def get_f_KILo_noLC(external_chi, Lambda, mean):
    prefactor = Lambda / np.pi #units 1/cMpc
    return interp1d(chis, prefactor*mean*np.sinc(Lambda*(external_chi - chis)),
                    bounds_error = False,
                    fill_value='extrapolate')

