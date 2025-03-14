from LIMxCMBL.init import *
from LIMxCMBL.kernels import *

def voxelComovingDepth(z, R):
    Hubb = h*100*ccl.background.h_over_h0(cosmo, 1./(z+1)) * (u.km/u.s/u.Mpc)
    result = cu.c / Hubb  # units length
    result *= (1.+z) / R
    return result

def pixelComovingArea(pixarea, z):
    return (pixarea/u.sr) * (ccl.comoving_radial_distance(cosmo, 1./(1+z))*u.Mpc)**2

def voxelComovingVolume(z, pixarea,  R=None):
    result = voxelComovingDepth(z, R=R)
    result *= pixelComovingArea(pixarea, z)
    return result

def CCAT_prime_Pei():
    #Sato-Polito+20 Table 1.
    zmin = 3.5
    zmax = 8.1
    ##ccat prime
    theta = 30*u.arcsec #theta fwhm
    Omegapix = theta**2 / (8  * np.log(2))

    Vvox = voxelComovingVolume(zmax, Omegapix, R=100)
    worst_Pei = (5.7e4 *u.Jy / u.sr )**2 * Vvox
    worst_Pei = worst_Pei.to((u.kJy / u.sr)**2 * u.Mpc**3)
    
    
    Vvox = voxelComovingVolume(zmin, Omegapix, R=100)
    best_Pei = (5.7e4 *u.Jy / u.sr )**2 * Vvox
    best_Pei = best_Pei.to((u.kJy / u.sr)**2 * u.Mpc**3)

    return worst_Pei, best_Pei

def COMAP_Pei():
    #Li Wechsler+ 16 1503.08833 Table 1
    zmin = 2.4
    zmax = 3.4

    R = 800
    theta = 6*u.arcmin #theta fwhm
    Omegapix = theta**2 / (8  * np.log(2))
    Omegasurv = 2.5 * u.deg**2
    tObs = 1500 * u.hr
    # observing time per pixel [s]
    tPixel = tObs * Omegapix / Omegasurv
    Tsys = 40 * u.K # system temperature [K]
    
    nFeed = 19.  # number of feeds
    dnu = 40 * u.MHz # spectral element width [Hz]
    nuCenter = 32 * u.GHz  # 30-34GHz is the COMAP band [Hz]
    
    # radiometer equation (App C1 in Li Wechsler+16)
    # giving the pixel noise standard deviation [K]
    # and convert with Rayleigh Jeans
    sigmaIPixel = (2. * nuCenter**2 * cu.k_B
                   * Tsys / np.sqrt(nFeed * dnu * tPixel) # [K]
                   / cu.c**2) / u.sr
    sigmaIPixel = (sigmaIPixel).to(u.kJy/u.sr)

    return ((sigmaIPixel**2 * voxelComovingVolume(zmin, Omegapix, R=R)).to(u.kJy**2 / u.sr**2 * u.Mpc**3),
            (sigmaIPixel**2 * voxelComovingVolume(zmax, Omegapix, R=R)).to(u.kJy**2 / u.sr**2 * u.Mpc**3))
