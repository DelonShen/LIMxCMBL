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

def SPHEREx_Pei():
    #1412.4872
    #2103.01971
    zmin = 5
    zmax = 8.2
    idxs = np.where((zs < zmax) & (zs>zmin))[0]
    R = 150
    Omegapix = (6*u.arcsec)**2
    Omegasurv = 2 * 100 * u.deg**2
    # borrowed from LIM branch of github/EmmanuelSchaan/HaloGen
    # Inferred from SPHEREx science book
    mAB5Sigma = 22 # 5sigma lim mag for point source (Cheng+18)
    f5Sigma = 10.**((8.9-mAB5Sigma)/2.5) * u.Jy   # 5sigma lim point source flux [Jy]
    sigmaFSource = f5Sigma / 5. # 1sigma lim point source flux [Jy]
    
    # This point source flux is the output of a spatial matched filter
    # for one frequency element.
    # It needs to be converted to pixel flux.
    # The SPHEREx doc, fig 9, gives the conversion using 
    # the effective number of pixels covered by the PSF
    nPixEff = 3.   # 2-5 in fig 9 of SPHEREx doc
    
    sigmaFPixel = sigmaFSource / np.sqrt(nPixEff)
    sigmaIPixel = sigmaFPixel / Omegapix 
    
    # convert from pixel variance
    # to white noise power spectrum
    worst = sigmaIPixel**2 * voxelComovingVolume(zmax, Omegapix, R=R)
    worst = worst.to((u.kJy/u.sr)**2 * u.Mpc**3)

    best = sigmaIPixel**2 * voxelComovingVolume(zmin, Omegapix, R=R)
    best = best.to((u.kJy/u.sr)**2 * u.Mpc**3)

    return worst, best

def HETDEX_Pei():
    #again borrowing from Manu, LIM branch of Halogen
    zmin = 1.9
    zmax = 3.5

    R = 800
    Omegapix = (3*u.arcsec)**2
    Omegasurv = 300 * u.deg**2

    def _HETDEX_Pei(z):
        # focus on Lyman-alpha
        nuHz = nu_Lya / (1.+z) # convert from rest to observed freq
        # flux noise from Hill+08, Cheng+18 (similar to Hill+16)
        # they quote the 5-sigma uncertainty --> divide by 5
        sigmaFPixel = 5.5e-17 / 5.  * u.erg / u.s / u.cm**2
        # convert from flux to intensity
        sigmaIPixel = sigmaFPixel / Omegapix * R/nuHz   # [erg/s/cm^2/sr/Hz]
        # convert to noise power spectrum
        result = sigmaIPixel**2 * voxelComovingVolume(z, Omegapix, R=R)
        return result.to(u.Mpc**3 * (u.kJy/u.sr)**2)
    return _HETDEX_Pei(zmax), _HETDEX_Pei(zmin)

def CHIME_Pei():
    zmin = 1.
    zmax = 1.3
    #from 2201.07869 App. A.3
    R = ((nu_HI/(1+(zmin + zmax)/2))/(390 * u.kHz)).to(u.dimensionless_unscaled)
    Omegapix = (40*u.arcmin)**2
    Omegasurv = 31000 * u.deg**2

    #1809.06384
    #0910.5007
    #2201.07869 App. A.3
    sigmaT = 2.9e-4 * u.K
    sigmaIPixel = (2. * (nu_HI/(1+zmin))**2 * cu.k_B
                   * sigmaT # [K]
                   / cu.c**2) / u.sr
    sigmaIPixel = (sigmaIPixel).to(u.kJy/u.sr)
    return ((sigmaIPixel**2 * voxelComovingVolume(zmax, Omegapix, R=R)).to((u.kJy/u.sr)**2 * u.Mpc**3),
            (sigmaIPixel**2 * voxelComovingVolume(zmin, Omegapix, R=R)).to((u.kJy/u.sr)**2 * u.Mpc**3))

Pei_dict = {
        'CCAT-prime': CCAT_prime_Pei,
        'COMAP': COMAP_Pei,
        'SPHEREx':SPHEREx_Pei,
        'HETDEX':HETDEX_Pei,
        'CHIME': CHIME_Pei
        }


experiments = {}
with open('LIMxCMBL/_experiments.txt') as f:
    f.readline() # skip header
    for line in f:
        _data = (line.strip().split())
        experiments[_data[0]] = {}
        experiments[_data[0]]['line_str'] = _data[1]
        experiments[_data[0]]['zmin'] = float(_data[2])
        experiments[_data[0]]['zmax'] = float(_data[3])
        experiments[_data[0]]['f_Pei'] = Pei_dict[_data[0]]
        experiments[_data[0]]['Omega_field'] = (float(_data[5]) * u.deg**2).to(u.rad**2) #rad^2


#2011.08193 table 1.
experiments['CCAT-prime']['Omega_pix'] = (30 * u.arcsec)**2 / (8 * np.log(2))
#1503.08833 Table 1
experiments['COMAP']['Omega_pix'] = (6 * u.arcmin)**2 / (8 * np.log(2))
#2103.01971 table 1
experiments['SPHEREx']['Omega_pix'] = (6 * u.arcsec)**2
experiments['HETDEX']['Omega_pix']  = (3 * u.arcsec)**2
#2201.07869 Table 2.
experiments['CHIME']['Omega_pix']  = (40*u.arcmin)**2



experiments['CCAT-prime']['R'] = 100
experiments['COMAP']['R'] = 800
experiments['SPHEREx']['R'] = 150
experiments['HETDEX']['R']  = 800
