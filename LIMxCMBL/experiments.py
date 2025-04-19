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

def old_COMAP_Pei(extra_info=False):
    #Li Wechsler+ 16 1503.08833 Table 1
    #updated to COMAP early science 2111.05931
    zmin = 2.4
    zmax = 3.4

    R = 800
    theta = 4.5*u.arcmin #theta fwhm
    Omegapix = theta**2 / (8  * np.log(2))
    Omegasurv = 12 * u.deg**2
    tObs = 5000 * u.hr #5000 hours number comes from 2111.05929
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
    if(extra_info):
        return tPixel, (Tsys / np.sqrt(nFeed * dnu * tPixel)), sigmaIPixel, voxelComovingVolume(zmin, Omegapix, R=R), voxelComovingVolume(zmax, Omegapix, R=R)

    return ((sigmaIPixel**2 * voxelComovingVolume(zmin, Omegapix, R=R)).to(u.kJy**2 / u.sr**2 * u.Mpc**3),
            (sigmaIPixel**2 * voxelComovingVolume(zmax, Omegapix, R=R)).to(u.kJy**2 / u.sr**2 * u.Mpc**3))

def SPHEREx_Pei():
    #1412.4872
    #2103.01971
    zmin = 5
    zmax = 8
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
    #and additional values from 2110.04298
    zmin = 1.9
    zmax = 3.5

    R = 800
    Omegapix = (3*u.arcsec)**2
    Omegasurv = 540 * u.deg**2

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
    #see 012.003.2025-04-15.CHIME-noise-figure-out.ipynb
    #basically follows 2201.07869 and 1405.1452 
    #borrows code from github/philbull/RadioFisher/process_chime_baselines.py


    zmin = 1
    zmax = 1.3
    
    def _Pei(zcenter):
        #table 2. of 2201.07869
        Tsys = 55*u.K
        Ssky = 31000 * u.deg**2
        lmbda = 21*u.cm*(1+zcenter)
        nu_center = cu.c / (lmbda)
        
        
        ttot = 1*u.yr
        Nant = 256
        npol = 2
        Ncyl = 4
        wcyl = 20*u.m
        lcyl = 78*u.m
        
        SFOV = (90*u.deg * (lmbda/wcyl *(u.rad)))
        
        
        eta = 0.7
        Ae = eta * lcyl/Nant * wcyl
        
        l4_over_Ae2 = ((lmbda/wcyl).si * u.rad)**2 * lmbda**2/(eta * lcyl / Nant)**2
        
        nu_HI = cu.c / (21 * u.cm)
    
    
        Ddish = 20. #I believe thisis the same as cylinder width in 2201.07869
        Dmin = 20.
        Ndish = Nant * Ncyl
        
        nu = nu_center.to(u.MHz).value # MHz
        l = 3e8 / (nu * 1e6) # Lambda [m]
        # Cut baselines d < d_fov
        
        nu_data = 800 #[MHz] they choose to tabulate at a different central nu but we're close enough
        outfile = "data/nx_CHIME_%d.dat" % nu_data
        AVG_SMALL_BASELINES = False
        Dcut = Ddish # Cut baselines below this separation
    
        data = np.loadtxt(outfile)
        x, n_x = data.T
        _u = x * nu
        nn = n_x / nu**2
    
        # Integrate n(u) to find normalisation (should give unity if no baseline cuts applied)   
        norm = simps(2.*np.pi*nn*_u, _u)
        nn_normalized = nn * 0.5 * Ndish * (Ndish - 1) / norm
    
    
    
        #take fid n_u to be median of n(u) as opposed to accounting for anisotropy
        fid_n_u = np.median(nn_normalized) * u.rad**2 #I think that n_u should be rad^2 bc u is units 1/rad
    
        noise_power = Tsys**2 / (nu_HI * npol * ttot) * Ssky/(SFOV) * l4_over_Ae2 * 1/fid_n_u
        noise_sigma = (2. * nu_center**2 * cu.k_B
                       * np.sqrt(noise_power)
                       / cu.c**2) / u.sr
        CN = (noise_sigma**2).to(u.kJy**2/u.sr**2)
        rnu = (cu.c * (1+zcenter)**2 / (cosmo['h']*(100*(u.km/u.s/u.Mpc))*ccl.background.h_over_h0(cosmo, 1./(zcenter+1)))).to(u.Mpc)

        return CN * rnu * (ccl.comoving_radial_distance(cosmo, 1/(1+zcenter)) * u.Mpc)**2        
    return _Pei(zmax), _Pei(zmin)

#####################################


experiments = {}
with open('LIMxCMBL/_experiments.txt') as f:
    f.readline() # skip header
    for line in f:
        _data = (line.strip().split())
        experiments[_data[0]] = {}
        experiments[_data[0]]['line_str'] = _data[1]
        experiments[_data[0]]['zmin'] = float(_data[2])
        experiments[_data[0]]['zmax'] = float(_data[3])
        experiments[_data[0]]['Omega_field'] = (float(_data[5]) * u.deg**2).to(u.rad**2) #rad^2


#2011.08193 table 1.
experiments['CCAT-prime']['Omega_pix'] = (30 * u.arcsec)**2 / (8 * np.log(2))
#1503.08833 Table 1
experiments['COMAP']['Omega_pix'] = (4.5 * u.arcmin)**2 / (8 * np.log(2))
#2103.01971 table 1
experiments['SPHEREx']['Omega_pix'] = (6 * u.arcsec)**2
experiments['HETDEX']['Omega_pix']  = (3 * u.arcsec)**2 #2110.04298
#2201.07869 says CHIME's angular resolution is ~40'
experiments['CHIME']['Omega_pix']  = (40*u.arcmin)**2



experiments['CCAT-prime']['R'] = 100
experiments['COMAP']['R'] = 800
experiments['SPHEREx']['R'] = 150
experiments['HETDEX']['R']  = 800


def Gamma_nu(nuobs, nurest):
    _z = (nurest/nuobs - 1).si.value
    return (cu.c * 
            (ccl.comoving_radial_distance(cosmo, 1./(1+_z))*u.Mpc)**2
            / (h*100*ccl.background.h_over_h0(cosmo,1./(1+_z)) * (u.km/u.s/u.Mpc)))
 
def COMAP_Pei():
    nurest = (115.27 * u.GHz)
    Omega_field = experiments['COMAP']['Omega_field']
   
    def _Pei(nuobs):
        return ((40 * u.K)**2 /19 / (5000 * u.hr) 
         * (nuobs)**2 * (2 * cu.k_B / cu.c**2)**2 
         * nurest  * (Omega_field.to(u.sr)).value * Gamma_nu(nuobs, nurest)).to((u.kJy)**2 * u.Mpc**3)
    nuobs = np.linspace(26,34,1000) * u.GHz
    _Peis = _Pei(nuobs)
    return nuobs, _Peis


Pei_dict = {
        'CCAT-prime': CCAT_prime_Pei,
        'COMAP': COMAP_Pei,
        'SPHEREx':SPHEREx_Pei,
        'HETDEX':HETDEX_Pei,
        'CHIME': CHIME_Pei
        }

for _e in experiments:
    experiments[_e]['f_Pei'] = Pei_dict[_e]
