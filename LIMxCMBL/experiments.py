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
#the relation to our noise model is approx OmegA_pix = Omega_beam single antenna / Nant
experiments['CHIME']['Omega_pix']  = (40*u.arcmin)**2



experiments['CCAT-prime']['R'] = 100
experiments['COMAP']['R'] = 800
experiments['SPHEREx']['R'] = 41
experiments['HETDEX']['R']  = 800
experiments['CHIME']['R'] = 1700


def Gamma_nu(nuobs, nurest):
    _z = (nurest/nuobs - 1).si.value
    return (cu.c * 
            (ccl.comoving_radial_distance(cosmo, 1./(1+_z))*u.Mpc)**2
            / (h*100*ccl.background.h_over_h0(cosmo,1./(1+_z)) * (u.km/u.s/u.Mpc)))
 
def COMAP_Pei(
        Omega_field = experiments['COMAP']['Omega_field'],
        Tsys = 40 * u.K,
        tsurvey = 5000 * u.hr,
        Nfeeds = 19
        ):
    nurest = (115.27 * u.GHz)
   
    def _Pei(nuobs):
        return ((Tsys)**2 /Nfeeds / (tsurvey) 
         * (nuobs)**2 * (2 * cu.k_B / cu.c**2 / u.sr)**2 
         * nurest  * (Omega_field.to(u.sr)).value * Gamma_nu(nuobs, nurest)).to((u.kJy/u.sr)**2 * u.Mpc**3)
    nuobs = np.linspace(26,34,1000) * u.GHz
    _Peis = _Pei(nuobs)
    return nuobs, _Peis

def CCAT_prime_Pei():
    #Sato-Polito+20 Table 1.
    zmin = 3.5
    zmax = 8.1
    ##ccat prime
    Omegapix = experiments['CCAT-prime']['Omega_pix']

    Vvox = voxelComovingVolume(zmin, Omegapix, R=100)
    ret = (5.7e4 *u.Jy / u.sr )**2 * Vvox
    ret = ret.to((u.kJy / u.sr)**2 * u.Mpc**3)

    return np.array([420, 420])*u.GHz, np.array([ret.value, ret.value]) * (u.kJy / u.sr)**2 * u.Mpc**3

def HETDEX_Pei(
        PF = 1e-34 * (u.erg / u.s / u.cm**2)**2,
        R = experiments['HETDEX']['R'],
        Opix = experiments['HETDEX']['Omega_pix']
        ):
   
    nurest = 2456.43 * u.THz #Lya
    def _Pei(nuobs):
        return (PF
                * R 
                / nuobs **3 
                / Opix 
                * nurest 
                / u.sr 
                * Gamma_nu(nuobs, nurest))
    nuobs = np.linspace(545,857,1000) * u.THz
    _Peis = _Pei(nuobs)
    return nuobs, _Peis

def SPHEREx_Pei(
        Omegapix = experiments['SPHEREx']['Omega_pix'],
        R = experiments['SPHEREx']['R'],
        Omegasurv = experiments['SPHEREx']['Omega_field'],
        mAB5Sigma = 22,
        nPixEff = 2.,
        ):
    f5Sigma = 10.**((8.9-mAB5Sigma)/2.5) * u.Jy
    sigmaFSource = f5Sigma / 5.
    sigmaFPixel = sigmaFSource / np.sqrt(nPixEff)
    sigmaIPixel = sigmaFPixel / Omegapix 
    nurest = 2456.43 * u.THz #Lya


    def _Pei(nuobs):
        return (sigmaIPixel**2
                / R * nurest/nuobs
                * Omegapix / u.sr
                * Gamma_nu(nuobs, nurest))
    nuobs = np.linspace(270,400,1000) * u.THz
    _Peis = _Pei(nuobs)
    return nuobs, _Peis

def CHIME_Pei(Ofield = experiments['CHIME']['Omega_field'],
              Tsys = 55*u.K,
              Ssky = 31000 * u.deg**2,
              ttot = 1*u.yr,
              Nant = 256,
              npol = 2,
              Ncyl = 4,
              wcyl = 20*u.m,
              dant = 0.3048 * u.m,
              eta = 0.7,
              bmin = 0.3048 * u.m,
              bmax = 102 * u.m
              ):

    nurest = cu.c/(21.106114054160 * u.cm)
    dnu = (800 * u.MHz - 400 * u.MHz)/(256 * 4)

    nuobs = np.linspace(617, 710,1000) * u.MHz
    lobs = cu.c/nuobs
    zobs = (nurest /nuobs - 1).si

    
    lcyl = dant * Nant

    Ae = eta * lcyl/Nant * wcyl
    Obeam = ((lobs**2/Ae)) * u.sr
    du2 = (2*np.pi)**2/Obeam
    du = np.sqrt(du2)
    
    SFOV = (90*u.deg * (lobs/wcyl *(u.rad)))
    umin = bmin / lobs * 1/u.rad
    umax = bmax / lobs * 1/u.rad
    Ndish = Nant * Ncyl
    
    tp = ttot * SFOV / Ofield
    n_u_z = Ndish * (Ndish - 1) / (2 * np.pi) / (umax**2 - umin**2)
    N_dU = (n_u_z * du**2).si
    N_dU_fid = N_dU[0]
    
    PeT = (Tsys ** 2 
          / tp
          / dnu 
           / npol 
          / (N_dU_fid))
    PeV = (PeT * Obeam**2 *
          (2 * cu.k_B * nuobs**2 / cu.c**2)**2 / u.sr**2).to(u.mJy**2)   

    PeI_dV = 1/(2*np.pi)**4 * np.pi * du**2 * (umax**2 - umin**2) * PeV

    PeI = (PeI_dV 
           * dnu / nuobs 
           * nurest/nuobs 
           * Obeam/u.sr * Gamma_nu(nuobs, nurest)).to((u.kJy/u.sr)**2*u.Mpc**3)
    return nuobs, PeI

#######

Pei_dict = {
        'CCAT-prime': CCAT_prime_Pei,
        'COMAP': COMAP_Pei,
        'SPHEREx':SPHEREx_Pei,
        'HETDEX':HETDEX_Pei,
        'CHIME': CHIME_Pei
        }

for _e in experiments:
    experiments[_e]['f_Pei'] = Pei_dict[_e]
