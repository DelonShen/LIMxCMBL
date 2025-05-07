#basically this script computes d(SNR^2)/N_ell

from LIMxCMBL.init import *
from scipy.integrate import simpson, trapezoid
import pickle
from tqdm import tqdm, trange

from LIMxCMBL.experiments import *

N0_fname = 'N0_so'
_oup_fname = '/scratch/users/delon/LIMxCMBL/LIMBER_SNR_011.007_'+N0_fname+'_'
full_sky = False
if(full_sky):
    _oup_fname += '_full_sky_'

for experiment in experiments:
    zmin = experiments[experiment]['zmin']
    zmax = experiments[experiment]['zmax']
    line_str = experiments[experiment]['line_str']
    
    Omega_field  =  experiments[experiment]['Omega_field'] #rad^2
    if(full_sky):
        Omega_field = 4*np.pi * u.rad**2

    chimin = ccl.comoving_angular_distance(cosmo, 1/(1+zmin))
    chimax = ccl.comoving_angular_distance(cosmo, 1/(1+zmax))
    
    kpar_fundamental = 2*np.pi/(chimax - chimin)

    ell_fundamental = np.sqrt((2*np.pi)**2 / Omega_field.to(u.rad**2))
    ell_fundamental = max(np.array(ell_fundamental), 10)

    ell_max_survey = np.sqrt((np.pi)**2 / experiments[experiment]['Omega_pix'].to(u.rad**2))
    ell_max_survey = np.array(ell_max_survey)


    _, Pei = experiments[experiment]['f_Pei']()
    Pei = np.max(Pei).to(u.Mpc**3  * (u.kJy/u.sr)**2)
    Pei = Pei.value # kJy2 Mpc3 / sr2

    Omega_field = np.array(Omega_field)
    
    print(experiment)
    print('zmin:',zmin)
    print('zmax:',zmax)
    print('Sky coverate[deg2]: %.1f'%(Omega_field/((np.pi/180)**2)))
    print('White noise[kJy2 Mpc3 / sr2]: %.1f'%Pei)
    print('ell sensitivity: %.1f to %.1f'%(ell_fundamental, ell_max_survey))
    
    n_bins = 100
    chi_bin_edges = np.linspace(chimin*(1+1e-8), chimax*(1 - 1e-8), n_bins + 1)
    chi_bin_centers = (chi_bin_edges[1:] + chi_bin_edges[:-1])/2
    dchi_binned = np.mean(np.diff(chi_bin_edges))
    


    I_kappa_fname = '/scratch/users/delon/LIMxCMBL/IHiKappa/'
    I_kappa_fname += '%s_LIMBER_IHik_idx_%d.npy'%(experiment, -1, )
    I_kappa = np.load(I_kappa_fname)
   
    def get_binned(base, n_external = 300):
        external_chis = np.linspace(chimin*(1+1e-8), chimax*(1 - 1e-8), n_external)
        
        oup = np.zeros((100, n_bins, n_bins), dtype=np.float64)
        for i, (l1, r1) in enumerate(zip(chi_bin_edges, chi_bin_edges[1:])):
            for j, (l2, r2) in enumerate(zip(chi_bin_edges, chi_bin_edges[1:])):
                idx1 = np.where((external_chis > l1) & (external_chis <= r1))[0]
                idx2 = np.where((external_chis > l2) & (external_chis <= r2))[0]
                oup[:,i,j] = (np.sum(base[:,
                                          idx1[0]:idx1[-1]+1,
                                          idx2[0]:idx2[-1]+1], 
                                     axis=(1, 2)) / len(idx1) / len(idx2))
        return oup
    
    #get <II>
    
    I_I_unbinned = np.load('/scratch/users/delon/LIMxCMBL/I_auto/'+
                              'I_auto_n_ext_%d_zmin_%.1f_zmax_%.1f.npy'%(3000, 
                                                                         zmin, 
                                                                         zmax))
    
    I_I = get_binned(I_I_unbinned, n_external = 3000)
    
    # get CMB lensing component
    from LIMxCMBL.kernels import get_f_Kkappa
    f_WkD = get_f_Kkappa()
    
    from LIMxCMBL.cross_spectrum import *
    ClKK = d_chib_integral(f_WkD, f_WkD) # dimensionless
    
    
    # beam=1.4, noise=7
    from scipy.interpolate import interp1d
    N0_ells = np.logspace(1, np.log10(5000), 500)
    with open('data/'+N0_fname+'.npy', 'rb') as f:
        print('loaded N0 from', N0_fname)
        N0 = np.load(f)
        
    f_N0 = interp1d(x = N0_ells, y = N0)
    
    
    SNR2_per_mode_noise_dom = {}

    from LIMxCMBL.noise import f_eIeI
    cov  = np.einsum('l  , xy->lxy', (ClKK + f_N0(ells)),  Pei * np.diag(f_eIeI(chi=chi_bin_centers, dchi=dchi_binned, Lambda=0)))
    cov = cov.astype(np.float64)
    
    SNR2_per_mode_noise_dom[0.0] = np.zeros_like(ells)
    for ell_idx in range(len(ells)):
        L = np.linalg.cholesky(cov[ell_idx])
        y = np.linalg.solve(L, I_kappa[ell_idx])
        x = np.linalg.solve(L.T, y)
        SNR2_per_mode_noise_dom[0.0][ell_idx] = np.dot(I_kappa[ell_idx], x)
    
    
    LIDXs = [18, 24]
    if(experiment == 'CHIME'):
        LIDXs[0] = 19

    for Lambda_idx in LIDXs:
        Lambda = Lambdas[Lambda_idx]
        print(Lambda)


        IHi_kappa_fname = '/scratch/users/delon/LIMxCMBL/IHiKappa/'
        IHi_kappa_fname += '%s_LIMBER_IHik_idx_%d.npy'%(experiment, Lambda_idx, )
        IHi_kappa = np.load(IHi_kappa_fname)
        ####################################################
        #eHI eHI############################################
        eComb_fname = '/scratch/users/delon/LIMxCMBL/eHIeHI/mpmath_comb_'
        eComb_fname +='zmin_%.5f_zmax_%.5f_Lambda_idx_%.d_from_quad_nbins_%d.npy'%(zmin, 
                                                                                   zmax, 
                                                                                   Lambda_idx, 
                                                                                   n_bins)
        eComb = np.load(eComb_fname)
        eHIeHI_binned = eComb
        ####################################################
        #noise-dom cov######################################
        cov  = np.einsum('l  , xy->lxy', (ClKK + f_N0(ells)),  Pei * eHIeHI_binned)
        cov = cov.astype(np.float64)
        SNR2_per_mode_noise_dom[Lambda] = np.zeros_like(ells)
        for ell_idx in range(len(ells)):
            L = np.linalg.cholesky(cov[ell_idx])
            y = np.linalg.solve(L, IHi_kappa[ell_idx])
            x = np.linalg.solve(L.T, y)
            SNR2_per_mode_noise_dom[Lambda][ell_idx] = np.dot(IHi_kappa[ell_idx], x)
        ####################################################
        ####################################################

    with open(_oup_fname+experiment+'_noise_dom.pkl', 'wb') as f:
        pickle.dump(SNR2_per_mode_noise_dom, f)
