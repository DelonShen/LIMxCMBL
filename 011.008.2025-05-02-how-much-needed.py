from LIMxCMBL.init import *
from scipy.integrate import simpson, trapezoid
import pickle
from tqdm import tqdm, trange

from LIMxCMBL.experiments import *

N0_fname = 'N0_so'
_oup_fname = '/scratch/users/delon/LIMxCMBL/SNR_011.008_'+N0_fname+'_'



SNR_detect = 5

for experiment in experiments:
    n_bins = 100
    if(experiment == 'SPHEREx'):
        n_bins = 15
    else:
        continue

    zmin = experiments[experiment]['zmin']
    zmax = experiments[experiment]['zmax']
    line_str = experiments[experiment]['line_str']
    
    Omega_field  =  experiments[experiment]['Omega_field'] #rad^2

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
    
    chi_bin_edges = np.linspace(chimin*(1+1e-8), chimax*(1 - 1e-8), n_bins + 1)
    chi_bin_centers = (chi_bin_edges[1:] + chi_bin_edges[:-1])/2
    dchi_binned = np.mean(np.diff(chi_bin_edges))
    
    I_kappa_fname = '/scratch/users/delon/LIMxCMBL/IHiKappa/Ik_'
    I_kappa_fname +='zmin_%.5f_zmax_%.5f_quad_next_%d.npy'%(zmin, 
                                                            zmax, 
                                                            1000)
    
    I_kappa_unbinned = np.load(I_kappa_fname)
    
    I_kappa = np.zeros((len(ells), n_bins))
    external_chis = np.linspace(chimin*(1+1e-8), chimax*(1 - 1e-8), 1000)
    for i, (l1, r1) in enumerate(zip(chi_bin_edges, chi_bin_edges[1:])):
                idx1 = np.where((external_chis > l1) & (external_chis <= r1))[0]
                I_kappa[:,i] = (np.sum(I_kappa_unbinned[:,idx1[0]:idx1[-1]+1,], 
                                     axis=(1)) / len(idx1))
    
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
    
    
    SNR2_per_mode_full = {}

    __oup_fname = '/scratch/users/delon/LIMxCMBL/SNR_011.007_'+N0_fname+'_'
    with open(__oup_fname+experiment+'_full.pkl', 'rb') as f:
        SNR2_per_mode_full=pickle.load(f)
    _Lambdas = np.array(sorted(list(SNR2_per_mode_full.keys())))


    full_y = np.zeros(len(_Lambdas))
    
    for i,Lambda in enumerate(_Lambdas):
        
        def SNR2_for_Omega_sky(Omega_sky, full=True):
            #assume Omega_sky in rad^2
            _ell_fundamental = np.sqrt((2*np.pi)**2 / Omega_sky)
            _ell_fundamental = max(np.array(ell_fundamental), 10)
            _idxs = np.where((ells >= _ell_fundamental) & (ells <= ell_max_survey))
            
            if(full):
                return trapezoid(x = np.log(ells)[_idxs], 
                             y = SNR2_per_mode_full[Lambda][_idxs] 
                             * ells[_idxs] ** 2 
                             / 2 / np.pi) * Omega_sky
            
        def find(full):
            Omega_min = 1e-20
            Omega_max = 4 * np.pi * 100 #not physical to go beyond full sky but for binary search    
    
            tolerance = 1e-6
            max_iterations = 1000
            target_SNR2 = SNR_detect**2
            for iteration in range(max_iterations):
                Omega_mid = (Omega_min + Omega_max) / 2
                SNR2_mid = SNR2_for_Omega_sky(Omega_mid, full=full)
                if abs((SNR2_mid - target_SNR2)/target_SNR2) < tolerance:
                    Omega_sky_solution = Omega_mid
                    break
    
                if SNR2_mid < target_SNR2:
                    Omega_min = Omega_mid
                else:
                    Omega_max = Omega_mid
            else:
                print('couldnt converge')
                assert(1==0)
            return Omega_mid
        full_y[i] = find(full=True)#/(4*np.pi)
        full_y[i] *= (180/np.pi)**2 #deg2

    with open(_oup_fname+experiment+'_full_deg2_detect.pkl', 'wb') as f:
        pickle.dump(full_y, f)

