For the cross spectras we are computing the integral

$$\langle \phi_{\mathbf \ell}(\bullet) \psi_{\mathbf m}(\bullet) \rangle \approx (2\pi)^2\delta^{(D)}(\mathbf \ell + \mathbf m)\times \int_0^\infty d\bar\chi\int_{-1}^1 d\delta \times (2\bar\chi) \times \frac{K_\phi(\bar\chi(1-\delta);\bullet)\times K_\psi(\bar\chi(1+\delta);\bullet)}{\bar\chi^2}\times \int_{k_\parallel} e^{i{k_\parallel}\times 2\bar\chi\delta}P^{\Phi\Psi}(k)$$

The inner integral is tabulate by
1. Running [`000.002.2025-01-13.inner_dkparp_integral.py`](000.002.2025-01-13.inner_dkparp_integral.py) with [`000.002.2025-01-13.inner_dkparp_integral.sh`](000.002.2025-01-13.inner_dkparp_integral.sh)
2. Tabulating the results of all these jobs with [`000.003.2025-01-15.compile_results.ipynb`](000.003.2025-01-15.compile_results.ipynb)

Once the inner integral is tabulate we can compute relevant spectra:
- **CMB lensing spectrum** computation can be found in `001.*`
- **Unfiltered LIM spectrum** [`[CII]`](010.000.2025-02-24.LIM_auto.ipynb)[`[CO]`](010.000.2025-03-11.LIM_auto_CO.ipynb)[`[Lya]`](010.000.2025-03-13.LIM_auto_Lya.ipynb)[`[HI]`](010.000.2025-03-18.LIM_auto_HI.ipynb)
- **Unfiltered LIM x CMB lensing cross spectrum** [`[CII]`](009.013.2025-03-04-Ik-quad-external.ipynb)[`[CO]`](009.015.2025-03-11-Ik-quad-external-CO.ipynb)[`[Lya]`](009.015.2025-03-11-Ik-quad-external-Lya.ipynb)[`[HI]`](009.015.2025-03-18-Ik-quad-external-HI.ipynb)
- **High-passed LIM x CMB lensing cross spectrum** [`009.016.2025-03-28.dblquad_IHiKappa_comb.py`](009.016.2025-03-28.dblquad_IHiKappa_comb.py)
- **High-passed LIM spectrum** [`010.023-03-25-qmc-comb-gpu.py`](010.023-03-25-qmc-comb-gpu.py)
- **Filtered LIM noise spectrum** [`009.010.2025-02-20-comb-cov-bin-mpm-quad.py`](009.010.2025-02-20-comb-cov-bin-mpm-quad.py)
    - Values of white noise amplitude computed in [`LIMxCMBL/experiments.py`](LIMxCMBL/experiments.py) 
    - Mathematica notebook computing the analytical form of the filtered LIM noise spectra in terms of sine- and cosine-integrals can be found at [`008.008.2025-02-17-analytical-eLOeLO.nb`](008.008.2025-02-17-analytical-eLOeLO.nb).

The noise spectra for CMB lensing, store in `LIMxCMBL/N0.npy`, is assumed to be only $N^{(0)}$ which we tabulate from [LensQuEst](https://github.com/DelonShen/LensQuEst) for a SO-like survey. 

With all the ingrediants in place, the SNR is computed for all experiments/lines in [`011.006.2025-03-28-SNR-calculation-generic.ipynb`](011.006.2025-03-28-SNR-calculation-generic.ipynb)
