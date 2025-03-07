For the cross spectras we are computing the integral

$$\langle \phi_{\mathbf \ell}(\bullet) \psi_{\mathbf m}(\bullet) \rangle \approx (2\pi)^2\delta^{(D)}(\mathbf \ell + \mathbf m)\times \int_0^\infty d\bar\chi\int_{-1}^1 d\delta \times (2\bar\chi) \times \frac{K_\phi(\bar\chi(1-\delta);\bullet)\times K_\psi(\bar\chi(1+\delta);\bullet)}{\bar\chi^2}\times \int_{k_\parallel} e^{i{k_\parallel}\times 2\bar\chi\delta}P^{\Phi\Psi}(k)$$

The inner integral is tabulate by
1. Running [`000.002.2025-01-13.inner_dkparp_integral.py`](000.002.2025-01-13.inner_dkparp_integral.py) with [`000.002.2025-01-13.inner_dkparp_integral.sh`](000.002.2025-01-13.inner_dkparp_integral.sh)
2. Tabulating the results of all these jobs with [`000.003.2025-01-15.compile_results.ipynb`](000.003.2025-01-15.compile_results.ipynb)

Once the inner integral is tabulate we can compute relevant spectra:
- **CMB lensing spectrum** computation can be found in `001.*`
- **Unfiltered LIM x CMB lensing cross spectrum** [`009.013.2025-03-04-Ik-quad-external.ipynb`](009.013.2025-03-04-Ik-quad-external.ipynb)
- **Low-passed LIM x CMB lensing cross spectrum** [`009.011.2025-02-26.ILo-kappa-dbl-quadvec.py`](009.011.2025-02-26.ILo-kappa-dbl-quadvec.py)
- **Unfiltered LIM spectrum** [`010.000.2025-02-24.LIM_auto.ipynb`](010.000.2025-02-24.LIM_auto.ipynb)
- **Low-passed LIM spectrum** [`010.017.2025-03-05-comb-mc-sample-bin-specific.py`](010.017.2025-03-05-comb-mc-sample-bin-specific.py)
- **(Un)filtered LIM noise spectrum** [`009.010.2025-02-20-comb-cov-bin-mpm-quad.py`](009.010.2025-02-20-comb-cov-bin-mpm-quad.py)

The noise spectra for CMB lensing, store in `LIMxCMBL/N0.npy`, is assumed to be only $N^{(0)}$ which we tabulate from [LensQuEst](https://github.com/DelonShen/LensQuEst) for a SO-like survey. 

SNR computed in 
- **CCAT-Prime**: TODO
- **AtLAST**: TODO
