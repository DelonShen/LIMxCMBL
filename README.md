For the cross spectras we are computing the integral
$$ 
{    \langle \phi_{\mathbf \ell}(\bullet) \psi_{\mathbf m}(\bullet) \rangle \approx 
    (2\pi)^2\delta^{(D)}(\mathbf \ell + \mathbf m)
    \times \int_0^\infty d\bar\chi\int_{-1}^1 d\delta \times (2\bar\chi) 
    \times \frac{K_\phi(\bar\chi(1-\delta);\bullet)\times K_\psi(\bar\chi(1+\delta);\bullet)}{\bar\chi^2}
    \times \int_{k_\parallel} e^{i{k_\parallel}\times 2\bar\chi\delta}P^{\Phi\Psi}(k)}
$$
The inner integral is tabulate by
1. Running `000.002.2025-01-13.inner_dkparp_integral.py` with `000.002.2025-01-13.inner_dkparp_integral.sh`
2. Tabulating the results of all these jobs with `000.003.2025-01-15.compile_results.ipynb`

Once the inner integral is tabulate, computing cross-spectra can be done by plugging in the kernels defined in the paper and implemented in `LIMxCMBL/kernels.py` into the machinery from `LIMxCMBL/cross_spectrum.py`. For example
- **Observable** for our fiducial experimental setup is computed in `002.002.2025-01-17.IHikappa_scratch.ipynb` and for an arbitrary experimental setup in `002.003.2025-01-20.compute_IHi_Kappa.py`
- **CMB lensing power spectrum** computation can be found in `001.*`

The noise spectra for CMB lensing, store in `LIMxCMBL/N0.npy`, is assumed to be only $N^{(0)}$ which we tabulate from [LensQuEst](https://github.com/DelonShen/LensQuEst) for a SO-like survey. The noise spectra for the LIM map with a $k_\parallel$ cut applied is computed with `LIMxCMBL/noise.py`
