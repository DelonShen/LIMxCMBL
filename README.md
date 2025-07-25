# Overview 
This repository contains the complete history of numerical calculations for the paper "Direct correlation of line intensity mapping and CMB lensing from lightcone evolution" [[2507.17752]](https://arxiv.org/abs/2507.17752). Sorry for the mess. Specific references to key calculations are outlined below.

---

*If you have any questions about the code, I'm happy to chat. Let me know at [delon@stanford.edu](mailto:delon@stanford.edu)*


# Angular Spectra

The angular spectrum of line-of-sight correlation functions is tabulate by [`000.002`](000.002.2025-01-13.inner_dkparp_integral.py) and [`000.003`](000.003.2025-01-15.compile_results.ipynb)
Once this is tabulate relevant spectra are computed:
- **CMB lensing spectrum** discussion of this computation can be found in [`001.003`](001.003.2025-05-03.not_exact_vs_exact.ipynb) but generally since computing the CMB lensing spectrum is so quick, it is recomputed each time it is needed.
- **Unfiltered LIM spectrum** [`[CII]`](010.000.2025-02-24.LIM_auto.ipynb)[`[CO]`](010.000.2025-03-11.LIM_auto_CO.ipynb)[`[Lya]`](010.000.2025-03-13.LIM_auto_Lya.ipynb)[`[HI]`](010.000.2025-03-18.LIM_auto_HI.ipynb)
- **Unfiltered LIM x CMB lensing cross spectrum** [`[CII]`](009.013.2025-03-04-Ik-quad-external.ipynb)[`[CO]`](009.015.2025-03-11-Ik-quad-external-CO.ipynb)[`[Lya]`](009.015.2025-03-11-Ik-quad-external-Lya.ipynb)[`[HI]`](009.015.2025-03-18-Ik-quad-external-HI.ipynb)
- **High-pass (Foreground) filtered LIM x CMB lensing cross spectrum** [`009.016`](009.016.2025-03-28.dblquad_IHiKappa_comb.py)
    - Same computation with the Limber approximation in [`009.017`](009.017.2025-05-06.IHiKappa_Limber.ipynb)
- **High-pass (Foreground) filtered LIM spectrum** [`010.023-03-25`](010.023-03-25-qmc-comb-gpu.py) for all lines and experiments.
- **High-pass filtered LIM noise spectrum** [`009.010`](009.010.2025-02-20-comb-cov-bin-mpm-quad.py)
    - Mathematica notebook computing the analytical form of the filtered LIM noise spectra in terms of sine- and cosine-integrals can be found at [`008.008`](008.008.2025-02-17-analytical-eLOeLO.nb). Outputs of these notebooks are then converted into python functions in [`LIMxCMBL/noise.py`](LIMxCMBL/noise.py).

The noise spectra for CMB lensing, stored in `data/N0_*`, is assumed to be only $N^{(0)}$ which we tabulate using [LensQuEst](https://github.com/DelonShen/LensQuEst) for Planck and Simons Observatory.

# Detectability
The signal-to-noise calculation in [`011.007`](011.007.2025-05-01-SNR-calc.py) assembles all the computed angular spectra to compute the detectability, specifically $d(\textbf{{\sf SNR}}^2)/d \ln \ell$, of a direct correlation.

# Miscellaneous
File names of figures in the paper, accessible in the TeX Source on arXiv, reference which file generated that figure. For example Fig. 5 of the paper is named `013.000.IHi_kappa.pdf` meaning it was generated by the file indexed as [`013.000`](013.000.2025-03-28-visualize-Ik.ipynb). Also of possible interest is [`LIMxCMBL/experiments.py`](LIMxCMBL/experiments.py) which contains our instrumental noise models for LIM experiments considered in the paper.

Numerics for the toy model can be found in `015.*` and code which generates components of the summary visualization (Fig. 1) can be found in `016.*`
