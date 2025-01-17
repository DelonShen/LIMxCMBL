import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl 

# cosmology
h=0.6736
omch2 = 0.1200
ombh2 = 0.02237
Omc = omch2/h**2
Omb = ombh2/h**2
ns = 0.9649
sigma8 =  0.8111

cosmo = ccl.Cosmology(Omega_c=Omc, Omega_b=Omb,
                          h=h, n_s=ns, sigma8=sigma8)


# sampling range
zmax = 20
amax = 1/(zmax+1)
chimax = ccl.comoving_radial_distance(cosmo, amax)


# sampling density
ells = np.logspace(1, np.log10(5000), 100)

eps = 0.3
chibs = np.linspace(10, chimax, 2**6)
deltas = np.logspace(-6, np.log10(1-eps), 2**7)
