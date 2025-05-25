from nbodykit.lab import *
from nbodykit import style, setup_logging

import matplotlib.pyplot as plt
import numpy as np
Nmesh = 256


def Plin(k):
    return np.abs(k)**(-8)*np.exp(-(k)**4)

mesh = LinearMesh(Plin, Nmesh=Nmesh, BoxSize=5000, seed=43)

fg = mesh.paint(mode='real')
fg += np.min(fg)

fgbar = np.mean(fg)

dfg = fg/fgbar

print(dfg.value.mean())
print(dfg.value.min())
print(dfg.value.max())
 

plt.imshow(mesh.preview(axes=[0,1]))
plt.savefig('figures/016.005_debug.png')



_fname = '/scratch/users/delon/LIMxCMBL/summary_plot/fg_Nmesh_%d.npy'%(Nmesh)

np.save(_fname, dfg.value)
print('fg.value outputted to', _fname)
