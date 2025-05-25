import sys
import matplotlib.pyplot as plt

import pyccl as ccl
from LIMxCMBL.init import *
import h5py
import hdf5plugin
from nbodykit.source.catalog import ArrayCatalog
import types
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D, art3d  # NOQA

for snapno in range(5):
    parts = np.zeros(512**3, dtype=[('Position', ('f4', 3))])
    c = 0
    for i in range(8):
        snap = h5py.File('/home/users/kokron/scratch/0/'+'snapdir_%03d/snap_%03d.%d.hdf5'%(snapno,snapno,i), 'r')
        snapu = snap['PartType1']['Coordinates'][...]
    
        # get 300 Mpc cutout of 1000 Mpc box
        _idxs = np.where(np.all(snapu/1000 <= 650, axis=-1) & np.all(350 <= snapu/1000, axis=-1))[0]
        lensnap = len(_idxs)
    
        parts['Position'][c:c+lensnap] = (snapu[_idxs] / 1000) - 350
        c+=lensnap
    
        snap.close()
    
    parts = parts[:c]
    print('%d / %d particles in zoom in '%(len(parts), 512**3))
    print(np.max(parts['Position']))
    print(np.min(parts['Position']))
    
    _fname = '/scratch/users/delon/LIMxCMBL/summary_plot/positions_snapno_%d.npy'%(snapno)
    np.save(_fname, parts)
    
    f = ArrayCatalog(parts)
    
    for Nmesh in [8,16,32,64,128,256,512]:
        fmesh = f.to_mesh(Nmesh=Nmesh, BoxSize=300)#, compensated=True)
        df = fmesh.paint()
        df/= np.mean(df)
        
        print(df.value.mean())
        print(df.value.min())
        print(df.value.max())
        
        
        colors = np.ones((Nmesh,Nmesh,Nmesh,)+ (4,))
        colors[...,0] = 0
        colors[...,1] = 0
        colors[...,2] = 0
        denom = (np.max(df.value)/3)
        colors[...,3] = np.where(df.value / denom <= 1, df.value / denom, 1)#np.max(df)
        
        filled = np.ones((Nmesh,Nmesh,Nmesh))
        
        _fname = '/scratch/users/delon/LIMxCMBL/summary_plot/snapno_%d_matter_Nmesh_%d_cutout.npy'%(snapno, Nmesh)
        np.save(_fname, colors)
        print('colors outputted to', _fname)
    
        _fname = '/scratch/users/delon/LIMxCMBL/summary_plot/snapno_%d_matter_Nmesh_%d_cutout_value.npy'%(snapno, Nmesh)
        np.save(_fname, df.value)
        print('df.value outputted to', _fname)
