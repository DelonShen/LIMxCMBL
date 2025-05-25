import sys
Nmesh = int(sys.argv[1])
snapno = 0
import matplotlib.pyplot as plt

import pyccl as ccl
from LIMxCMBL.init import *
import h5py
import hdf5plugin
import types
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D, art3d  # NOQA

#from https://stackoverflow.com/questions/48672663/matplotlib-render-all-internal-voxels-with-alpha
def voxels(self, *args, **kwargs):

    if len(args) >= 3:
        # underscores indicate position only
        def voxels(__x, __y, __z, filled, **kwargs):
            return (__x, __y, __z), filled, kwargs
    else:
        def voxels(filled, **kwargs):
            return None, filled, kwargs

    xyz, filled, kwargs = voxels(*args, **kwargs)

    # check dimensions
    if filled.ndim != 3:
        raise ValueError("Argument filled must be 3-dimensional")
    size = np.array(filled.shape, dtype=np.intp)

    # check xyz coordinates, which are one larger than the filled shape
    coord_shape = tuple(size + 1)
    if xyz is None:
        x, y, z = np.indices(coord_shape)
        
    def _broadcast_color_arg(color, name):
        if np.ndim(color) in (3, 4):
            # 3D array of strings, or 4D array with last axis rgb
            if np.shape(color)[:3] != filled.shape:
                raise ValueError(
                    "When multidimensional, {} must match the shape of "
                    "filled".format(name))
            return color
        else:
            raise ValueError("Invalid {} argument".format(name))

    # intercept the facecolors, handling defaults and broacasting
    facecolors = kwargs.pop('facecolors', None)
    if facecolors is None:
        facecolors = self._get_patches_for_fill.get_next_color()
    facecolors = _broadcast_color_arg(facecolors, 'facecolors')

    # broadcast but no default on edgecolors
    edgecolors = kwargs.pop('edgecolors', None)
    edgecolors = _broadcast_color_arg(edgecolors, 'edgecolors')

    # include possibly occluded internal faces or not
    internal_faces = kwargs.pop('internal_faces', False)

    # always scale to the full array, even if the data is only in the center
    self.auto_scale_xyz(x, y, z)

    # points lying on corners of a square
    square = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [1, 0, 0]
    ], dtype=np.intp)

    voxel_faces = defaultdict(list)

    def permutation_matrices(n):
        """ Generator of cyclic permutation matices """
        mat = np.eye(n, dtype=np.intp)
        for i in range(n):
            yield mat
            mat = np.roll(mat, 1, axis=0)

    for permute in permutation_matrices(3):
        pc, qc, rc = permute.T.dot(size)
        pinds = np.arange(pc)
        qinds = np.arange(qc)
        rinds = np.arange(rc)

        square_rot = square.dot(permute.T)

        for p in pinds:
            for q in qinds:
                p0 = permute.dot([p, q, 0])
                i0 = tuple(p0)
                if filled[i0]:
                    voxel_faces[i0].append(p0 + square_rot)

                # draw middle faces
                for r1, r2 in zip(rinds[:-1], rinds[1:]):
                    p1 = permute.dot([p, q, r1])
                    p2 = permute.dot([p, q, r2])
                    i1 = tuple(p1)
                    i2 = tuple(p2)
                    if filled[i1] and (internal_faces or not filled[i2]):
                        voxel_faces[i1].append(p2 + square_rot)
                    elif (internal_faces or not filled[i1]) and filled[i2]:
                        voxel_faces[i2].append(p2 + square_rot)

                # draw upper faces
                pk = permute.dot([p, q, rc-1])
                pk2 = permute.dot([p, q, rc])
                ik = tuple(pk)
                if filled[ik]:
                    voxel_faces[ik].append(pk2 + square_rot)

    # iterate over the faces, and generate a Poly3DCollection for each voxel
    polygons = {}
    for coord, faces_inds in voxel_faces.items():
        # convert indices into 3D positions
        if xyz is None:
            faces = faces_inds
        else:
            faces = []
            for face_inds in faces_inds:
                ind = face_inds[:, 0], face_inds[:, 1], face_inds[:, 2]
                face = np.empty(face_inds.shape)
                face[:, 0] = x[ind]
                face[:, 1] = y[ind]
                face[:, 2] = z[ind]
                faces.append(face)

        poly = art3d.Poly3DCollection(faces,
            facecolors=facecolors[coord],
            edgecolors=edgecolors[coord],
            **kwargs
        )
        self.add_collection3d(poly)
        polygons[coord] = poly

    return polygons


_fname = '/scratch/users/delon/LIMxCMBL/summary_plot/snapno_%d_matter_Nmesh_%d_cutout.npy'%(snapno, Nmesh)

colors = np.load(_fname)

print('time to plot')
fig = plt.figure(dpi=800)



ax = fig.add_subplot(projection='3d')

x, y, z = np.indices((Nmesh, Nmesh, Nmesh))

_center = Nmesh / 2
perp_length = int(Nmesh // 1.618 / 2)
filled = ((x >=  _center - perp_length) 
          & (x <=  _center + perp_length)
          & (z >=  _center - perp_length)
          & (z <=  _center + perp_length))

ax.voxels = types.MethodType(voxels, ax)

#filled = np.ones((Nmesh,Nmesh,Nmesh))
ax.voxels(filled, 
          facecolors=colors, 
          edgecolors=colors,
          internal_faces=True,
         lw=0,)


_min = _center - perp_length
_max = _center + perp_length + 1
ax.set_xlim([_min, _max])
ax.set_zlim([_min, _max])
ax.set_ylim([0, Nmesh])

edges_kw = dict(color='k', linewidth=3, zorder=1e3, solid_capstyle='round')
#external
ax.plot([_min, _max], [0, 0], _min, **edges_kw)
ax.plot([_max,_max], [0, Nmesh], _min, **edges_kw)
ax.plot([_min, _min], [0, 0], [_min, _max], **edges_kw)
ax.plot([_min, _max], [Nmesh, Nmesh], _max, **edges_kw)
ax.plot([_min, _min], [0, Nmesh],[_max, _max], **edges_kw)
ax.plot([_max, _max], [Nmesh, Nmesh], [_min, _max], **edges_kw)


edges_kw = dict(color='k', linewidth=1, zorder=1e3)
#faceing us
ax.plot([_max, _max], [0, Nmesh],[_max, _max], **edges_kw)
ax.plot([_min, _max], [0, 0], _max, **edges_kw)
ax.plot([_max, _max], [0, 0], [_min, _max], **edges_kw)

edges_kw = dict(color='k', linewidth=0.5, zorder=1e3)
#internal
ax.plot([_min, _min], [0, Nmesh],[_min, _min], **edges_kw)
ax.plot([_min, _max], [Nmesh, Nmesh], _min, **edges_kw)
ax.plot([_min, _min], [Nmesh, Nmesh], [_min, _max], **edges_kw)


ax.axis('off')
#ax.set_xticks([])
#ax.set_yticks([])
#ax.set_zticks([])

ax.set_proj_type('persp', focal_length=0.3)

xlim = ax.get_xlim3d()
ylim = ax.get_ylim3d()
zlim = ax.get_zlim3d()
ax.set_box_aspect((xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]))

ax.view_init(elev=25, azim=-40, roll=0)

#ax.set_rasterized(True)
#plt.savefig('figures/016.002.delta_plus_one_Nmesh_%d.pdf'%(Nmesh), bbox_inches='tight')

plt.savefig('figures/016.002.delta_plus_one_Nmesh_%d.png'%(Nmesh), bbox_inches='tight', pad_inches=0)
