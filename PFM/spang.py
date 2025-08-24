from PFM import util, viz
import numpy as np
from dipy.data import get_sphere
import logging
import tifffile
from tqdm import tqdm

log = logging.getLogger('log')


class Spang:
    """
    A Spang (short for spatio-angular density) is a representation of a 
    spatio-angular density f(r, s) stored as a 4D array of voxel values 
    and spherical harmonic coefficients [x, y, z, j]. A Spang object is 
    a discretized member of object space U. 
    """

    def __init__(self, f=np.zeros((10, 10, 10, 15), dtype=np.float32),
                 vox_dim=(130, 130, 130), lamb=525, n_samp=1.33, sphere=get_sphere('symmetric724')):
        self.X = f.shape[0]
        self.Y = f.shape[1]
        self.Z = f.shape[2]
        self.J = f.shape[3]

        self.f = f

        self.vox_dim = vox_dim
        self.lamb = lamb
        self.n_samp = n_samp

        self.sphere = sphere
        self.sphere = sphere.subdivide()

        self.N = len(self.sphere.theta)
        self.calc_B()

    def calc_B(self):
        # Calculate odf to sh matrix
        B = np.zeros((self.N, self.J))
        for (n, j), x in np.ndenumerate(B):
            l, m = util.j2lm(j)
            B[n, j] = util.spZnm(l, m, self.sphere.theta[n], self.sphere.phi[n])
        self.B = B
        self.Binv = np.linalg.pinv(self.B, rcond=1e-15)

    def density(self, norm=True):
        if norm:
            return self.f[..., 0] / np.max(self.f[..., 0])
        else:
            return self.f[..., 0]

    def save_tiff(self, filename='sh.tif', data=None):
        if data is None:
            data = self.f

        log.info('Writing ' + filename)
        with tifffile.TiffWriter(filename, imagej=True) as tif:
            if data.ndim == 4:
                dat = np.moveaxis(data, [2, 3, 1, 0], [0, 1, 2, 3])
                tif.save(dat[None, :, :, :, :].astype(np.float32))  # TZCYXS
            elif data.ndim == 3:
                d = np.moveaxis(data, [2, 1, 0], [0, 1, 2])
                tif.save(d[None, :, None, :, :].astype(np.float32))  # TZCYXS

    def read_tiff(self, filename, roi=None):
        log.info('Reading ' + filename)
        with tifffile.TiffFile(filename) as tf:
            self.f = np.ascontiguousarray(np.moveaxis(tf.asarray(), [0, 1, 2, 3], [2, 3, 1, 0]))
            if roi != None:
                self.f = self.f[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1], roi[2][0]:roi[2][1], :]

        self.X = self.f.shape[0]
        self.Y = self.f.shape[1]
        self.Z = self.f.shape[2]

        log.info('Reading Done')

    def visualize(self, out_path='out/', viz_type=['Peak'], scalemap=None, scale=1, interact=True, mag=1, peak_scale=1,
                  bw_invert=False, mask=None, skip_n=1, density_max=1, gamma=1, titles=True, titles_txt=None,
                  scalebar=True, profiles=None,
                  outer_box=True, color_axes=True, color_axes_lw_rat=1, camtilt=False, det_axes=None, azimuth=0,
                  elevation=0, n_frames=18, select_frame=None, video=False, rois=None, rois_color=None, lines=None):
        self.X = self.f.shape[0]
        self.Y = self.f.shape[1]
        self.Z = self.f.shape[2]
        viz.visualize(self, out_path=out_path, viz_type=viz_type, scalemap=scalemap, scale=scale, interact=interact,
                      mag=mag, peak_scale=peak_scale, bw_invert=bw_invert, mask=mask, skip_n=skip_n,
                      density_max=density_max, gamma=gamma, titles=titles, titles_txt=titles_txt, scalebar=scalebar,
                      outer_box=outer_box, profiles=profiles,
                      color_axes=color_axes, color_axes_lw_rat=color_axes_lw_rat, camtilt=camtilt, det_axes=det_axes,
                      azimuth=azimuth, elevation=elevation, n_frames=n_frames, select_frame=select_frame, video=video,
                      rois=rois, rois_color=rois_color, lines=lines)

    def visualize_4D(self, file_name='ori.tif', mask=None):
        sphere = get_sphere('symmetric724')
        sph_len = len(sphere.theta)
        B = np.zeros((sph_len, 15))
        for (n, j), x in np.ndenumerate(B):
            l, m = util.j2lm(j)
            B[n, j] = util.spZnm(l, m, sphere.theta[n], sphere.phi[n])
        Binv = np.linalg.pinv(B, rcond=1e-15)
        BinvT = Binv.T

        def orient2rgb(dirs):
            M = np.array([[1 / np.sqrt(2), 0, 1 / np.sqrt(2)], [0, 1, 0], [-1 / np.sqrt(2), 0, 1 / np.sqrt(2)]])
            vv = np.einsum('cd,nd->nc', M, dirs)
            colors = np.einsum('nc,n->nc', vv, 1 / np.linalg.norm(vv, axis=1))
            return np.abs(colors)

        def compute_image4D(image_4D, mask, z):
            mask_z = mask[..., z]

            masked_sh = self.f[mask_z, z]  # Assemble masked sh
            masked_radii = np.einsum('vj,pj->vp', Binv.T, masked_sh)  # Radii
            masked_radii = np.abs(masked_radii)

            index = np.argmax(masked_radii, axis=0)
            peak_vals = np.amax(masked_radii, axis=0)
            del masked_radii
            peak_dirs = sphere.vertices[index]

            peak_colors = orient2rgb(peak_dirs)
            image_4D[mask_z, z] = peak_colors

        if mask is None:
            mask = np.ones((self.X, self.Y, self.Z), dtype=np.bool)

        image_4D = np.zeros((self.X, self.Y, self.Z, 3))
        from joblib import Parallel, delayed
        Parallel(n_jobs=-1, backend='threading')(
            tqdm([delayed(compute_image4D)(image_4D, mask, z) for z in range(self.Z)]))

        image_4D[mask] = np.einsum('nc,n->nc', image_4D[mask], self.density()[mask])

        if file_name is not None:
            self.save_tiff(filename=file_name, data=image_4D)
        return image_4D

    def visualize_4D_OP(self, file_name='ori.tif', mask=None):
        def compute_OP(OP, mask, vec, z):
            def vec2sft(vec):
                t, p = util.xyz2tp(vec[0], vec[1], vec[2])

                sft = np.zeros(6)
                sft[0] = util.spZnm(0, 0, t, p)
                sft[1] = util.spZnm(2, -2, t, p)
                sft[2] = util.spZnm(2, -1, t, p)
                sft[3] = util.spZnm(2, 0, t, p)
                sft[4] = util.spZnm(2, 1, t, p)
                sft[5] = util.spZnm(2, 2, t, p)

                return sft

            sft = vec2sft(vec)
            sft = sft[1:6]

            mask_z = mask[..., z]
            sh_list = self.f[mask_z, z, :]
            sh_norm = np.einsum('nm,n->nm', sh_list[..., 1:6], 1 / sh_list[..., 0])
            op_list = np.sqrt(4 * np.pi / 5) * np.einsum('nm,m->n', sh_norm, sft)
            OP[mask_z, z] = op_list

        image_4D_OP = np.zeros((self.X, self.Y, self.Z, 3))
        if mask is None:
            mask = np.ones((self.X, self.Y, self.Z), dtype=bool)

        vec0 = [1, 0, 0]
        vec1 = [0, 1, 0]
        vec2 = [0, 0, 1]

        from joblib import Parallel, delayed
        Parallel(n_jobs=-1, backend='threading')(
            tqdm([delayed(compute_OP)(image_4D_OP[..., 0], mask, vec0, z) for z in range(self.Z)]))
        Parallel(n_jobs=-1, backend='threading')(
            tqdm([delayed(compute_OP)(image_4D_OP[..., 1], mask, vec1, z) for z in range(self.Z)]))
        Parallel(n_jobs=-1, backend='threading')(
            tqdm([delayed(compute_OP)(image_4D_OP[..., 2], mask, vec2, z) for z in range(self.Z)]))

        den = self.f[mask, 0]
        sh_list = image_4D_OP[mask]
        image_4D_OP[mask] = np.einsum('nc,n->nc', (sh_list - sh_list.min()) / (sh_list.max() - sh_list.min()),
                                      (den - den.min()) / (den.max() - den.min()))

        if file_name is not None:
            self.save_tiff(filename=file_name, data=image_4D_OP)
        return image_4D_OP

    def visualize_4D_Peak_MP(self, file_name='ori.tif', mask=None):
        sphere = get_sphere('symmetric724')
        sph_len = len(sphere.theta)
        B = np.zeros((sph_len, 15))
        for (n, j), x in np.ndenumerate(B):
            l, m = util.j2lm(j)
            B[n, j] = util.spZnm(l, m, sphere.theta[n], sphere.phi[n])
        Binv = np.linalg.pinv(B, rcond=1e-15)
        BinvT = Binv.T

        def compute_MP(Binv_MP, BinvT, sphere, n):
            vec = sphere.vertices[n, :]
            MP = np.einsum('vp,p->v', sphere.vertices, vec)
            MP = np.abs(MP)
            Binv_MP[n, :] = np.einsum('vj,v->j', BinvT, MP)

        BinvT_MP = np.zeros_like(BinvT)
        from joblib import Parallel, delayed
        Parallel(n_jobs=-1, backend='threading')(
            tqdm([delayed(compute_MP)(BinvT_MP, BinvT, sphere, n) for n in range(sph_len)]))

        def orient2rgb(v):
            # M = np.array([[1 / np.sqrt(2), 0, 1 / np.sqrt(2)], [0, 1, 0], [-1 / np.sqrt(2), 0, 1 / np.sqrt(2)]])
            # vv = np.einsum('cv,nv->nc', M, v)
            vv = v
            vv = np.einsum('nc,n->nc', vv, 1 / np.linalg.norm(vv, axis=1))
            return np.abs(vv)

        def compute_Peak_MP(image_Peak_MP, BinvT_MP, sphere, mask, z):
            mask_z = mask[..., z]
            masked_sh = self.f[mask_z, z, :]
            masked_radii = np.einsum('vj,nj->nv', BinvT_MP, masked_sh)
            index = np.argmax(masked_radii, axis=1)
            peak_dirs = sphere.vertices[index]
            image_Peak_MP[mask_z, z, ...] = orient2rgb(peak_dirs)

        image_Peak_MP = np.zeros((self.X, self.Y, self.Z, 3))
        from joblib import Parallel, delayed
        Parallel(n_jobs=-1, backend='threading')(
            tqdm([delayed(compute_Peak_MP)(image_Peak_MP, BinvT_MP, sphere, mask, z) for z in range(self.Z)]))

        image_Peak_MP[mask] = np.einsum('nc,n->nc', image_Peak_MP[mask], self.density()[mask])

        if file_name is not None:
            self.save_tiff(filename=file_name, data=image_Peak_MP)
        return image_Peak_MP

    def GFA(self, mask=None):
        if mask is None:
            k = self.f ** 2
            gfa = np.sqrt(1 - k[..., 0] / k.sum(axis=-1))
        else:
            gfa = np.zeros_like(self.density())
            k = self.f[mask, :] ** 2
            gfa[mask] = np.sqrt(1 - k[..., 0] / k.sum(axis=-1))

        return gfa
