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

    def visualize(self, out_path='out/', viz_type=['Peak'], scalemap=None, scale=3, interact=True, mag=1, peak_scale=1,
                  bw_invert=False, mask=None, skip_n=1, density_max=1, gamma=1, titles=False, titles_txt=None,
                  scalebar=False, profiles=None,
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