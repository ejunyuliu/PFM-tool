from PFM import util,spang
from PFM.SPIM import data
from PFM.SPIM.microscope_simplified import ill, det, micro
import numpy as np
import logging
import os

log = logging.getLogger('log')

from tqdm import tqdm


class MultiMicroscope:
    """A MultiMicroscope represents an experiment that collects intensity data 
    under several different conditions (different polarization states or 
    illumination schemes).

    A MultiMicroscope mainly consists of a list of Microscopes.
    """

    def __init__(self, spang, data, sigma_ax=0.25, spang_coupling=True):
        self.Gaunt = np.load(os.path.join(os.path.dirname(__file__), '../../harmonics/gaunt_l4.npy'))

        self.spang = spang
        self.data = data

        self.X = spang.X
        self.Y = spang.Y
        self.Z = spang.Z
        self.J = spang.J
        self.P = data.P
        self.V = data.V

        n_samp = spang.n_samp

        m = []  # List of microscopes

        # Cycle through paths
        ill_optical_axes = [[1, 0, 0], [0, 0, 1]]
        det_optical_axes = [[0, 0, 1], [1, 0, 0]]

        for i, det_optical_axis in enumerate(det_optical_axes):
            ill_ = ill.Illuminator(optical_axis=ill_optical_axes[i],
                                   na=0, n=n_samp)
            det_ = det.Detector(optical_axis=det_optical_axes[i],
                                na=data.det_nas[i], n=n_samp, sigma_ax=sigma_ax)
            m.append(micro.Microscope(ill=ill_, det=det_, spang_coupling=spang_coupling))  # Add microscope

        self.micros = m
        self.lamb = spang.lamb

        self.sigma_ax = sigma_ax

        self.jmax = m[0].h(0, 0, 0).jmax

    def calc_point_H(self, vx, vy, vz, v):
        out = np.zeros((self.J, self.P))
        for p in range(self.P):
            pol = self.data.pols_norm[v, p, :]
            tf = self.micros[v].H(vx, vy, vz, pol)
            out[:, p] = tf.coeffs
        return out  # return j x p

    def calc_H(self):
        # Transverse transfer function
        log.info('Computing H for view 0')
        dx = np.fft.rfftfreq(self.X, d=self.data.vox_dim[0]) * self.lamb / self.micros[0].det.na
        dy = np.fft.rfftfreq(self.Y, d=self.data.vox_dim[1]) * self.lamb / self.micros[0].det.na
        dz = np.fft.rfftfreq(self.Z, d=self.data.vox_dim[2]) * self.lamb / self.micros[0].det.na
        self.Hxy = np.zeros((dx.shape[0], dy.shape[0], self.J, self.P), dtype=np.float32)

        # Calc illumination once
        sh_ills = []
        for p in range(self.P):
            pol = self.data.ill_pols_norm[0, p, :]
            sh_ills.append(self.micros[0].ill.H(pol))

        # Calc detection and multiply
        for x, nux in enumerate(tqdm(dx)):
            for y, nuy in enumerate(dy):
                for p, sh_ill in enumerate(sh_ills):
                    self.Hxy[x, y, :, p] = (sh_ill * self.micros[0].det.H(nux, nuy, 0)).coeffs
        self.Hxy = self.Hxy / np.max(np.abs(self.Hxy))
        if self.micros[0].spang_coupling:
            self.Hz = np.exp(-(dz ** 2) / (2 * (self.sigma_ax ** 2)), dtype=np.float32)
        else:
            self.Hz = np.ones(dz.shape, dtype=np.float32)
        self.H0 = np.zeros((self.X, self.Y, dz.shape[0], self.J, self.P), dtype=self.Hxy.dtype)
        temp = np.einsum('xyjp,z->xyzjp', self.Hxy, self.Hz)
        xstart = slice(0, (self.X // 2) + 1)
        xend = slice(None, -(self.X // 2), -1)
        ystart = slice(0, (self.Y // 2) + 1)
        yend = slice(None, -(self.Y // 2), -1)
        self.H0[xstart, ystart, ...] = temp
        self.H0[xend, ystart, ...] = temp[1:-1, :]
        self.H0[xstart, yend, ...] = temp[:, 1:-1]
        self.H0[xend, yend, ...] = temp[1:-1, 1:-1]

        log.info('Computing H for view 1')
        dx = np.fft.rfftfreq(self.X, d=self.data.vox_dim[0]) * self.lamb / self.micros[1].det.na
        dy = np.fft.rfftfreq(self.Y, d=self.data.vox_dim[1]) * self.lamb / self.micros[1].det.na
        dz = np.fft.rfftfreq(self.Z, d=self.data.vox_dim[2]) * self.lamb / self.micros[1].det.na
        self.Hyz = np.zeros((dy.shape[0], dz.shape[0], self.J, self.P), dtype=np.float32)

        # Calc illumination once
        sh_ills = []
        for p in range(self.P):
            pol = self.data.ill_pols_norm[1, p, :]
            sh_ills.append(self.micros[1].ill.H(pol))

        # Calc detection and multiply            
        for y, nuy in enumerate(tqdm(dy)):
            for z, nuz in enumerate(dz):
                for p, sh_ill in enumerate(sh_ills):
                    self.Hyz[y, z, :, p] = (sh_ill * self.micros[1].det.H(0, nuy, nuz)).coeffs
        self.Hyz = self.Hyz / np.max(np.abs(self.Hyz))

        if self.micros[0].spang_coupling:
            self.Hx = np.exp(-(dx ** 2) / (2 * (self.sigma_ax ** 2)))
        else:
            self.Hx = np.ones(dx.shape)
        self.H1 = np.zeros((self.X, self.Y, dz.shape[0], self.J, self.P), dtype=self.Hxy.dtype)
        temp = np.einsum('yzjp,x->xyzjp', self.Hyz, self.Hx)
        xstart = slice(0, (self.X // 2) + 1)
        xend = slice(None, -(self.X // 2), -1)
        ystart = slice(0, (self.Y // 2) + 1)
        yend = slice(None, -(self.Y // 2), -1)
        self.H1[xstart, ystart, ...] = temp
        self.H1[xend, ystart, ...] = temp[1:-1, :]
        self.H1[xstart, yend, ...] = temp[:, 1:-1]
        self.H1[xend, yend, ...] = temp[1:-1, 1:-1]

        self.Hxyz = np.stack([self.H0, self.H1], axis=-1)
        self.Hxyz = np.reshape(self.Hxyz, self.H0.shape[0:3] + (15, self.P * self.V,))

    def pinv(self, g, eta):
        log.info('Applying pseudoinverse operator')

        G = np.fft.rfftn(g, axes=(0, 1, 2))
        G2 = np.reshape(G, G.shape[0:3] + (self.P * self.V,))

        from joblib import Parallel, delayed
        F = np.zeros(self.Hxyz.shape[0:3] + (self.J,), dtype=np.complex64)
        Parallel(n_jobs=-1, backend='threading')(
            tqdm([delayed(self.compute_pinv)(F, G2, z, eta) for z in range(self.Hxyz.shape[2])]))

        del G2, G
        f = np.fft.irfftn(F, s=g.shape[0:3], axes=(0, 1, 2))
        return np.real(f)

    def compute_pinv(self, F, G2, z, eta):
        u, s, vh = np.linalg.svd(self.Hxyz[:, :, z, :], full_matrices=False)
        sreg = np.where(s > 1e-7, s / (s ** 2 + eta), 0)
        Pinv = np.einsum('xysv,xyv,xyvd->xysd', u, sreg, vh)
        F[:, :, z, :] = np.einsum('xysd,xyd->xys', Pinv, G2[:, :, z, :])

    def fwd(self, f, snr_Poisson=None, snr_Gaussian=None):
        log.info('Applying forward operator')

        # 3D FT
        F = np.fft.rfftn(f, axes=(0, 1, 2))

        # Tensor multiplication
        from joblib import Parallel, delayed
        G2 = np.zeros(self.Hxyz.shape[0:3] + (self.Hxyz.shape[4],), dtype=np.complex64)
        Parallel(n_jobs=-1, backend='threading')(
            tqdm([delayed(self.compute_fwd)(G2, F, z) for z in range(self.Hxyz.shape[2])]))
        G = np.reshape(G2, G2.shape[0:3] + (self.P,) + (self.V,))

        # 3D IFT
        g = np.real(np.fft.irfftn(G, s=f.shape[0:3], axes=(0, 1, 2)))

        # Apply Poisson and Gaussian noise
        if (snr_Poisson is not None) | (snr_Gaussian is not None):
            g_ = g.copy()

            if snr_Poisson is not None:
                g[g < 0] = 0
                norm = snr_Poisson ** 2 / np.max(g)
                arr_poisson = np.vectorize(np.random.poisson)
                g = arr_poisson(g * norm) / norm

            if snr_Gaussian is not None:
                snr = 10 ** (snr_Gaussian / 10.0)
                xpower = np.mean(g ** 2)
                npower = xpower / snr
                gnoise = np.random.standard_normal(g.shape) * np.sqrt(npower)
                g = g + gnoise

            noise = np.mean((g - g_) ** 2)
            signal = np.mean(g_ ** 2)
            SNR = 10 * np.log10(signal / noise)
            print(SNR)

        g = g / g.max()
        return g

    def compute_fwd(self, G2, F, z):
        G2[:, :, z, :] = np.einsum('xysp,xys->xyp', self.Hxyz[:, :, z, :, :], F[:, :, z, :])

    def save_H(self, filename):
        np.save(filename, self.Hxyz)

    def load_H(self, filename):
        self.Hxyz = np.load(filename)
