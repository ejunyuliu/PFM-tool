import numpy as np
import logging
from tqdm import tqdm
import os
from PFM import util
import scipy.ndimage

log = logging.getLogger('log')


class Detector:
    """A Detector is specified by its optical axis, numerical aperture,
    the index of refraction of the sample, and precence of a polarizer.

    By default we use the paraxial approximation.
    """

    def __init__(self, spang, data, view):
        self.spang = spang
        self.data = data
        self.X = spang.X
        self.Y = spang.Y
        self.Z = spang.Z
        self.J = spang.J
        self.P = data.P
        self.V = data.V
        self.lamb = spang.lamb

        self.det_Zaxis = list(data.det_axes[view, 2])
        self.na = data.det_nas[view]
        self.n = spang.n_samp
        self.ls_sigma = data.ill_fwhm[view] / 2.3548

        self.Gaunt_633 = np.load(os.path.join(os.path.dirname(__file__),
                                              '../../harmonics/gaunt_633.npy'))
        self.Gaunt = np.load(os.path.join(os.path.dirname(__file__), '../../harmonics/gaunt_l4.npy'))

        self.rotate = np.array([[1, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0],
                                [0, 0, 0, -1 / 2, 0, np.sqrt(3) / 2],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 0, np.sqrt(3) / 2, 0, 1 / 2]])

    def calc_H(self):
        sh_dets = []

        if self.data.det_pols is None:
            self.pol = None
            sh_dets.append(self.H())

        if self.data.det_pols is not None:
            if self.det_Zaxis == [0, 0, 1]:
                for p in range(self.P):
                    self.pol = self.data.det_pols_norm[0, p, :]
                    sh_dets.append(self.H())

            if self.det_Zaxis == [1, 0, 0]:
                for p in range(self.P):
                    self.pol = self.data.det_pols_norm[1, p, :]
                    sh_dets.append(self.H())

        return np.array(sh_dets)

    def H(self):
        mtx = np.zeros((self.X, self.Y, self.Z, 6), dtype=np.complex64)

        if self.det_Zaxis == [0, 0, 1]:  # z-detection
            rz = np.fft.fftfreq(self.Z, 1 / self.Z) * self.data.vox_dim[2]
            hz = np.exp(-(rz ** 2) / (2 * self.ls_sigma ** 2), dtype=np.float32)

            temp = np.zeros((self.X, self.Y, rz.shape[0], 6), dtype=np.complex64)
            from joblib import Parallel, delayed
            Parallel(n_jobs=-1, backend='threading')(tqdm(
                [delayed(self.compute_sh_det0)(temp, z, rz, hz) for z in range(len(rz))]))

            mtx = temp
            mtx = mtx * 4 * np.pi / 3

            # import tifffile
            # # data = np.fft.fftshift(np.fft.fftn(mtx, axes=(0, 1)), axes=(0, 1, 2))
            # data = np.abs(mtx)
            # data=np.fft.fftshift(data, axes=(0, 1, 2))
            # with tifffile.TiffWriter('H0_det.tif', imagej=True) as tif:
            #     if data.ndim == 4:
            #         dat = np.moveaxis(data, [2, 3, 1, 0], [0, 1, 2, 3])
            #         tif.save(dat[None, :, :, :, :].astype(np.float32))  # TZCYXS
            #
            # print(1)

        if self.det_Zaxis == [1, 0, 0]:  # x-detection
            rx = np.fft.fftfreq(self.X, 1 / self.X) * self.data.vox_dim[0]
            hx = np.exp(-(rx ** 2) / (2 * self.ls_sigma ** 2), dtype=np.float32)

            temp = np.zeros((rx.shape[0], self.Y, self.Z, 6), dtype=np.complex64)
            from joblib import Parallel, delayed
            Parallel(n_jobs=-1, backend='threading')(
                tqdm([delayed(self.compute_sh_det1)(temp, x, rx, hx) for x in range(len(rx))]))

            mtx = temp
            mtx = mtx * 4 * np.pi / 3
            mtx = np.einsum('rs,xyzs->xyzr', self.rotate, mtx)

        mtx = np.fft.rfftn(np.real(mtx), axes=(0, 1, 2))
        return mtx

    def compute_sh_det0(self, mtx, z, rz, hz):
        e = [1, -1, 0]
        for j in range(3):
            for j_ in range(3):
                Glm = self.Gaunt_633[:, e[j] + 1, e[j_] + 1]
                mtx[:, :, z, :] = mtx[:, :, z, :] + np.einsum('s,xy->xys', Glm, self.cal_B_mtx(j, j_, rz[z]))
        mtx[:, :, z, :] = mtx[:, :, z, :] * hz[z]

    def compute_sh_det1(self, mtx, x, rx, hx):
        e = [1, -1, 0]
        for j in range(3):
            for j_ in range(3):
                Glm = self.Gaunt_633[:, e[j] + 1, e[j_] + 1]
                mtx[x, :, :, :] = mtx[x, :, :, :] + np.einsum('s,yz->yzs', Glm, self.cal_B_mtx(j, j_, rx[x]))
        mtx[x, :, :, :] = mtx[x, :, :, :] * hx[x]

    def cal_B_mtx(self, j, j_, ro):
        mtx = self.cal_beta_mtx(0, j, ro) * self.cal_beta_mtx(0, j_, ro).conjugate() + \
              self.cal_beta_mtx(1, j, ro) * self.cal_beta_mtx(1, j_, ro).conjugate()
        return mtx

    # def cal_beta_mtx(self, i, j, ro):
    #     g = self.list_g()[i][j]
    #
    #     if self.det_Zaxis == [0, 0, 1]:  # z-detection
    #         vm = self.n / self.lamb
    #         vc = self.na * 2 / self.lamb
    #         dx = np.fft.rfftfreq(self.X, d=self.data.vox_dim[0])
    #         dy = np.fft.rfftfreq(self.Y, d=self.data.vox_dim[1])
    #
    #         mx, my = np.meshgrid(dx, dy)
    #         mx, my = mx.T, my.T
    #         nu = np.sqrt(mx ** 2 + my ** 2)
    #         nu[nu >= vc / 2] = vm - np.finfo(np.float32).eps
    #         nu_phi = np.arctan2(my, mx)
    #         A_mat = self.A(nu, vm)
    #         A_mat[nu >= vc / 2] = 0
    #         g_mat = g(nu, nu_phi, vm)
    #         Phi_mat = self.Phi(nu, ro, vm)
    #         M = self.M(i)
    #
    #         temp = A_mat * g_mat * Phi_mat * M
    #         xstart = slice(0, (self.X // 2) + 1)
    #         xend = slice(None, -(self.X // 2), -1)
    #         ystart = slice(0, (self.Y // 2) + 1)
    #         yend = slice(None, -(self.Y // 2), -1)
    #         mtx = np.zeros((self.X, self.Y), dtype=temp.dtype)
    #         mtx[xstart, ystart] = temp
    #         mtx[xend, ystart] = temp[1:-1, :]
    #         mtx[xstart, yend] = temp[:, 1:-1]
    #         mtx[xend, yend] = temp[1:-1, 1:-1]
    #
    #         return np.fft.ifftn(mtx, axes=(0, 1))
    #
    #     if self.det_Zaxis == [1, 0, 0]:  # x-detection
    #         vm = self.n / self.lamb
    #         vc = self.na * 2 / self.lamb
    #
    #         dy = np.fft.rfftfreq(self.Y, d=self.data.vox_dim[1])
    #         dz = np.fft.rfftfreq(self.Z, d=self.data.vox_dim[2])
    #
    #         my, mz = np.meshgrid(dy, dz)
    #         my, mz = my.T, mz.T
    #         nu = np.sqrt(my ** 2 + mz ** 2)
    #         nu[nu >= vc / 2] = vm - np.finfo(np.float32).eps
    #         nu_phi = np.arctan2(mz, my)
    #         A_mat = self.A(nu, vm)
    #         A_mat[nu >= vc / 2] = 0
    #         g_mat = g(nu, nu_phi, vm)
    #         Phi_mat = self.Phi(nu, ro, vm)
    #         M = self.M(i)
    #
    #         temp = A_mat * g_mat * Phi_mat * M
    #         ystart = slice(0, (self.Y // 2) + 1)
    #         yend = slice(None, -(self.Y // 2), -1)
    #         zstart = slice(0, (self.Z // 2) + 1)
    #         zend = slice(None, -(self.Z // 2), -1)
    #         mtx = np.zeros((self.Y, self.Z), dtype=temp.dtype)
    #         mtx[ystart, zstart] = temp
    #         mtx[yend, zstart] = temp[1:-1, :]
    #         mtx[ystart, zend] = temp[:, 1:-1]
    #         mtx[yend, zend] = temp[1:-1, 1:-1]
    #
    #         return np.fft.ifftn(mtx, axes=(0, 1))

    def cal_beta_mtx(self, i, j, ro):
        g = self.list_g()[i][j]

        if self.det_Zaxis == [0, 0, 1]:  # z-detection
            vm = self.n / self.lamb
            vc = self.na * 2 / self.lamb
            dx = np.fft.fftfreq(self.X, d=self.data.vox_dim[0])
            dy = np.fft.fftfreq(self.Y, d=self.data.vox_dim[1])

            mx, my = np.meshgrid(dx, dy)
            mx, my = mx.T, my.T
            nu = np.sqrt(mx ** 2 + my ** 2)
            nu[nu >= vc / 2] = vm - np.finfo(np.float32).eps
            nu_phi = np.arctan2(my, mx)
            A_mat = self.A(nu, vm)
            A_mat[nu >= vc / 2] = 0
            g_mat = g(nu, nu_phi, vm)
            Phi_mat = self.Phi(nu, ro, vm)
            M = self.M(i)

            mtx = A_mat * g_mat * Phi_mat * M

            # import tifffile
            # def save_tiff(data, filename='sh.tif'):
            #     log.info('Writing ' + filename)
            #     with tifffile.TiffWriter(filename, imagej=True) as tif:
            #         if data.ndim == 4:
            #             dat = np.moveaxis(data, [2, 3, 1, 0], [0, 1, 2, 3])
            #             tif.save(dat[None, :, :, :, :].astype(np.float32))  # TZCYXS
            #         elif data.ndim == 3:
            #             d = np.moveaxis(data, [2, 1, 0], [0, 1, 2])
            #             tif.save(d[None, :, None, :, :].astype(np.float32))  # TZCYXS
            #
            # if ro == (np.fft.fftfreq(self.Z, 1 / self.Z) * self.data.vox_dim[2]).min():
            #     data = np.fft.ifftshift(np.fft.ifftn(mtx, axes=(0, 1)), axes=(0, 1))
            #     save_tiff(data=np.real(data[..., None]), filename='beta_mtx_ls.tif')
            #     print(1)

            return np.fft.ifftn(mtx, axes=(0, 1))

        if self.det_Zaxis == [1, 0, 0]:  # x-detection
            vm = self.n / self.lamb
            vc = self.na * 2 / self.lamb

            dy = np.fft.fftfreq(self.Y, d=self.data.vox_dim[1])
            dz = np.fft.fftfreq(self.Z, d=self.data.vox_dim[2])

            my, mz = np.meshgrid(dy, dz)
            my, mz = my.T, mz.T
            nu = np.sqrt(my ** 2 + mz ** 2)
            nu[nu >= vc / 2] = vm - np.finfo(np.float32).eps
            nu_phi = np.arctan2(mz, my)
            A_mat = self.A(nu, vm)
            A_mat[nu >= vc / 2] = 0
            g_mat = g(nu, nu_phi, vm)
            Phi_mat = self.Phi(nu, ro, vm)
            M = self.M(i)

            mtx = A_mat * g_mat * Phi_mat * M

            return np.fft.ifftn(mtx, axes=(0, 1))

    def list_g(self):
        def g00(nu, nu_phi, vm):
            return np.sin(nu_phi) ** 2 + np.cos(nu_phi) ** 2 * np.sqrt(1 - (nu / vm) ** 2)

        def g10(nu, nu_phi, vm):
            return 0.5 * np.sin(nu_phi * 2) * (np.sqrt(1 - (nu / vm) ** 2) - 1)

        def g01(nu, nu_phi, vm):
            return 0.5 * np.sin(nu_phi * 2) * (np.sqrt(1 - (nu / vm) ** 2) - 1)

        def g11(nu, nu_phi, vm):
            return np.cos(nu_phi) ** 2 + np.sin(nu_phi) ** 2 * np.sqrt(1 - (nu / vm) ** 2)

        def g02(nu, nu_phi, vm):
            return nu * np.cos(nu_phi)

        def g12(nu, nu_phi, vm):
            return nu * np.sin(nu_phi)

        return [[g00, g01, g02], [g10, g11, g12]]

    def A(self, nu, vm):
        return (1 - (nu / vm) ** 2) ** (-0.25)

    def Phi(self, nu, roz, vm):
        return np.exp(1.0j * 2 * np.pi * roz * np.sqrt(vm ** 2 - nu ** 2))

    def M(self, i):
        M = []
        if self.pol is None:
            M = 1

        if self.pol is not None:
            if self.det_Zaxis == [0, 0, 1]:
                M = (i == 0) * self.pol[0] + (i == 1) * self.pol[1]
            if self.det_Zaxis == [1, 0, 0]:
                M = (i == 0) * self.pol[1] + (i == 1) * self.pol[2]

        return M
