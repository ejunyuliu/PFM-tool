import numpy as np
import os
import tifffile
import logging

log = logging.getLogger('log')


class Data:
    """
    A Data object represents all of the data a multiview polarized light 
    microscope can collect stored in a 5D array of values [x, y, z, pol, view].
    A Data object is a discretized member of data space V. 
    """

    def __init__(self, g=np.zeros((10, 10, 10, 4, 2), dtype=np.float32),
                 vox_dim=[130, 130, 130],
                 ill_fwhm=[2000, 2000], det_nas=[1.1, 0.67],
                 det_axes=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                    [[0, 0, 1], [0, -1, 0], [1, 0, 0]]]),
                 ill_pols=np.array([[[0, 0, -1], [0, 1, -1], [0, 1, 0], [0, 1, 1]],
                                    [[1, 0, 0], [1, 1, 0], [0, 1, 0], [-1, 1, 0]]]),
                 det_pols=None):
        self.g = g
        self.X = g.shape[0]
        self.Y = g.shape[1]
        self.Z = g.shape[2]
        self.vox_dim = vox_dim

        self.ill_pols = ill_pols
        if self.ill_pols is not None:
            self.P = ill_pols.shape[1]
            self.V = ill_pols.shape[0]
            self.ill_pols_norm = ill_pols / np.linalg.norm(ill_pols, axis=2)[:, :, None]

        self.det_pols = det_pols
        if self.det_pols is not None:
            self.P = det_pols.shape[1]
            self.V = det_pols.shape[0]
            self.det_pols_norm = det_pols / np.linalg.norm(det_pols, axis=2)[:, :, None]

        self.ill_fwhm = ill_fwhm
        self.det_nas = det_nas

        self.det_axes = det_axes

    def save_tiff(self, filename='sh.tif', data=None):
        if data is None:
            data = self.g

        log.info('Writing ' + filename)
        with tifffile.TiffWriter(filename, imagej=True) as tif:
            if data.ndim == 4:
                dat = np.moveaxis(data, [2, 3, 1, 0], [0, 1, 2, 3])
                tif.save(dat[None, :, :, :, :].astype(np.float32))  # TZCYXS
            elif data.ndim == 3:
                d = np.moveaxis(data, [2, 1, 0], [0, 1, 2])
                tif.save(d[None, :, None, :, :].astype(np.float32))  # TZCYXS

    def read_tiff(self, folder, roi=None, format='diSPIM', name='0', bkg=10, padding=None):
        if format == 'diSPIM-2Vx3P':
            data_list = []
            for i, view in enumerate(['SPIMA', 'SPIMB']):
                filename = folder + view + '/' + view + '_' + name + '.tif'
                data = tifffile.imread(filename)  # ZPYX
                if data.ndim == 3:
                    data = data[:, None, :, :]
                data = data.transpose((3, 2, 0, 1))
                if roi is not None:
                    data = data[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1], roi[2][0]:roi[2][1], :]
                if data.dtype == np.uint16:
                    data[data < bkg] = bkg
                    data = (data / np.iinfo(np.uint16).max).astype(np.float32)
                data_list.append(data)

            self.g = np.zeros(data_list[0].shape[0:4] + (2,))
            for i, view in enumerate(['SPIMA', 'SPIMB']):
                self.g[..., i] = data_list[i]

        if format == 'diSPIM-2Vx3Tx7P':
            data_list = []
            for i, view in enumerate(['SPIMA', 'SPIMB']):
                data_list_tmp = []
                for j, tilt in enumerate(['0', '1', '-1']):
                    filename = folder + view + '/' + view + '_Tilt_' + tilt + '.tif'
                    data = tifffile.imread(filename)  # ZPYX
                    data = data.transpose((3, 2, 0, 1))
                    if roi is not None:
                        data = data[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1], roi[2][0]:roi[2][1], :]
                    if data.dtype == np.uint16:
                        data[data < bkg] = bkg
                        data = (data / np.iinfo(np.uint16).max).astype(np.float32)
                    data_list_tmp.append(data)
                data_list.append(data_list_tmp)

            self.g = np.zeros(data_list[0][0].shape[0:3] + (21, 2,))
            for i, view in enumerate(['SPIMA', 'SPIMB']):
                for j, tilt in enumerate(['0', '1', '-1']):
                    self.g[:, :, :, 7 * j:7 * (j + 1), i] = data_list[i][j]

        if padding is not None:
            self.g = np.pad(self.g, ((padding, padding), (padding, padding), (padding, padding), (0, 0), (0, 0)),
                            'reflect')

    def read_calibration(self, folder, XY_range=15):
        self.g = np.zeros((2 * XY_range, 2 * XY_range, 30, 21, 2), dtype=np.float32)
        for i, view in enumerate(['SPIMA', 'SPIMB']):
            for j, tilt in enumerate(['0', '1', '-1']):
                filename = folder + view + '/' + view + '_Tilt_' + tilt + '_0.tif'
                data = tifffile.imread(filename)
                data = data.transpose((2, 1, 0))  # XYZ
                data = data.reshape(data.shape[0:2] + (7, 30))  # XYPZ
                data = data.transpose((0, 1, 3, 2))

                X_mid = int(data.shape[0] / 2)
                Y_mid = int(data.shape[0] / 2)
                self.g[:, :, :, 7 * j:7 * (j + 1), i] = data[X_mid - XY_range:X_mid + XY_range,
                                                        Y_mid - XY_range:Y_mid + XY_range, ...]  # XYZPV

    def calibration_fit(self):
        offsets = []
        for j in range(2):  # Views
            for k in range(3):  # Tilts
                x0 = np.linspace(0, 180, 1000)
                x0t = np.array(np.deg2rad(x0))
                x1 = [0, 45, 60, 90, 120, 135, 180]
                x1t = np.array(np.deg2rad(x1))

                data_subset = self.g[..., 7 * k:7 * (k + 1), j]
                means = np.mean(data_subset, axis=(0, 1, 2))
                y = means  # means

                # Least squares fit
                A = np.zeros((len(x1t), 3))
                A[:, 0] = 1
                A[:, 1] = np.cos(2 * x1t)
                A[:, 2] = np.sin(2 * x1t)
                abc = np.linalg.lstsq(A, y, rcond=None)[0]

                def abc2theta(abc, theta):
                    return abc[0] + abc[1] * np.cos(2 * theta) + abc[2] * np.sin(2 * theta)

                y_lst = np.array([abc2theta(abc, xx) for xx in x0t])

                offset = x0[np.argmax(y_lst)]
                offsets.append(offset)

        return np.mean(offsets)
