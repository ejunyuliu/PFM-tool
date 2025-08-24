import numpy as np
from PFM.SPIM.microscope import ill, det
from tqdm import tqdm
import logging
import os

log = logging.getLogger('log')


class Microscope:
    """
    A Microscope represents an experiment that collects a single frame of 
    intensity data.  

    A Microscope is specified by its illumination path (an Illuminator object),
    and its detection path (a Detector object).
    """

    def __init__(self, spang, data, view):
        self.ill = ill.Illuminator(data=data, view=view)
        self.det = det.Detector(spang=spang, data=data, view=view)

        self.J = spang.J
        self.P = data.P

        self.Gaunt = np.load(os.path.join(os.path.dirname(__file__), '../../harmonics/gaunt_l4.npy'))

    def calc_H(self):
        det_mtx = self.det.calc_H()
        ill_mtx = self.ill.calc_H()

        H = np.zeros(det_mtx.shape[1:4] + (self.J, self.P,), dtype=np.complex64)
        from joblib import Parallel, delayed
        Parallel(n_jobs=-1, backend='threading')(
            tqdm([delayed(self.compute_view)(z, det_mtx, ill_mtx, H) for z in range(H.shape[2])]))
        del det_mtx, ill_mtx
        H = H / (np.max(np.abs(H)))
        return H

    def compute_view(self, z, sh_dets, sh_ills, H):
        mat = np.einsum('jls,pxys->pxyjl', self.Gaunt[:, 0:6, 0:6], sh_dets[:, :, :, z, :])
        mat = np.einsum('pxyjl,pl->xyjp', mat, sh_ills)
        H[:, :, z, :, :] = mat
