import numpy as np
import PFM.util as util
import logging

log = logging.getLogger('log')


class Illuminator:
    """An Illuminator is specified by its optical axis, numerical aperture, 
    the index of refraction of the sample, and polarizer orientation.

    By default we use the paraxial approximation.
    """

    def __init__(self, data, view):
        self.data = data
        self.P = data.P
        self.view = view

    def calc_H(self):
        sh_ills = np.zeros((self.P, 6))
        for p in range(self.P):
            pol = self.data.ill_pols_norm[self.view, p, :]
            sh_ills[p, :] = self.H(pol)

        return sh_ills

    def H(self, pol=None):
        out = []
        for j in range(6):
            l, m = util.j2lm(j)
            theta, phi = util.xyz2tp(*pol)
            if l == 0:
                cc = 1.0
            else:
                cc = 0.4
            out.append(cc * util.spZnm(l, m, theta, phi))
        return np.array(out)
