import numpy as np
import PFM.harmonics.shcoeffs as sh
import PFM.harmonics.tfcoeffs as tf
import PFM.util as util
from scipy import special


class Detector:
    """A Detector is specified by its optical axis, numerical aperture, 
    the index of refraction of the sample, and precence of a polarizer.

    By default we use the paraxial approximation.
    """

    def __init__(self, optical_axis=[0, 0, 1], na=0.8, n=1.33, sigma_ax=1.0,
                 polarizer=False, paraxial=True, detect_all=False):
        self.optical_axis = optical_axis
        self.na = na
        self.n = n
        self.alpha = self.na / self.n
        self.polarizer = polarizer
        self.paraxial = paraxial
        self.detect_all = detect_all
        self.sigma_ax = sigma_ax

    def h(self, x, y, z):
        if self.detect_all:
            return tf.TFCoeffs([[1.0, 0, 0, 0, 0, 0], 6 * [0], 6 * [0]])

        # Find cylindrical coordinates based on view
        if self.optical_axis == [0, 0, 1]:  # z-detection
            r = np.sqrt(x ** 2 + y ** 2)
            phi = np.arctan2(y, x)
            z_ax = z
        elif self.optical_axis == [1, 0, 0]:  # x-detection
            r = np.sqrt(y ** 2 + z ** 2)
            phi = np.arctan2(z, y)
            z_ax = x

        a1 = self.a1
        a2 = self.a2
        n0 = [a1(r) + (self.alpha ** 2 / 4) * a2(r), 0, 0, (-a1(r) + (self.alpha ** 2 / 2) * a2(r)) / np.sqrt(5), 0, 0]
        n_2 = 6 * [0]
        n2 = 6 * [0]
        if self.polarizer:
            n_2 = [2 * a2(r) * np.sin(2 * phi), -np.sqrt(0.6) * a1(r), 0, (4.0 / np.sqrt(5)) * a2(r) * np.sin(2 * phi),
                   0, 0]
            n2 = [2 * a2(r) * np.cos(2 * phi), 0, 0, (4.0 / np.sqrt(5)) * a2(r) * np.cos(2 * phi), 0,
                  np.sqrt(0.6) * a1(r)]
        if self.optical_axis == [1, 0, 0]:  # x-detection
            n0 = sh.SHCoeffs(n0).rotate().coeffs
            n_2 = sh.SHCoeffs(n_2).rotate().coeffs
            n2 = sh.SHCoeffs(n2).rotate().coeffs

        return tf.TFCoeffs([n0, n_2, n2])

    def H(self, x, y, z):
        if self.detect_all:
            return tf.TFCoeffs([[1.0, 0, 0, 0, 0, 0], 6 * [0], 6 * [0]])

        # Find cylindrical coordinates based on view
        if self.optical_axis == [0, 0, 1]:  # z-detection
            nu = np.sqrt(x ** 2 + y ** 2)
            phi_nu = np.arctan2(y, x)
            z_ax = z
        elif self.optical_axis == [1, 0, 0]:  # x-detection
            nu = np.sqrt(y ** 2 + z ** 2)
            phi_nu = np.arctan2(z, y)
            z_ax = x

        A1 = self.A1
        A2 = self.A2
        n0 = [A1(nu) + (self.alpha ** 2 / 4) * A2(nu), 0, 0, (-A1(nu) + (self.alpha ** 2 / 2) * A2(nu)) / np.sqrt(5), 0,
              0]
        n_2 = 6 * [0]
        n2 = 6 * [0]
        if self.polarizer:
            # TODO: Update these (unused for now)
            n_2 = [2 * C(nu) * np.sin(2 * phi_nu), -np.sqrt(0.6) * A(nu), 0,
                   (4.0 / np.sqrt(5)) * C(nu) * np.sin(2 * phi_nu), 0, 0]
            n2 = [2 * C(nu) * np.cos(2 * phi_nu), 0, 0, (4.0 / np.sqrt(5)) * C(nu) * np.cos(2 * phi_nu), 0,
                  np.sqrt(0.6) * A(nu)]
        if self.optical_axis == [1, 0, 0]:  # x-detection
            n0 = sh.SHCoeffs(n0).rotate().coeffs
            n_2 = sh.SHCoeffs(n_2).rotate().coeffs
            n2 = sh.SHCoeffs(n2).rotate().coeffs

        ax_weight = np.exp(-(z_ax ** 2) / (2 * (self.sigma_ax ** 2)))
        return sh.SHCoeffs(n0) * ax_weight

    # PSF helper functions
    def a1(self, r):
        if r == 0:
            return 1.0  # TODO Update this
        else:
            return (1 / np.pi) * (special.jn(1, 2 * np.pi * r) / (2 * np.pi * r)) ** 2

    def a2(self, r):
        if r == 0:
            return 0
        else:
            return (2 / np.pi) * (special.jn(2, 2 * np.pi * r) / (2 * np.pi * r)) ** 2

    # OTF helper functions
    def myacos(self, r):
        if isinstance(r, np.ndarray):
            r[np.abs(r) < 2] = 2
        else:
            r = r if abs(r) < 2 else 2
        return np.arccos(np.abs(r / 2))

    def mysqrt(self, r):
        if isinstance(r, np.ndarray):
            r[np.abs(r) < 2] = 2
        else:
            r = r if abs(r) < 2 else 2
        return (np.abs(r / 2)) * np.sqrt(1 - (np.abs(r / 2)) ** 2)

    def A1(self, r):
        return (2 / np.pi) * (self.myacos(r) - self.mysqrt(r))

    def A2(self, r):
        poly = (3.0 - 2.0 * (np.abs(r / 2) ** 2))
        return (2 / np.pi) * (self.myacos(r) - poly * self.mysqrt(r))
