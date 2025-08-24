import numpy as np
from scipy.special import sph_harm
import logging

import os

log = logging.getLogger('log')


# ======================================== SH function ========================================

# SciPy real spherical harmonics with identical interface to SymPy's Znm
# Useful for fast numerical evaluation of Znm
def spZnm(l, m, theta, phi):
    if m > 0:
        return np.real((sph_harm(m, l, phi, theta) +
                        ((-1) ** m) * sph_harm(-m, l, phi, theta)) / (np.sqrt(2)))
    elif m == 0:
        return np.real(sph_harm(m, l, phi, theta))
    elif m < 0:
        return np.real((sph_harm(m, l, phi, theta) -
                        ((-1) ** m) * sph_harm(-m, l, phi, theta)) / (np.sqrt(2) * 1j))


def maxl2maxj(l):
    return int(0.5 * (l + 1) * (l + 2))


# Convert between spherical harmonic indices (l, m) and microscope-index (j)
def j2lm(j):
    if j < 0:
        return None
    l = 0
    while True:
        x = 0.5 * l * (l + 1)
        if abs(j - x) <= l:
            return l, int(j - x)
        else:
            l = l + 2


def xyz2tp(x, y, z):
    arccos_arg = z / np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if np.isclose(arccos_arg, 1.0):  # Avoid arccos floating point issues
        arccos_arg = 1.0
    elif np.isclose(arccos_arg, -1.0):
        arccos_arg = -1.0
    return np.arccos(arccos_arg), np.arctan2(y, x)


def mkdir(filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        log.info('Making folder ' + folder)
        os.makedirs(folder, exist_ok=True)


# ======================================== For pol-diSPIM ========================================

def pols_from_tilt(di0, di1, pol_offset=0, t0A=0, t0B=0, t1A=9, t1B=5, t_1A=-5, t_1B=-7):
    theta_deg = [0, 45, 60, 90, 120, 135, 180]
    theta = np.deg2rad(np.array(theta_deg) - pol_offset)  # Correct degrees
    pols = np.zeros((2, 3 * theta.shape[-1], 3))

    # Tilt 0
    dA = np.deg2rad(t0A)
    dB = np.deg2rad(t0B)
    pols[0, 0:7, 0] = np.sin(dA) * np.cos(theta)
    pols[0, 0:7, 1] = np.cos(dA) * np.cos(theta)
    pols[0, 0:7, 2] = -np.sin(theta)
    pols[1, 0:7, 0] = -np.sin(theta)
    pols[1, 0:7, 1] = np.cos(dB) * np.cos(theta)
    pols[1, 0:7, 2] = np.sin(dB) * np.cos(theta)

    # Tilt 1
    dA = np.deg2rad(t1A)
    dB = np.deg2rad(t1B)
    pols[0, 7:14, 0] = np.sin(dA) * np.cos(theta)
    pols[0, 7:14, 1] = np.cos(dA) * np.cos(theta)
    pols[0, 7:14, 2] = -np.sin(theta)
    pols[1, 7:14, 0] = -np.sin(theta)
    pols[1, 7:14, 1] = np.cos(dB) * np.cos(theta)
    pols[1, 7:14, 2] = np.sin(dB) * np.cos(theta)

    # Tilt -1
    dA = np.deg2rad(t_1A)
    dB = np.deg2rad(t_1B)
    pols[0, 14:21, 0] = np.sin(dA) * np.cos(theta)
    pols[0, 14:21, 1] = np.cos(dA) * np.cos(theta)
    pols[0, 14:21, 2] = -np.sin(theta)
    pols[1, 14:21, 0] = -np.sin(theta)
    pols[1, 14:21, 1] = np.cos(dB) * np.cos(theta)
    pols[1, 14:21, 2] = np.sin(dB) * np.cos(theta)

    return np.stack([pols[0, di0, :], pols[1, di1, :]])
