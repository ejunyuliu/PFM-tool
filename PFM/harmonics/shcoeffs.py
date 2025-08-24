import numpy as np
import subprocess
from PFM import util
from PFM.harmonics import gaunt
import matplotlib.pyplot as plt
from PIL import Image
import os
import itertools

# Compute gaunt coefficients
# gaunt.calc_gaunt_tensor('gaunt_l4.npy', lmax=4) # Expensive precomputation
G = np.load(os.path.join(os.path.dirname(__file__), 'gaunt_l4.npy'))


class SHCoeffs:
    """An SHCoeffs object stores real spherical harmonic coefficients for even
    bands. It provides methods for adding, multiplying, and plotting these
    coefficients.

    Inspired by a similar class in SHTOOLS https://shtools.github.io/SHTOOLS/

    Uses the following "lexicographic ordering" of the even spherical harmonics:

    y_0^0, y_2^-2, y_2^-1, y_2^0, y_2^1, y_2^2, y_4^-4, y_4^-3...
    """

    def __init__(self, coeffs):
        self.lmax, mm = util.j2lm(len(coeffs) - 1)
        self.jmax = int(0.5 * (self.lmax + 1) * (self.lmax + 2))
        self.mmax = 2 * self.lmax + 1
        self.rmax = int(self.lmax / 2) + 1

        # Fill the rest of the last band with zeros
        temp = np.zeros(self.jmax)
        temp[:len(coeffs)] = coeffs
        self.coeffs = temp

    def __add__(self, other):
        return SHCoeffs(self.coeffs + other.coeffs)

    def __mul__(self, other):
        # Slow alternative
        # result = gaunt.multiply_sh_coefficients(self.coeffs, other.coeffs)

        if not isinstance(other, SHCoeffs):
            return SHCoeffs(np.array(self.coeffs) * other)

        # Pad inputs
        x1 = np.pad(np.array(self.coeffs), (0, 2 * (self.lmax + 2) + 1), 'constant')
        x2 = np.pad(np.array(other.coeffs), (0, 2 * (self.lmax + 2) + 1), 'constant')

        # Multiply
        result = np.dot(np.dot(G, x1), x2)

        return SHCoeffs(result)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if not isinstance(other, SHCoeffs):
            return SHCoeffs(np.array(self.coeffs) / other)

        # Pad inputs
        x1 = np.pad(np.array(self.coeffs), (0, 15 - len(self.coeffs)), 'constant')
        x2 = np.pad(np.array(other.coeffs), (0, 15 - len(other.coeffs)), 'constant')

        mat = np.einsum('jls,s->jl', G, x2)
        mat_inv = np.linalg.inv(mat)
        result = np.einsum('jl,l->j', mat_inv, x1)

        return SHCoeffs(result)

    def __repr__(self):
        string = 'SHCoeffs: \n'
        string += str(self.coeffs) + '\n'
        return string

    def rotate(self):
        # Only rotate by 90 degrees about the y axis for diSPIM.
        # Generalize later.
        m = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],  # Careful with -1 here
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 0, -1 / 2, 0, np.sqrt(3) / 2],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, np.sqrt(3) / 2, 0, 1 / 2]])
        return SHCoeffs(np.dot(m, self.coeffs))

    def plot(self, folder=''):
        if not os.path.exists(folder):
            os.makedirs(folder)

        self.plot_dist(filename=folder + '/dist.png')
        self.plot_spectrum(filename=folder + '/spectrum.pdf')

    def plot_spectrum(self, filename='spectrum.pdf'):
        print('Plotting: ' + filename)
        f, ax = plt.subplots(1, 1, figsize=(4, 4))

        # Create image of spherical harmonic coefficients
        image = np.zeros((self.rmax, self.mmax))
        for j, c in enumerate(self.coeffs):
            l, m = util.j2lm(j)
            image[int(l / 2), self.lmax + m] = c

        # Label rows and columns
        for l in range(self.lmax + 1):
            if l == 0:
                prepend = 'l='
            else:
                prepend = ''
            if l % 2 == 0:
                ax.annotate(r'$' + prepend + str(l) + '$', xy=(1, 1), xytext=(-0.75, int(l / 2)),
                            textcoords='data', ha='right', va='center')

        ax.annotate(r'$m=$', xy=(1, 1), xytext=(-0.75, -0.75),
                    textcoords='data', ha='right', va='center')
        for m in range(2 * self.lmax + 1):
            ax.annotate('$' + str(m - self.lmax) + '$', xy=(1, 1),
                        xytext=(int(m), -0.75),
                        textcoords='data', ha='center', va='center')

        # Label each pixel
        for (y, x), value in np.ndenumerate(image):
            if value != 0:
                ax.annotate("{0:.2f}".format(value), xy=(1, 1), xytext=(x, y),
                            textcoords='data', ha='center', va='center')

        ax.imshow(image, cmap='bwr', interpolation='nearest',
                  vmin=-np.max(self.coeffs), vmax=np.max(self.coeffs))
        ax.axis('off')

        f.savefig(filename, bbox_inches='tight')

    def plot_dist(self, filename='dist.png', n_pts=2500, r=1, mag=1, show=False,
                  force_positive=True):
        from mayavi import mlab
        print('Plotting: ' + filename)

        # Flip sign if [1,1,1] direction is negative
        coeffs = self.coeffs
        if force_positive:
            test_radii = 0
            for i, c in enumerate(coeffs):
                l, m = util.j2lm(i)
                test_radii += c * util.spZnm(l, m, np.pi / 4, np.pi / 4)
            if test_radii < 0:
                coeffs = -coeffs

        # Calculate radii
        tp = util.fibonacci_sphere(n_pts)
        xyz = util.fibonacci_sphere(n_pts, xyz=True)
        radii = np.zeros(tp.shape[0])
        for i, c in enumerate(coeffs):
            l, m = util.j2lm(i)
            radii += c * util.spZnm(l, m, tp[:, 0], tp[:, 1])
        # radii = radii/np.max(np.abs(radii)) # Handle edge cases below
        radii = np.nan_to_num(radii / np.max(np.abs(radii)))

        # Split into positive and negatives
        n = r * radii.clip(max=0)
        p = r * radii.clip(min=0)

        # Triangulation
        from scipy.spatial import ConvexHull
        ch = ConvexHull(xyz)
        triangles = ch.simplices

        # Create figure
        mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 400))
        mlab.clf()

        # Plot
        mlab.triangular_mesh(p * xyz[:, 0], p * xyz[:, 1], p * xyz[:, 2], triangles, color=(1, 0, 0))
        s = mlab.triangular_mesh(n * xyz[:, 0], n * xyz[:, 1], n * xyz[:, 2], triangles, color=(0, 0, 1))
        s.scene.light_manager.light_mode = "vtk"

        # View and save
        mlab.view(azimuth=45, elevation=45, distance=5, focalpoint=None,
                  roll=None, reset_roll=True, figure=None)
        mlab.savefig(filename, magnification=mag)
        subprocess.call(['convert', filename, '-transparent', 'white', filename])
        if show:
            mlab.show()
