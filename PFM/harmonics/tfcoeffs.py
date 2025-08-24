import numpy as np
import subprocess
from PFM import util
from PFM.harmonics import gaunt
import PFM.harmonics.shcoeffs as sh
import matplotlib.pyplot as plt
from PIL import Image
import os

P = np.load(os.path.join(os.path.dirname(__file__), 'chcoeff_n2.npy'))
G = np.load(os.path.join(os.path.dirname(__file__), 'gaunt_l4.npy')) 

class TFCoeffs:
    """A TFCoeffs object stores the transfer function coefficients for a 
    microscope's point response function. 

    The coefficients are stored in 2D array of spherical and circular harmonic
    coefficients. 

    The first index of self.coeffs corresponds to the cirular harmonics and the
    second index corresponds to the spherical harmonics. We use the following
    "lexicographic ordering" of the harmonics:

    z_0  y_0^0, z_0  y_2^-2, z_0  y_2^0, ...
    z_-2 y_0^0, z_-2 y_2^-2, z_-2 y_2^0,
    z_2  y_0^0, z_2  y_2^-2, z_2  y_2^0,
    .
    .
    .
    """

    def __init__(self, coeffs):
        self.nlen = len(coeffs)
        self.lmax, mm = util.j2lm(len(coeffs[0]) - 1)
        self.jmax = int(0.5*(self.lmax + 1)*(self.lmax + 2))
        self.mmax = 2*self.lmax + 1
        self.rmax = int(self.lmax/2) + 1

        # Fill the rest of the last n band with zeros
        temp = np.zeros((self.nlen, self.jmax))
        temp[:len(coeffs), :len(coeffs[0])] = coeffs
        self.coeffs = temp

    def __add__(self, other):
        return TFCoeffs(self.coeffs + other.coeffs)
        
    def __mul__(self, other):
        if not isinstance(other, TFCoeffs):
            return TFCoeffs(self.coeffs*other)

        # Pad inputs (TODO: not general)
        npad = ((0,0), (0, 2*(self.lmax + 2) + 1))
        x1 = np.pad(np.array(self.coeffs), npad, 'constant')
        x2 = np.pad(np.array(other.coeffs), npad, 'constant')

        # Reshape (TODO: not general)
        Ploc = P[:3, :3, :3]
        
        # Multiply
        result = np.einsum('abc,def,ad,be->cf', Ploc, G, x1, x2)
        return TFCoeffs(result)

    def __truediv__(self, scalar):
        return TFCoeffs(self.coeffs/scalar)

    def __repr__(self):
        string = 'TFCoeffs: \n'
        return string + str(self.coeffs)

    # def plot(self, folder=''):
    #     if not os.path.exists(folder):
    #         os.makedirs(folder)
        
    #     self.plot_dist(filename=folder+'/dist.png')
    #     self.plot_spectrum(filename=folder+'/spectrum.pdf')

    # def plot_spectrum(self, filename='spectrum.pdf'):
    #     print('Plotting: ' + filename)
    #     f, ax = plt.subplots(1, 1, figsize=(4, 4))

    #     # Create image of spherical harmonic coefficients
    #     image = np.zeros((self.rmax, self.mmax))
    #     for j, c in enumerate(self.coeffs):
    #         l, m = util.j2lm(j)
    #         image[int(l/2), self.lmax + m] = c

    #     # Label rows and columns
    #     for l in range(self.lmax + 1):
    #         if l == 0:
    #             prepend = 'l='
    #         else:
    #             prepend = ''
    #         if l%2 == 0:
    #             ax.annotate(r'$'+prepend+str(l)+'$', xy=(1, 1), xytext=(-0.75, int(l/2)),
    #                          textcoords='data', ha='right', va='center')
                
    #     ax.annotate(r'$m=$', xy=(1, 1), xytext=(-0.75, -0.75),
    #                  textcoords='data', ha='right', va='center')
    #     for m in range(2*self.lmax + 1):
    #         ax.annotate('$'+str(m - self.lmax)+'$', xy=(1, 1),
    #                      xytext=(int(m), -0.75),
    #                      textcoords='data', ha='center', va='center')

    #     # Label each pixel
    #     for (y,x), value in np.ndenumerate(image):
    #         if value != 0:
    #             ax.annotate("{0:.2f}".format(value), xy=(1, 1), xytext=(x, y),
    #                      textcoords='data', ha='center', va='center')
            
    #     ax.imshow(image, cmap='bwr', interpolation='nearest',
    #               vmin=-np.max(self.coeffs), vmax=np.max(self.coeffs))
    #     ax.axis('off')

    #     f.savefig(filename, bbox_inches='tight')

    # def plot_dist(self, filename='dist.png', n_pts=2500, r=1, mag=1, show=False):
    #     from mayavi import mlab
    #     print('Plotting: ' + filename)
        
    #     # Calculate radii
    #     tp = util.fibonacci_sphere(n_pts)
    #     xyz = util.fibonacci_sphere(n_pts, xyz=True)
    #     radii = np.zeros(tp.shape[0])
    #     for i, c in enumerate(self.coeffs):
    #         l, m = util.j2lm(i)
    #         radii += c*util.spZnm(l, m, tp[:,0], tp[:,1])
    #     radii = radii/np.max(radii)
        
    #     # Split into positive and negatives
    #     n = radii.clip(max=0) 
    #     p = radii.clip(min=0)*(-1)

    #     # Triangulation
    #     from scipy.spatial import ConvexHull
    #     ch = ConvexHull(xyz)
    #     triangles = ch.simplices

    #     # Create figure
    #     mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 400))
    #     mlab.clf()
        
    #     # Plot
    #     mlab.triangular_mesh(p*xyz[:,0], p*xyz[:,1], p*xyz[:,2], triangles, color=(1, 0, 0))
    #     s = mlab.triangular_mesh(n*xyz[:,0], n*xyz[:,1], n*xyz[:,2], triangles, color=(0, 0, 1))
    #     s.scene.light_manager.light_mode = "vtk"
        
    #     # View and save
    #     mlab.view(azimuth=45, elevation=45, distance=5, focalpoint=None,
    #               roll=None, reset_roll=True, figure=None)
    #     mlab.savefig(filename, magnification=mag)
    #     subprocess.call(['convert', filename, '-transparent', 'white', filename])
    #     if show:
    #         mlab.show()
