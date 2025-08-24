import os
import numpy as np
from PFM import util
from PFM.SPIM.microscope_simplified import ill, det
from PFM.harmonics import gaunt, shcoeffs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

class Microscope:
    """
    A Microscope represents an experiment that collects a single frame of 
    intensity data.  

    A Microscope is specified by its illumination path (an Illuminator object),
    and its detection path (a Detector object).
    """
    def __init__(self, ill=ill.Illuminator(), det=det.Detector(),
                 color=(0,1,0.3), spang_coupling=True):
        self.ill = ill
        self.det = det
        self.h0 = self.ill.h()*self.det.h(0, 0, 0)
        self.n_len = self.h0.coeffs.shape[0]
        self.j_len = self.h0.coeffs.shape[1]
        self.hnorm = (self.ill.h()*self.det.h(0, 0, 0)).coeffs[0, 0]
        self.Hnorm = (self.ill.H()*self.det.H(0, 0, 0)).coeffs[0]
        self.color = color
        self.spang_coupling = spang_coupling

    def h(self, x, y, z):
        return self.ill.h()*self.det.h(x, y, z)/self.hnorm
    
    def H(self, x, y, z, pol):
        if self.spang_coupling:
            return self.ill.H(pol)*self.det.H(x, y, z)/self.Hnorm
        else:
            return self.ill.H(pol)*self.det.H(0, 0, 0)/self.Hnorm
        
    def plot(self, func=h, filename='micro.pdf', n_px=2**6, plot_m=[-2, 0, 2],
             contours=True):
        print('Plotting: ' + filename)

        mlen = len(plot_m)
        nlen = 3
        
        # Calculate data for transvsere plotting
        w = 2.05
        [X, Y] = np.meshgrid(np.linspace(-w, w, n_px),
                             np.linspace(-w, w, n_px))
        
        data = np.zeros((n_px, n_px, self.n_len, self.j_len))
        for index, x in np.ndenumerate(X):
            if self.det.optical_axis == [0,0,1]: # z-detection
                data[index[0], index[1], :, :] = (func(x, Y[index], 0)).coeffs
            elif self.det.optical_axis == [1,0,0]: # x-detection
                data[index[0], index[1], :, :] = (func(0, x, Y[index])).coeffs

        # Layout windows
        if plot_m is None:
            mcols = self.h0.rmax
        else:
            mcols = len(plot_m)

        cols = mcols*self.n_len
        inches = 1

        f, axs = plt.subplots(self.h0.rmax, cols,
                              figsize=(inches*cols, inches*self.h0.rmax),
                              gridspec_kw={'hspace':0.0, 'wspace':0.05})
        
        for index, ax in np.ndenumerate(axs):
            l = 2*index[0]
            if plot_m is None:
                m = index[1] - int(self.h0.mmax/2)
            else:
                plot_n = [-2, 0, 2]
                n_ind = index[1]//mlen
                n = plot_n[n_ind]
                m = plot_m[index[1]%mlen]
            j = util.lm2j(l, m)
            ax.axis('off')
            
            if index[0] == 0:
                ax.annotate('$m='+str(m)+'$', xy=(1, 1), xytext=(0.5, 1.15),
                            textcoords='axes fraction', ha='center', va='center')
                if index[1] % mlen == np.floor(mlen/2):
                    ax.annotate('$n='+str(n)+'$', xy=(1, 1), xytext=(0.5, 1.4),
                                textcoords='axes fraction', ha='center', va='center')
            if index[1] % mlen == mlen - 1 and index[1] != axs.shape[1] - 1:
                ax.annotate("",
                            xy=(1, -0.05), xycoords='axes fraction',
                            xytext=(1, 1.3), textcoords='axes fraction',
                            arrowprops=dict(arrowstyle="-", 
                                            linewidth=0.5))
                
            if index[1] == 0:
                ax.annotate('$l='+str(l)+'$', xy=(1, 1), xytext=(-0.15, 0.5),
                            textcoords='axes fraction', ha='center', va='center',
                            rotation=90)
            
            if j is not None:
                ax.imshow(data[:,:,n,j], cmap="bwr", vmin=-1, vmax=1, interpolation='none')
                levels = [-0.1, -1e-5, 1e-5, 0.1]
                if contours:
                    ct = ax.contour(data[:,:,n,j], levels, colors='k',linewidths=0.5)
            else:
                ax.imshow(np.zeros(data[:,:,0,0].shape), cmap="bwr", vmin=-1, vmax=1, interpolation='none')
                
        f.savefig(filename, bbox_inches='tight')

    def calc_SVD(self, n_px=2**6):
        w = 2.0
        self.xcoords = np.linspace(-w, w, n_px),
        [X, Y] = np.meshgrid(self.xcoords, self.xcoords)
        R = np.sqrt(X**2 + Y**2)
        Phi = np.nan_to_num(np.arctan(Y/X))

        # For each position calculate K and solve eigenequation
        sigma = np.zeros((n_px, n_px, self.n_len))
        for index, x in np.ndenumerate(X):
            if self.det.optical_axis == [0,0,1]: # z-detection
                u, s, v = self.calc_point_SVD(x, Y[index], 0)
            elif self.det.optical_axis == [1,0,0]: # x-detection
                u, s, v = self.calc_point_SVD(0, x, Y[index])
            sigma[index[0], index[1], :] = s

        self.sigma = sigma
        self.sigma_max = np.max(sigma)

    def calc_point_SVD(self, x, y, z):
        HH = self.H(x, y, z).coeffs
        K = np.dot(HH, HH.T)
        mu, v = np.linalg.eigh(K)
        u = np.dot(HH.T, v)

        return u[:,::-1], np.sqrt(mu[::-1]), v[:,::-1] # Transpose

    def plot_SVS(self, filename='svs.pdf', n_px=2**6, marks=[[1e-5,0], [0.5,0], [1.0,0], [1.5,0]]):
        print('Plotting: ' + filename)

        # Layout windows
        inches = 1.5
        rows = self.sigma.shape[-1] + 1
        cols = len(marks) + 1
        f, axs = plt.subplots(rows, cols,
                              figsize=(inches*cols, inches*(rows - 0.75)),
                              gridspec_kw={'hspace':0.05, 'wspace':0.10, 'height_ratios':[1,1,1,0.05]})

        # Label top row with arrow
        axs[0,0].annotate('', xy=(-0.03, 1.1), xytext=(0.55, 1.1),
                          textcoords='axes fraction', xycoords='axes fraction', ha='center', va='center',
                          arrowprops=dict(arrowstyle='<->, head_width=0.05, head_length=0.1',
                                          connectionstyle="arc3", linewidth=0.5),)
        axs[0,0].annotate(r'$2\textrm{NA}/\lambda$', xy=(0,0), xytext=(0.25, 1.2),
                          textcoords='axes fraction', xycoords='axes fraction', ha='center', va='center', fontsize=7)

        # Top row singular values        
        for j, ax in enumerate(axs[:-1,0]):
            ax.axis('off')
            ax.annotate('$j='+str(j)+'$', xy=(1, 1), xytext=(-0.15, 0.5),
                        textcoords='axes fraction', ha='center', va='center', rotation=90)
            extent= [-2,2,-2,2]
            origin = 'lower'
            ax.imshow(self.sigma[:,:,j]/self.sigma_max, cmap="bwr", vmin=-1, vmax=1, interpolation='none', extent=extent, origin=origin)
            ax.set_xlim([-2.05,2.05])
            ax.set_ylim([-2.05,2.05])            
            levels = [-0.1, -1e-5, 1e-5, 0.1]
            ct = ax.contour(self.sigma[:,:,j]/self.sigma_max, levels, colors='k',linewidths=0.5, extent=extent, origin=origin)
            for mark in marks:
                ax.plot(mark[0]*np.cos(mark[1]), mark[0]*np.sin(mark[1]), 'xk', ms=2.5, mew=0.5)

        # Colorbars
        X, Y = np.meshgrid(np.linspace(0, 1, 100),
                             np.linspace(0, 1, 100))
        axs[-1,0].imshow(X, cmap="bwr", vmin=-1, vmax=1, interpolation='none', extent=[0,1,0,1], origin='lower', aspect='auto')
        axs[-1,0].contour(X, levels, colors='k',linewidths=0.5, extent=[0,1,0,1], origin='lower',)
        axs[-1,0].set_xlim([0,1])
        axs[-1,0].set_ylim([0,1])
        axs[-1,0].tick_params(direction='out', bottom=True, top=False)
        axs[-1,0].xaxis.set_ticks([0, 0.5, 1.0])
        axs[-1,0].yaxis.set_ticks([])
        for j, ax in enumerate(axs[-1, 1:]):
            ax.axis('off')

        # For saving singular function pngs
        folder = 'singular'
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Object space singular functions
        for j, ax in np.ndenumerate(axs[:-1,1:]):
            if self.det.optical_axis == [0,0,1]: # z-detection
                u, s, v = self.calc_point_SVD(marks[j[1]][0], marks[j[1]][1], 0)
            elif self.det.optical_axis == [1,0,0]: # x-detection
                u, s, v = self.calc_point_SVD(0, marks[j[1]][0], marks[j[1]][1])

            # Labels
            if j[0] == 0:
                rho_label = str(marks[j[1]][0])
                if marks[j[1]][0] < 1e-3:
                    rho_label = '0'
                ax.annotate(r'$\rho = '+rho_label+'$', xy=(1, 1), xytext=(0.5, 1.15),
                            textcoords='axes fraction', ha='center', va='center')
                ax.annotate(r'$\phi_{\rho} = '+str(marks[j[1]][1])+'$', xy=(1, 1), xytext=(0.5, 0.95),
                            textcoords='axes fraction', ha='center', va='center')
            ax.axis('off')
            ax.annotate('$\sigma='+'{:.2f}'.format(s[j[0]]/self.sigma_max)+'$', xy=(1, 1), xytext=(0.5, 0),
                        textcoords='axes fraction', ha='center', va='center')

            # Create singular function plots
            sh = shcoeffs.SHCoeffs(u[:,j[0]])
            shfilename = folder+'/'+str(j[0])+str(j[1])+'.png'
            sh.plot_dist(filename=folder+'/'+str(j[0])+str(j[1])+'.png', r=1.1)

            from PIL import Image
            im1 = np.asarray(Image.open(shfilename))
            ax.imshow(im1, interpolation='none')

        
        f.savefig(filename, bbox_inches='tight')

    def scene_string(self):
        asy_string = ''
        ill_string = "circle(optical_axis, alpha, false, color);\n"
        ill_string = ill_string.replace('optical_axis', str(tuple(self.det.optical_axis)))
        ill_string = ill_string.replace('alpha', str(np.arcsin(self.ill.na/self.ill.n)))
        ill_string = ill_string.replace('color', str(self.color))
        asy_string += ill_string

        if self.ill.polarizer is not None:
            pol_string = "arrow(optical_axis, polarizer, color, false);\n"
            pol_string = pol_string.replace('optical_axis', str(tuple(self.ill.optical_axis)))
            pol_string = pol_string.replace('polarizer', str(tuple(self.ill.polarizer)))
            pol_string = pol_string.replace('color', str(self.color))
            asy_string += pol_string

        if self.det.polarizer is not None:
            pol_string = "arrow(optical_axis, polarizer, color, true);\n"
            pol_string = pol_string.replace('optical_axis', str(tuple(self.det.optical_axis)))
            pol_string = pol_string.replace('polarizer', str(tuple(self.det.polarizer)))
            pol_string = pol_string.replace('color', str(self.color))
            asy_string += pol_string
        
        det_string = "circle(optical_axis, alpha, true, color);\n"
        det_string = det_string.replace('optical_axis', str(tuple(self.det.optical_axis)))
        det_string = det_string.replace('alpha', str(np.arcsin(self.det.na/self.det.n)))
        det_string = det_string.replace('color', str(self.color))
        asy_string += det_string
        
        return asy_string

    def plot_scene(self, filename):
        print('Plotting: ' + filename)        
        util.draw_scene(self.scene_string(), filename=filename, save_file=True)
