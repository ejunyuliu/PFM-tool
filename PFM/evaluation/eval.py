import numpy as np
from PFM import util, viz, spang
from skimage.metrics import structural_similarity as ssim
from dipy.data import get_sphere
import matplotlib.pyplot as plt


class Eval_FWHM:
    def __init__(self, point_list, dis, vox_dim=0.13, padding=None):
        self.vox_dim = vox_dim
        self.point_list = point_list
        self.dis = dis
        self.padding = padding

    def calc(self, spangf, axes=['x']):
        if self.padding is not None:
            spangf = spangf[self.padding:-self.padding, self.padding:-self.padding, self.padding:-self.padding, ...]

        den = spangf[..., 0]

        fwhm_list = []

        for axis in axes:
            fwhm_arry = np.zeros(len(self.point_list))

            for idx, point in enumerate(self.point_list):
                if axis == 'x':
                    pro = den[(point[0] - self.dis[0] // 2):(point[0] + self.dis[0] // 2), point[1], point[2]]
                if axis == 'y':
                    pro = den[point[0], (point[1] - self.dis[1] // 2):(point[1] + self.dis[1] // 2), point[2]]
                if axis == 'z':
                    pro = den[point[0], point[1], (point[2] - self.dis[2] // 2):(point[2] + self.dis[2] // 2)]

                if pro.max() != pro.min():
                    pro = (pro - pro.min()) / (pro.max() - pro.min())
                else:
                    pro = pro * 0 + 1

                high = np.where(pro > 0.5)[0]
                s, e = high[0], high[-1]
                if s > 0:
                    s = s - (pro[s] - 0.5) / (pro[s] - pro[s - 1])
                if e < len(pro) - 1:
                    e = e + (pro[e] - 0.5) / (pro[e] - pro[e + 1])
                fwhm = e - s
                fwhm_arry[idx] = fwhm

            fwhm_list.append(fwhm_arry * self.vox_dim)

        return fwhm_list[0]


class Eval_ColorHistogram:
    def __init__(self, sphere=get_sphere('symmetric362')):
        self.sphere = sphere

        self.J = 15
        self.N = len(self.sphere.theta)

        self.calc_B()

    def orient2rgb(self, v):
        M = np.array([[1 / np.sqrt(2), 0, 1 / np.sqrt(2)], [0, 1, 0], [-1 / np.sqrt(2), 0, 1 / np.sqrt(2)]])
        vv = np.einsum('ab,nb->na', M, v)
        vv = np.einsum('na,n->na', vv, 1 / np.linalg.norm(vv, axis=1))
        return np.abs(vv)

    def calc_B(self):
        # Calculate odf to sh matrix
        B = np.zeros((self.N, self.J))
        for (n, j), x in np.ndenumerate(B):
            l, m = util.j2lm(j)
            B[n, j] = util.spZnm(l, m, self.sphere.theta[n], self.sphere.phi[n])
        self.B = B
        self.Binv = np.linalg.pinv(self.B, rcond=1e-15)
        self.color = self.orient2rgb(self.sphere.vertices)

    def calc(self, spangf):
        rcd = np.zeros(self.N)

        th = 0.1
        mask = (spangf[..., 0] / np.max(spangf[..., 0])) > th

        xyz = np.ascontiguousarray(np.array(np.nonzero(mask)).T)
        masked_sh = spangf[mask]
        masked_radii = np.einsum('vj,pj->vp', self.Binv.T, masked_sh)  # Radii
        index = np.argmax(masked_radii, axis=0)

        val = np.abs(self.sphere.vertices[index, :]).sum(axis=0)
        value = np.array([val[0], val[1], val[2]])
        value = value / value.sum()

        # self.draw_radar(value)

        return value

    def draw_radar(self, value):
        angles = np.array([-30, 90, 210, -30]) / 180 * np.pi

        labels = ['x', 'y', 'z', 'x']

        ax = plt.subplot(111, polar=True)
        ax.plot(angles, value, color='b')
        ax.fill(angles, value, 'm', alpha=0.75)  # 填充
        plt.show()


class Eval_NCC:
    def __init__(self, phant):
        self.phant = phant

    def calc(self, spangf):
        image, label = spangf[..., 0], self.phant.f[..., 0]

        # image = (image - image.min()) / (image.max() - image.min())
        # label = (label - label.min()) / (label.max() - label.min())

        mean_image = np.mean(image)
        mean_label = np.mean(label)

        std_image = np.std(image)
        std_label = np.std(label)

        final_NCC = np.mean((image - mean_image) * (label - mean_label) / (std_label * std_image))

        return np.nan_to_num(final_NCC)


class Eval_MSE:
    def __init__(self, phant, padding=None):
        self.phant = phant
        self.padding = padding

    def calc(self, spangf):
        if self.padding is not None:
            spangf = spangf[self.padding:-self.padding, self.padding:-self.padding, self.padding:-self.padding, ...]

        image, label = spangf[..., 0], self.phant.f[..., 0]

        # image = image / image.max()
        # label = label / label.max()

        image = (image - image.min()) / (image.max() - image.min())
        label = (label - label.min()) / (label.max() - label.min())

        final_mse = np.mean((image - label) ** 2)
        return np.nan_to_num(final_mse)


class Eval_SSIM:
    def __init__(self, phant, padding=None):
        self.phant = phant
        self.padding = padding

    def calc(self, spangf):
        if self.padding is not None:
            spangf = spangf[self.padding:-self.padding, self.padding:-self.padding, self.padding:-self.padding, ...]

        image, label = spangf[..., 0], self.phant.f[..., 0]

        # image = image / image.max()
        # label = label / label.max()

        image[image < 0] = 0

        image = (image - image.min()) / (image.max() - image.min())
        label = (label - label.min()) / (label.max() - label.min())

        image = image.reshape(image.shape[0], image.shape[1], -1)
        label = label.reshape(label.shape[0], label.shape[1], -1)
        label = label.astype(image.dtype)

        final_ssim = ssim(image, label)
        return np.nan_to_num(final_ssim)

    # def calc(self, spangf):
    #     if self.padding is not None:
    #         spangf = spangf[self.padding:-self.padding, self.padding:-self.padding, self.padding:-self.padding, ...]
    #
    #     image, label = spangf[..., 0], self.phant.f[..., 0]
    #
    #     image = (image - image.min()) / (image.max() - image.min())
    #     label = (label - label.min()) / (label.max() - label.min())
    #
    #     mean_image = np.mean(image)
    #     mean_label = np.mean(label)
    #
    #     std_image = np.std(image)
    #     std_label = np.std(label)
    #
    #     c1 = 1e-4
    #     c2 = 9e-4
    #
    #     S1 = 2 * mean_label * mean_image + c1
    #     S2 = np.mean((image - mean_image) * (label - mean_label)) + c2
    #     S3 = mean_label ** 2 + mean_image ** 2 + c1
    #     S4 = std_label ** 2 + std_image ** 2 + c2
    #
    #     final_ssim = S1 * S2 / S3 / S4
    #     return np.nan_to_num(final_ssim)


class Eval_PSIM:
    def __init__(self, phant, threshold=0.2, padding=None):
        self.phant = phant
        self.threshold = threshold
        self.padding = padding

    def calc(self, spangf):
        if self.padding is not None:
            spangf = spangf[self.padding:-self.padding, self.padding:-self.padding, self.padding:-self.padding, ...]

        spangf, labelf, BinvT, Bvertices = spangf, self.phant.f, self.phant.Binv.T, self.phant.sphere.vertices

        mask = labelf[..., 0] > (self.threshold * np.max(labelf[..., 0]))

        spang_sh = spangf[mask]
        label_sh = labelf[mask]

        spang_odf = np.einsum('vj,pj->vp', BinvT, spang_sh)  # Radii
        spang_index = np.argmax(spang_odf, axis=0)
        spang_dirs = Bvertices[spang_index]

        label_odf = np.einsum('vj,pj->vp', BinvT, label_sh)  # Radii
        label_index = np.argmax(label_odf, axis=0)
        label_dirs = Bvertices[label_index]

        peak_cos = (spang_dirs * label_dirs).sum(axis=1)

        peak_dif = np.abs(peak_cos).mean()

        return np.nan_to_num(peak_dif)


class Eval_ONCC:
    def __init__(self, phant, threshold=0.2, padding=None):
        self.phant = phant
        self.threshold = threshold
        self.padding = padding

    def calc(self, spangf):
        if self.padding is not None:
            spangf = spangf[self.padding:-self.padding, self.padding:-self.padding, self.padding:-self.padding, ...]

        spangf, labelf, BinvT, Bvertices = spangf, self.phant.f, self.phant.Binv.T, self.phant.sphere.vertices

        mask = labelf[..., 0] > (self.threshold * np.max(labelf[..., 0]))

        spang_sh = spangf[mask]
        label_sh = labelf[mask]

        spang_odf = np.einsum('vj,nj->nv', BinvT, spang_sh)  # Radii
        spang_odf = spang_odf / spang_odf.max(axis=1, keepdims=True)

        label_odf = np.einsum('vj,nj->nv', BinvT, label_sh)  # Radii
        label_odf = label_odf / label_odf.max(axis=1, keepdims=True)

        mean_image = np.mean(spang_odf, axis=1, keepdims=True)
        mean_label = np.mean(label_odf, axis=1, keepdims=True)

        std_image = np.std(spang_odf, axis=1, keepdims=True)
        std_label = np.std(label_odf, axis=1, keepdims=True)

        oncc = np.mean((spang_odf - mean_image) * (label_odf - mean_label) / (std_label * std_image))

        return np.nan_to_num(oncc)


class Eval_OSIM:
    def __init__(self, phant, threshold=0.2, padding=None):
        self.phant = phant
        self.threshold = threshold
        self.padding = padding

    def calc(self, spangf):
        if self.padding is not None:
            spangf = spangf[self.padding:-self.padding, self.padding:-self.padding, self.padding:-self.padding, ...]

        spangf, labelf, BinvT, Bvertices = spangf, self.phant.f, self.phant.Binv.T, self.phant.sphere.vertices

        mask = labelf[..., 0] > (self.threshold * np.max(labelf[..., 0]))

        spang_sh = spangf[mask]
        label_sh = labelf[mask]

        # density = labelf[mask, 0]
        # density = density / density.max()

        spang_sh = np.einsum('pj,p->pj', spang_sh, spang_sh[:, 0])
        label_sh = np.einsum('pj,p->pj', label_sh, label_sh[:, 0])

        spang_odf = np.einsum('vj,nj->nv', BinvT, spang_sh)  # Radii
        spang_odf = spang_odf / spang_odf.max(axis=1, keepdims=True)

        label_odf = np.einsum('vj,nj->nv', BinvT, label_sh)  # Radii
        label_odf = label_odf / label_odf.max(axis=1, keepdims=True)

        # odf_dif = 1 - (np.sqrt(((label_odf - spang_odf) ** 2).mean(axis=1)) * density).sum() / density.sum()
        odf_dif = 1 - (np.sqrt(((label_odf - spang_odf) ** 2).mean(axis=1))).mean()

        # spang_odf = np.einsum('vj,nj->nv', BinvT, spang_sh)  # Radii
        # spang_odf = spang_odf / spang_odf.sum(axis=1, keepdims=True) * BinvT.shape[0]
        #
        # label_odf = np.einsum('vj,nj->nv', BinvT, label_sh)  # Radii
        # label_odf = label_odf / label_odf.sum(axis=1, keepdims=True) * BinvT.shape[0]
        #
        # odf_dif = 1 - (np.sqrt(((label_odf - spang_odf) ** 2).mean()))

        return np.nan_to_num(odf_dif)


def OpDif(spangf, labelf, threshold=0):
    def tp2sft(sft, theta, phi, n):
        t = theta[n]
        p = phi[n]

        sft[n, 0] = util.spZnm(0, 0, t, p)
        sft[n, 1] = util.spZnm(2, -2, t, p)
        sft[n, 2] = util.spZnm(2, -1, t, p)
        sft[n, 3] = util.spZnm(2, 0, t, p)
        sft[n, 4] = util.spZnm(2, 1, t, p)
        sft[n, 5] = util.spZnm(2, 2, t, p)

    from dipy.data import get_sphere
    sphere = get_sphere('symmetric362')  # repulsion100 symmetric362
    N = len(sphere.theta)

    mask = np.where(
        (spangf[..., 0] > threshold * spangf[..., 0].max()) & (labelf[..., 0] > threshold * labelf[..., 0].max()))
    spangf = spangf[mask]
    labelf = labelf[mask]
    spangf = np.einsum('nm,n->nm', spangf[..., 1:6], 1 / spangf[..., 0])
    labelf = np.einsum('nm,n->nm', labelf[..., 1:6], 1 / labelf[..., 0])

    from joblib import Parallel, delayed
    sft = np.zeros((N, 6))
    Parallel(n_jobs=-1, backend='threading')([delayed(tp2sft)(sft, sphere.theta, sphere.phi, n) for n in range(N)])
    sft = sft[:, 1:6]

    spang_op = np.sqrt(4 * np.pi / 5) * np.einsum('l,sl->s', spangf.mean(axis=0), sft)
    label_op = np.sqrt(4 * np.pi / 5) * np.einsum('l,sl->s', labelf.mean(axis=0), sft)
    # return spang_op - label_op, label_op, sphere.vertices
    return spang_op, label_op, sphere.vertices

    # spang_op = np.sqrt(4 * np.pi / 5) * np.einsum('ml,sl->ms', spangf, sft)
    # label_op = np.sqrt(4 * np.pi / 5) * np.einsum('ml,sl->ms', labelf, sft)
    # return np.abs(spang_op - label_op).mean(axis=0), np.abs(label_op), sphere.vertices
