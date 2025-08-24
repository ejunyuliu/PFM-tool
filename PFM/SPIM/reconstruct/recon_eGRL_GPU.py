from tqdm import tqdm
import logging
import cupy as cp

log = logging.getLogger('log')


class recon_dual:
    def __init__(self, multi, th=1.6785e-3):
        self.th = th

        self.dispim = multi

        self.Ha = cp.array(multi.H0)
        self.Hb = cp.array(multi.H1)

        self.gaunt = cp.array(multi.Gaunt * 3.5449077)
        self.s = cp.array(multi.data.g.shape[0:3])

        self.calc_H()

    def calc_H(self):
        log.info('Computing H_back and H_con')

        self.Ha_back = self.Ha.conjugate()
        self.Ha_con = cp.einsum('xyzjp,xyzsp->xyzjs', self.Ha_back, self.Ha)
        del self.Ha

        self.Hb_back = self.Hb.conjugate()
        self.Hb_con = cp.einsum('xyzjp,xyzsp->xyzjs', self.Hb_back, self.Hb)
        del self.Hb

    def ConvFFT3(self, Vol, OTF, order):
        Vol_fft = cp.fft.rfftn(Vol, axes=(0, 1, 2))
        temp = []
        if order == 0:
            temp = cp.einsum('xyzj,xyzjp->xyzp', Vol_fft, OTF)
        if order == 1:
            temp = cp.einsum('xyzp,xyzjp->xyzj', Vol_fft, OTF)
        if order == 2:
            temp = cp.einsum('xyzjs,xyzs->xyzj', OTF, Vol_fft)
        Vol = cp.real(cp.fft.irfftn(temp, s=Vol.shape[0:3], axes=(0, 1, 2)))
        return Vol

    def SHMul(self, SH0, SH1):
        mat = cp.einsum('jls,xyzs->xyzjl', self.gaunt, SH0)
        outSH = cp.einsum('xyzjl,xyzl->xyzj', mat, SH1)
        return outSH

    def SHDiv(self, SH0, SH1):
        mat = cp.einsum('jls,xyzs->xyzjl', self.gaunt, SH1)
        mat_inv = cp.linalg.inv(mat)
        outSH = cp.einsum('xyzjl,xyzl->xyzj', mat_inv, SH0)
        return outSH

    def SH_cut(self, SH):
        if self.th is not None:
            mask = SH[..., 0] < self.th
            SH[..., 1:15][mask] = 0
            SH[..., 0][mask] = self.th
        return SH

    def recon(self, g, iter_num=10):
        log.info('Applying eGRL recon')

        img = cp.array(g)
        imga, imgb = img[..., 0], img[..., 1]

        ek = cp.zeros(g.shape[0:3] + (15,))
        ek[..., 0] = img.mean(axis=(3, 4))

        del img
        mid_a = self.ConvFFT3(imga, self.Ha_back, order=1)
        del imga
        mid_b = self.ConvFFT3(imgb, self.Hb_back, order=1)
        del imgb

        for iter in tqdm(range(iter_num)):
            bwd = self.ConvFFT3(ek, self.Ha_con, order=2)
            dif = self.SHDiv(mid_a, bwd)
            del bwd
            ek = self.SHMul(ek, dif)
            del dif
            ek = self.SH_cut(ek)

            bwd = self.ConvFFT3(ek, self.Hb_con, order=2)
            dif = self.SHDiv(mid_b, bwd)
            del bwd
            ek = self.SHMul(ek, dif)
            del dif
            ek = self.SH_cut(ek)

        ek_cpu = cp.asnumpy(ek)
        del ek, mid_a, mid_b

        cp._default_memory_pool.free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        return ek_cpu