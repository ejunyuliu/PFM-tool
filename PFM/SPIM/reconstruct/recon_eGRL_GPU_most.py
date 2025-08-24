import numpy as np
from tqdm import tqdm
import logging

import cupy as cp

log = logging.getLogger('log')


class recon_dual:
    def __init__(self, multi, th=1.6785e-3):
        self.th = th
        self.dispim = multi

        self.Ha = multi.H0
        self.Hb = multi.H1

        self.gaunt_gpu = cp.array(multi.Gaunt * 3.5449077)
        self.s = multi.data.g.shape[0:3]

        self.calc_H()

    def calc_H(self):
        log.info('Computing H_back and H_con')

        self.Ha_back = self.Ha.conjugate()
        self.Hb_back = self.Hb.conjugate()

        Ha_con_gpu = cp.zeros((self.Ha.shape[0:3]) + (15, 15,), dtype=cp.complex64)
        Ha_gpu = cp.array(self.Ha)
        for z in tqdm(range(self.Ha.shape[2])):
            Ha_con_gpu[:, :, z, :, :] = cp.einsum('xyjp,xysp->xyjs', Ha_gpu[:, :, z, :, :].conjugate(),
                                                  Ha_gpu[:, :, z, :, :])
        self.Ha_con = cp.asnumpy(Ha_con_gpu)
        del Ha_gpu, Ha_con_gpu

        Hb_con_gpu = cp.zeros((self.Hb.shape[0:3]) + (15, 15,), dtype=cp.complex64)
        Hb_gpu = cp.array(self.Hb)
        for z in tqdm(range(self.Hb.shape[2])):
            Hb_con_gpu[:, :, z, :, :] = cp.einsum('xyjp,xysp->xyjs', Hb_gpu[:, :, z, :, :].conjugate(),
                                                  Hb_gpu[:, :, z, :, :])
        self.Hb_con = cp.asnumpy(Hb_con_gpu)
        del Hb_gpu, Hb_con_gpu

        cp._default_memory_pool.free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    def ConvFFT1(self, Vol, OTF):
        Vol_fft_gpu = cp.fft.rfftn(cp.array(Vol), axes=(0, 1, 2))
        OTF_gpu = cp.array(OTF)

        temp_gpu = cp.zeros(OTF_gpu.shape[0:3] + (OTF_gpu.shape[3],), dtype=cp.complex64)
        for z in range(temp_gpu.shape[2]):
            temp_gpu[:, :, z, :] = cp.einsum('xyp,xyjp->xyj', Vol_fft_gpu[:, :, z, :], OTF_gpu[:, :, z, :, :])

        del Vol_fft_gpu, OTF_gpu
        Vol_gpu = cp.real(cp.fft.irfftn(temp_gpu, s=self.s, axes=(0, 1, 2)))
        del temp_gpu

        return Vol_gpu

    def ConvFFT2(self, Vol_gpu, OTF):
        Vol_fft_gpu = cp.fft.rfftn(Vol_gpu, axes=(0, 1, 2))
        OTF_gpu = cp.array(OTF)

        temp_gpu = cp.zeros(OTF_gpu.shape[0:3] + (OTF_gpu.shape[3],), dtype=cp.complex64)
        for z in range(temp_gpu.shape[2]):
            temp_gpu[:, :, z, :] = cp.einsum('xyjs,xys->xyj', OTF_gpu[:, :, z, :, :], Vol_fft_gpu[:, :, z, :])

        del Vol_fft_gpu, OTF_gpu
        Vol_gpu = cp.real(cp.fft.irfftn(temp_gpu, s=self.s, axes=(0, 1, 2)))
        del temp_gpu

        return Vol_gpu

    def SHMul(self, SH0_gpu, SH1_gpu):
        outSH_gpu = cp.zeros_like(SH0_gpu)

        for z in range(SH0_gpu.shape[2]):
            mat_gpu = cp.einsum('jls,xys->xyjl', self.gaunt_gpu, SH0_gpu[:, :, z, :])
            outSH_gpu[:, :, z, :] = cp.einsum('xyjl,xyl->xyj', mat_gpu, SH1_gpu[:, :, z, :])

        del SH0_gpu, SH1_gpu

        return outSH_gpu

    def SHDiv(self, SH0_gpu, SH1_gpu):
        outSH_gpu = cp.zeros_like(SH0_gpu)

        for z in range(SH0_gpu.shape[2]):
            mat_gpu = cp.einsum('jls,xys->xyjl', self.gaunt_gpu, SH1_gpu[:, :, z, :])
            mat_inv_gpu = cp.linalg.inv(mat_gpu)
            outSH_gpu[:, :, z, :] = cp.einsum('xyjl,xyl->xyj', mat_inv_gpu, SH0_gpu[:, :, z, :])

        del SH0_gpu, SH1_gpu

        return outSH_gpu

    def SH_cut(self, SH):
        if self.th is not None:
            mask = SH[..., 0] < self.th
            SH[..., 1:15][mask] = 0
            SH[..., 0][mask] = self.th
        return SH

    def recon(self, g, iter_num=10):
        log.info('Applying eGRL recon')

        img = g
        imga, imgb = img[..., 0], img[..., 1]

        mid_a_gpu = self.ConvFFT1(imga, self.Ha_back)
        mid_b_gpu = self.ConvFFT1(imgb, self.Hb_back)

        ek_gpu = cp.zeros(g.shape[0:3] + (15,))
        ek_gpu[..., 0] = cp.array(img.mean(axis=(3, 4)))

        for iter in tqdm(range(iter_num)):
            bwd_gpu = self.ConvFFT2(ek_gpu, self.Ha_con)
            dif_gpu = self.SHDiv(mid_a_gpu, bwd_gpu)
            del bwd_gpu
            ek_gpu = self.SHMul(ek_gpu, dif_gpu)
            del dif_gpu
            ek_gpu = self.SH_cut(ek_gpu)

            bwd_gpu = self.ConvFFT2(ek_gpu, self.Hb_con)
            dif_gpu = self.SHDiv(mid_b_gpu, bwd_gpu)
            del bwd_gpu
            ek_gpu = self.SHMul(ek_gpu, dif_gpu)
            del dif_gpu
            ek_gpu = self.SH_cut(ek_gpu)

        del mid_a_gpu, mid_b_gpu
        ek = cp.asnumpy(ek_gpu)
        del ek_gpu

        cp._default_memory_pool.free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        return ek
