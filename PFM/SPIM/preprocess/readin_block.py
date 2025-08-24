from PFM import spang,  util
from PFM.SPIM import data
from PFM.SPIM.microscope_simplified import multi as multi_sim
from PFM.SPIM.microscope import multi as multi_com
import numpy as np
import math


def read_correct(lamb, n_samp, ill_fwhm, det_nas,
                 t0A, t0B, t1A, t1B, t_1A, t_1B,
                 pic_pols_A, pic_pols_B, sel_pols_A, sel_pols_B,
                 vox_dim, roi, data_format,
                 root_path, cal_in_path,
                 data_in_path, recon_out_folder, viz_out_folder,
                 block_size, time_point='0', PSF='complete', cal_bg=0):
    padding = None

    util.mkdir(recon_out_folder)
    util.mkdir(viz_out_folder)

    # Calculate calibration
    cal_data = data.Data(ill_fwhm=ill_fwhm, ill_pols=None, det_nas=[0, 0], det_pols=None)
    cal_data.read_calibration(root_path + cal_in_path, XY_range=15)
    cal_data.g = cal_data.g - cal_bg
    pol_offset = cal_data.calibration_fit()

    cal_averages = np.mean(cal_data.g, axis=(0, 1, 2))
    cal_averages = np.stack([cal_averages[sel_pols_A, 0], cal_averages[sel_pols_B, 1]])

    ill_pols = util.pols_from_tilt(sel_pols_A, sel_pols_B, pol_offset=pol_offset, t0A=t0A, t0B=t0B, t1A=t1A, t1B=t1B,
                                   t_1A=t_1A, t_1B=t_1B)

    # Readin
    data0 = data.Data(vox_dim=vox_dim, ill_fwhm=ill_fwhm, det_nas=det_nas, ill_pols=ill_pols, det_pols=None)

    data0.read_tiff(root_path + data_in_path, name=time_point, roi=roi, format=data_format, padding=padding)

    index_A = np.where(np.isin(pic_pols_A, sel_pols_A) == True)[0]
    index_B = np.where(np.isin(pic_pols_B, sel_pols_B) == True)[0]
    data0.g = np.stack((data0.g[..., index_A, 0], data0.g[..., index_B, 1]), axis=-1)

    spang0 = spang.Spang(f=np.zeros(data0.g.shape[0:3] + (15,)), vox_dim=vox_dim, lamb=lamb, n_samp=n_samp)

    # System model
    g = np.zeros(tuple(block_size) + data0.g.shape[3:])
    data1 = data.Data(g=g, vox_dim=vox_dim, ill_fwhm=ill_fwhm, det_nas=det_nas, ill_pols=ill_pols, det_pols=None)
    spang1 = spang.Spang(f=np.zeros(data1.g.shape[0:3] + (15,)), vox_dim=vox_dim, lamb=lamb, n_samp=n_samp)
    if PSF == 'simplified':
        dispim1 = multi_sim.MultiMicroscope(spang1, data1)
    if PSF == 'complete':
        dispim1 = multi_com.MultiMicroscope(spang1, data1)

    dispim1.calc_H()

    # Calibrate
    spang1.f[..., 0] = 1
    data1.g = dispim1.fwd(spang1.f)
    model_averages = np.mean(data1.g, axis=(0, 1, 2))

    data0.g = data0.g * cal_averages[0, 0] * model_averages / cal_averages.T

    return dispim1, spang0, data0



def reconstruct(dispim1, spang0, data0, block_size, res_size, Iter, iter_num):
    # inital
    data_size = data0.g.shape[0:3]

    # create mask
    dx = np.ones((block_size[0]))
    dx[:res_size[0]] = np.linspace(0, 1, res_size[0])
    dx[-res_size[0]:] = dx[:res_size[0]][::-1]

    dy = np.ones((block_size[1]))
    dy[:res_size[1]] = np.linspace(0, 1, res_size[1])
    dy[-res_size[1]:] = dy[:res_size[1]][::-1]

    dz = np.ones((block_size[2]))
    dz[:res_size[2]] = np.linspace(0, 1, res_size[2])
    dz[-res_size[2]:] = dz[:res_size[2]][::-1]

    mx, my, mz = np.meshgrid(dx, dy, dz)
    mx, my, mz = mx.swapaxes(0, 1), my.swapaxes(0, 1), mz.swapaxes(0, 1)

    # mask = mx * my * mz
    mask = mx * my * mz

    # pad raw data
    x_num = math.ceil((data_size[0] + res_size[0]) / (block_size[0] - res_size[0]))
    y_num = math.ceil((data_size[1] + res_size[1]) / (block_size[1] - res_size[1]))
    z_num = math.ceil((data_size[2] + res_size[2]) / (block_size[2] - res_size[2]))

    x_pad = (block_size[0] - res_size[0]) * x_num - data_size[0]
    y_pad = (block_size[1] - res_size[1]) * y_num - data_size[1]
    z_pad = (block_size[2] - res_size[2]) * z_num - data_size[2]

    data0.g = np.pad(data0.g, ((res_size[0], x_pad), (res_size[1], y_pad), (res_size[2], z_pad), (0, 0), (0, 0)),
                     'reflect')
    spang0.f = np.zeros(data0.g.shape[0:3] + (15,))

    # recon process
    for i in range(x_num):
        for j in range(y_num):
            for k in range(z_num):
                print(str(i + 1) + '/' + str(x_num) + ', ' + str(j + 1) + '/' + str(y_num) + ', ' + str(
                    k + 1) + '/' + str(z_num))

                x_slice = slice(i * (block_size[0] - res_size[0]), i * (block_size[0] - res_size[0]) + block_size[0])
                y_slice = slice(j * (block_size[1] - res_size[1]), j * (block_size[1] - res_size[1]) + block_size[1])
                z_slice = slice(k * (block_size[2] - res_size[2]), k * (block_size[2] - res_size[2]) + block_size[2])
                g = data0.g[x_slice, y_slice, z_slice]

                f = Iter.recon(g, iter_num)

                f = np.einsum('xyzj,xyz->xyzj', f, mask)
                spang0.f[x_slice, y_slice, z_slice] = spang0.f[x_slice, y_slice, z_slice] + f

    data0.g = data0.g[res_size[0]:(-x_pad), res_size[1]:(-y_pad), res_size[2]: (-z_pad)]
    spang0.f = spang0.f[res_size[0]:(-x_pad), res_size[1]:(-y_pad), res_size[2]: (-z_pad)]