from PFM.SPIM.preprocess import readin_block
import numpy as np
from PFM.SPIM.reconstruct import recon_eGRL_CPU, recon_eGRL_GPU_most, recon_eGRL_GPU
from os.path import normpath, basename
import subprocess


def get_gpu_memory():
    """
    Get the free memory (in GB) of the current GPU (GPU 0).

    Returns:
        float: Free memory in GB, or None if no GPU is available or detection fails.
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader', '-i', '0'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8'
        )
        if result.returncode != 0 or "failed" in result.stderr.lower():
            print("No available NVIDIA GPU or detection failed")
            return None

        free_mem_mb = int(result.stdout.strip())
        free_mem_gb = free_mem_mb / 1024  # Convert MB to GB
        return round(free_mem_gb, 2)

    except Exception as e:
        print("Error detecting GPU free memory:", e)
        return 0


if __name__ == '__main__':
    lamb = 561  # Laser wavelength in nm
    n_samp = 1.33  # Refractive index of the sample
    ill_fwhm = [2000, 2000]  # Illumination beam full width at half maximum (FWHM), in nm
    det_nas = [1.1, 0.67]  # Detection numerical apertures for two views (View A and View B)

    # Light sheet tilt angle
    t0A = 0.2
    t0B = 0.2
    t1A = 9.2
    t1B = 4.8
    t_1A = -5.3
    t_1B = -6.5

    # List of polarization angles (by index) captured for each view
    pic_pols_A = np.arange(21)
    pic_pols_B = np.arange(21)

    # Selected polarization indices to use for reconstruction
    sel_pols_A = [0, 2, 4, 7, 9, 11, 14, 16, 18]
    sel_pols_B = [0, 2, 4, 7, 9, 11, 14, 16, 18]

    vox_dim = [130, 130, 130]  # Voxel dimensions, in nm
    data_format = 'diSPIM-2Vx3Tx7P'  # Data format indicating the file structure, i.e., 2 views × 3 tilts × 7 polarizations

    root_path = './test_data/20200322_XylemCell_FullCalibration/'  # Dataset root folder
    cal_in_path = '20200322_FullCalibration/LS_561/'  # Relative path of calibration data
    data_in_path = '20200322_XylemCell/Sample2_42V/Reg/'  # Relative path of sample raw data

    # Output path for reconstructions and rendering
    recon_out_folder = basename(normpath(root_path)) + '/' + data_in_path
    viz_out_folder = recon_out_folder

    # Region of interest (full volume by default)
    roi = None

    block_size = [100, 100, 100]  # Block size for chunked processing
    res_size = [10, 10, 10]  # Redundant margin added around each block

    iter_num = 10  # Number of iterations for the reconstruction algorithm

    # Step 1: Load and preprocess raw data
    dispim1, spang0, data0 = readin_block.read_correct(lamb, n_samp, ill_fwhm, det_nas,
                                                       t0A, t0B, t1A, t1B, t_1A, t_1B,
                                                       pic_pols_A, pic_pols_B, sel_pols_A, sel_pols_B,
                                                       vox_dim, roi, data_format,
                                                       root_path, cal_in_path,
                                                       data_in_path, recon_out_folder, viz_out_folder,
                                                       block_size)

    # Step 2: Run dual-view eGRL reconstruction on CPU or GPU
    Iter = []
    gpu_mem = get_gpu_memory()
    if gpu_mem <= 4:
        Iter = recon_eGRL_CPU.recon_dual(dispim1)
    if (gpu_mem > 4) & (gpu_mem < 12):
        Iter = recon_eGRL_GPU_most.recon_dual(dispim1)
    if gpu_mem >= 12:
        Iter = recon_eGRL_GPU.recon_dual(dispim1)

    readin_block.reconstruct(dispim1, spang0, data0, block_size, res_size, Iter, iter_num)

    # Step 4: Save the reconstruction result as TIFF
    filename = 'eGRL_' + str(len(sel_pols_A) * 2) + 'V_' + str(iter_num) + 'it.tif'
    spang0.save_tiff(filename=recon_out_folder + filename)

    # Step 5: Visualize the peak orientation map
    spang0.visualize(out_path=viz_out_folder + 'rendering/', viz_type=['Peak'], mask=spang0.density() > 0.2,
                     interact=True, titles=False, scalebar=False, video=False, n_frames=18, scale=3, skip_n=5,
                     peak_scale=3)
