# A package for fluorescence spatio-angular imaging under light sheet microscopy: simulation, reconstruction and rendering (PFM-tool)

This is the companion code to our paper:

[Observing biological spatio-angular structures and dynamics with statistical image reconstruction and polarized fluorescence microscopy]().

![Example](figures/example.png)

PFM-tool is a 3D spatio-angular reconstruction pipeline for polarized fluorescence light sheet microscopy, incoperating
efficient generalized Richarson-Lucy algorithm (eGRL) to restore density and orientation distribution in each voxel.

## System Requirements

Tested Environment:

eGRL pipeline for reconstructing GUV dataset based on GPU:

    - Windows 10 PRO 22H2
    - Python 3.9
    - NVIDIA GeForce RTX 4090 24GB
    - CUDA 12.8 and cuDNN 7.6.5

## Installation

Create an anaconda environment `PFM-tool`. This will take ~5 minutes.

    cd PFM-tool
    conda env create -f environment-windows.yml

Activate the `PFM-tool` environment. You will need to activate this environment
every time you want to run polaris.

    conda activate PFM-tool

Install `PFM-tool` locally so that you can access it from anywhere.

    pip install -e ./


## Dataset

We have uploaded the following datasets for testing, please place the downloaded folder `test_data` into directory `./examples/`:

| Dataset                    | Probe                   | Expected orientation | Testing script              |
|:---------------------------|:------------------------|:---------------------|:----------------------------|
| Giant Unilamellar Vesicle  | FM1-43                  | Normal to membrane   | ./examples/recon_GUV_GPU.py |
| Tobacco xylem cell         | Pontamine fast scarlet  | Along the ribs       | ./examples/recon_Xylem.py   |

## Acknowledgement

The rendering codes are cloned and modified from [polaris](https://github.com/talonchandler/polaris).