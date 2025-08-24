# PFM-tool

## Package Overview

PFM is a Python software package for polarized fluorescence microscopy imaging simulation, reconstruction, and analysis.
This project implements spatio-angular imaging models, supporting simulation and reconstruction for single/multi-view
polarized microscopy.

## Project Structure

```
PFM/
├── spang.py                 # Spang class: Core data structure for spatio-angular density
├── util.py                  # Utility functions
├── visualization.py         # 3D visualization module
├── viz.py                   # Visualization control functions
├── SPIM/                    # Selective Plane Illumination Microscopy module
│   ├── data.py             # Data class: Multi-view polarized microscopy data structure
│   ├── microscope/         # Complete microscope model
│   │   ├── micro.py        # Microscope class: Main microscope control
│   │   ├── ill.py          # Illuminator class: Illumination system model
│   │   ├── det.py          # Detector class: Detection system model
│   │   └── multi.py        # Multi-view microscope system
│   ├── microscope_simplified/ # Simplified microscope model
│   │   ├── micro.py        # Simplified microscope control
│   │   ├── ill.py          # Simplified illumination model
│   │   ├── det.py          # Simplified detection model  
│   │   └── multi.py        # Simplified multi-view system
│   ├── preprocess/         # Data preprocessing
│   │   └── readin_block.py # Data reading and block processing
│   └── reconstruct/        # Image reconstruction algorithms
│       ├── recon_eGRL_CPU.py  # CPU version of eGRL reconstruction
│       ├── recon_eGRL_GPU.py  # GPU version of eGRL reconstruction
│       └── recon_eGRL_GPU_most.py # compremised GPU version
└── evaluation/             # Evaluation and testing module
    └── eval.py            # Evaluation metrics (FWHM, etc.)

```

## Core Components Details

### 1. Spang Class (spang.py)

**Function**: Core data structure representing spatio-angular distribution function f(r, s)

- Stores 4D array: [x, y, z, j] - spatial coordinates and spherical harmonic coefficients
- Provides visualization, and orientation analysis
- Supports TIFF format data storage and reading

**Key Methods**:

- `density()`: Extract density data
- `visualize()`: 3D visualization
- `save_tiff()`/`read_tiff()`: Data storage

### 2. SPIM Module

#### 2.1 Data Class (SPIM/data.py)

**Function**: Multi-view polarized microscopy data structure

- Stores 5D data: [x, y, z, pol, view] - spatial, polarization states, views
- Manages microscope parameters: voxel dimensions, numerical apertures, detection axes
- Handles illumination and detection polarization configurations

#### 2.2 Microscope Class (SPIM/microscope/micro.py)

**Function**: Complete mathematical model of microscope system

- Integrates illumination system (Illuminator) and detection system (Detector)
- Calculates system transfer function matrix H
- Uses Gaunt coefficients for spherical harmonic transforms

#### 2.3 Illuminator Class (SPIM/microscope/ill.py)

**Function**: Illumination system modeling

- Calculates spherical harmonic representation of polarized light illumination
- Supports multiple polarization state configurations
- Uses paraxial approximation model

#### 2.4 Detector Class (SPIM/microscope/det.py)

**Function**: Detection system modeling

- Models detector spatial response function
- Calculates Point Spread Function (PSF)
- Supports different numerical apertures and optical axis configurations

### 3. Reconstruction Algorithms (SPIM/reconstruct/)

#### 3.1 recon_eGRL Series

**Function**: Implements enhanced Generalized Richardson-Lucy (eGRL) reconstruction algorithm

- `recon_eGRL_CPU.py`: CPU version, suitable for small datasets
- `recon_eGRL_GPU.py`: GPU version with CUDA acceleration
- `recon_eGRL_GPU_most.py`: Highly optimized GPU version for limited GPU memory

**Algorithm Features**:

- Iterative solution of inverse problems
- Based on Poisson noise model