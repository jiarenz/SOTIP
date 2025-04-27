# Improved Spiral Projection MR Fingerprinting via Memory-Efficient Synergic Optimization of 3D Spiral Trajectory, Image Reconstruction, and Parameter Estimation (SOTIP)

This repository contains the official implementation of:

> **Jiaren Zou, Yun Jiang, Sydney Kaplan, Nicole Seiberlich, and Yue Cao.**  
> *Improved Spiral Projection MR Fingerprinting via Memory-Efficient Synergic Optimization of 3D Spiral Trajectory, Image Reconstruction, and Parameter Estimation (SOTIP).*  
> IEEE Transactions on Medical Imaging, 2025.  
> [DOI: 10.1109/TMI.2025.3559467]

---

## Overview

**SOTIP** is a memory- and computation-efficient model-based deep learning (MBDL) framework for full 3D spiral MR Fingerprinting (MRF).  
It enables fast, high-resolution T1 and T2 mapping through:

- **Memory-efficient MBDL reconstruction** for non-Cartesian MRF.
- **Joint optimization** of temporal subspace image reconstruction and parameter estimation.
- **Rotation angle optimization** of 3D spiral sampling trajectories.

---

## Repository Structure

```text
SOTIP-master/
    fcnn.py                  # Fully Connected Neural Network for parameter estimation
    network_experiments.sh   # Bash script to organize training experiments
    phantom_generation.py    # Script to load phantoms and in vivo data
    train_CNN.py             # Main training script
    env.yaml                 # Environment file. Additional requirement: MIRTorch (https://github.com/guanhuaw/MIRTorch)
    unet/
        model.py             # U-Net for temporal subspace coefficient (TSC) image reconstruction
        unet_parts.py        # U-Net building blocks
    utils/
        data_processing.py   # Data preprocessing utilities
        data_analysis.py     # Evaluation and analysis tools
```
