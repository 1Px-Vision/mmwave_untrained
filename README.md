# 77-GHz mmWave Stripmap SAR Simulation with PSF-LoS Optimization and Untrained Reconstruction

Simulator for generating and reconstructing 77-GHz mmWave radar images from an input target image. The code simulates a 2-D stripmap/serpentine scan, generates complex radar echo data, applies Line-of-Sight (LoS) antenna-beam weighting, performs exact PSF-normalized matched-filter backprojection, and optionally improves the image using an untrained optimization method or a complex-valued ResNet prior.

## Main Features
* 77-GHz mmWave / automotive-radar frequency band simulation.
* 2-D serpentine stripmap scanning trajectory similar to a real scanning stage.
* Complex stepped-frequency radar echo generation.
* Line-of-Sight antenna-beam weighting.
* Point-spread-function weighted matched-filter backprojection.
* PSF/sensitivity normalization to reduce strong side-lobe artifacts.
* Center-point PSF diagnostic image.
* Direct untrained PSF-LoS image optimization.
* Optional complex-valued ResNet untrained neural prior.
* Total variation regularization with multiple TV variants.
* GPU acceleration using PyTorch CUDA.

## System Block Diagram  
