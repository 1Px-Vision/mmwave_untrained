# 77-GHz mmWave Stripmap SAR Simulation with PSF-LoS Optimization and Untrained Reconstruction

Simulator for generating and reconstructing 77-GHz mmWave radar images from an input target image. The code simulates a 2-D stripmap/serpentine scan, generates complex radar echo data, applies Line-of-Sight (LoS) antenna-beam weighting, performs exact PSF-normalized matched-filter backprojection, and optionally improves the image using an untrained optimization method or a complex-valued ResNet prior.

![](https://github.com/1Px-Vision/mmwave_untrained/blob/main/3D-LoS-Stripmap-Scan-Geometry.jpg)

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

The direct, untrained reconstruction optimizes the image itself rather than training on a dataset. The image is initialized from the PSF-normalized backprojection and then refined by minimizing the radar-domain measurement error.

![](https://github.com/1Px-Vision/mmwave_untrained/blob/main/mmWave_System_Block_Diagram.jpg)

### Loss Function

The untrained PSF-LoS reconstruction is optimized by minimizing a radar-domain data-fidelity loss combined with image-domain regularization terms:

```math
\mathcal{L}
=
\mathcal{L}_{\mathrm{meas}}
+
\lambda_{\mathrm{TV}}\mathcal{L}_{\mathrm{TV}}
+
\lambda_{\mathrm{sparse}}\|\rho\|_1
+
\lambda_{\mathrm{lap}}\mathcal{L}_{\mathrm{lap}}
```

where the complex measurement loss is defined as:

$\mathcal{L}_{meas}= \left\|
\Re(H\rho)-\Re(S)
\right\|
+
\left\|
\Im(H\rho)-\Im(S)
\right\|
$
Here, (S) is the measured or simulated complex radar echo, (\rho) is the reconstructed scene reflectivity, and (H) is the PSF-LoS forward radar operator. The measurement term enforces consistency between the predicted radar echo (H\rho) and the acquired echo (S) in both real and imaginary components.

The regularization terms improve the stability and visual quality of the reconstruction:

* $\mathcal{L}_{\mathrm{TV}}$: Total variation regularization term used to suppress noise, reduce speckle artifacts, and promote spatially smooth reflectivity regions.

* $|\rho|_1$: Sparsity constraint applied to the reconstructed reflectivity map. This term encourages compact scattering responses and reduces weak background artifacts.

* $\mathcal{L}_{\mathrm{lap}}$: Laplacian smoothness regularization term used to reduce high-frequency artifacts, ringing effects, and PSF side-lobes in the reconstructed radar image.

* $\lambda_{\mathrm{TV}}$, $\lambda_{\mathrm{sparse}}$, and $\lambda_{\mathrm{lap}}$: Regularization weights that control the relative contribution of the total variation, sparsity, and Laplacian terms in the total loss function.


This formulation allows the reconstruction to be optimized without requiring a pre-trained dataset. The image is recovered by directly minimizing the mismatch between the predicted and measured mmWave radar echoes while enforcing physically meaningful image priors.

## Network Structure
The ComplexResNetPrior is used as an untrained neural prior for complex-valued radar image reconstruction. The network processes the real and imaginary components of the reflectivity map using parallel complex convolution operations.

```
ComplexResNetPrior
├── Head
│   └── ComplexConv2d → ComplexBatchNorm2d → ComplexReLU
├── Body
│   ├── ResBlock 1
│   │   ├── ComplexConv2d
│   │   ├── ComplexBatchNorm2d
│   │   ├── ComplexReLU
│   │   ├── ComplexConv2d
│   │   └── ComplexBatchNorm2d
│   ├── ResBlock 2
│   ├── ...
│   └── ResBlock depth
└── Tail
    └── ComplexConv2d
```

Each residual block refines the complex-valued feature representation while preserving the input information through skip connections. The head module extracts low-level, complex features; the body module performs iterative residual feature enhancement; and the tail module maps the final complex features to the reconstructed radar reflectivity image. The network is not trained using an external dataset. Instead, its weights are optimized directly for a single radar measurement by minimizing the mismatch between the predicted complex echo and the measured or simulated echo.

## Recommended GPU Run

```
!python mmwave_stripmap_untrained.py \
  --use-uploaded-image /content/Radar_in.jpg \
  --device cuda \
  --nx 64 --nz 64 \
  --na 128 --nf 128 \
  --n-iter 800 \
  --recon-mode psf-los \
  --target-preprocess heatmap \
  --beam-fwhm-deg 42 \
  --ap-window hann \
  --freq-window hann \
  --tv-type 3 \
  --tv-weight 2e-4 \
  --sparse-weight 2e-5 \
  --lap-weight 5e-5

  ```

### Main Outputs
 ```
Result_mmWaveSAR_PSF_LoS/
├── 01_target_reflectivity.png
├── 02_scan_trajectory.png
├── 03_simulated_echo.png
├── 04_psf_weighted_backprojection.png
├── 05_los_sensitivity.png
├── 06_center_psf.png
├── 07_psf_los_optimized_reconstruction.png
├── 08_complex_resnet_reconstruction.png
├── loss_curve_psf_los.png
├── loss_curve_complex_resnet.png
└── simulated_mmwave_psf_los_data.npz
 ```

## Qualitative mmWave Radar Imaging Results

The figure shows qualitative reconstruction results obtained using the proposed mmWave stripmap SAR simulation and untrained PSF-LoS reconstruction framework. Each target object is shown together with its corresponding reconstructed radar image. The top image of each pair represents the optical reference target, while the bottom image shows the reconstructed mmWave radar reflectivity map. The color scale represents the relative scattering intensity, where blue regions correspond to weak radar returns and yellow/red regions indicate strong radar reflections.

![](https://github.com/1Px-Vision/mmwave_untrained/blob/main/mmWave_Result.jpg)

The results demonstrate that the proposed method can recover the main geometric structure of different metallic and high-reflectivity objects, including a star-shaped target, knife, pliers, cutter, hammer, scissors, metallic can, handgun-like object, fork, carabiner, star-shaped metallic sample, and wrench. Strong responses are mainly concentrated around object edges, corners, tips, and elongated metallic structures, which are typical dominant scattering regions in mmWave radar imaging. Although the reconstructed radar images are blurrier than the optical references, the principal object shapes remain distinguishable. This behavior is expected because mmWave radar imaging depends on the electromagnetic scattering response rather than visible texture. Specular reflections, limited aperture sampling, point-spread-function side-lobes, and Line-of-Sight geometry can produce artifacts, shadowed regions, and non-uniform intensity distributions. Overall, the results confirm that the PSF-LoS stripmap reconstruction and untrained optimization approach can generate interpretable radar images from different object geometries without requiring a supervised training dataset.
