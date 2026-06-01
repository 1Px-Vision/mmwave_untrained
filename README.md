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
