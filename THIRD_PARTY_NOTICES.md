# Third-Party Sources and Licenses

UMGS builds on or interoperates with the projects below. Each component remains
subject to its own license; this file supplements, rather than replaces, the
license files shipped beside redistributed source.

## 3D Gaussian Splatting and CUDA extensions

- Gaussian Splatting: <https://github.com/graphdeco-inria/gaussian-splatting>
- Differentiable Gaussian rasterizer:
  <https://github.com/graphdeco-inria/diff-gaussian-rasterization>
- Simple-KNN: <https://gitlab.inria.fr/bkerbl/simple-knn>
- Fused SSIM: <https://github.com/rahul-goel/fused-ssim>

The corresponding upstream license files are retained as `LICENSE.md` and in
the respective `submodules/` directories.

## LPIPS

The bundled `lpipsPyTorch/` implementation is derived from
<https://github.com/S-aiueo32/lpips-pytorch> and is distributed under the BSD
2-Clause license reproduced as `lpipsPyTorch/LICENSE`. It downloads the public
LPIPS weights from <https://github.com/richzhang/PerceptualSimilarity> on first
use unless they are already cached.

## MINIMA

Cross-sensor correspondence uses MINIMA with its RoMa-large backend:

- Repository: <https://github.com/LSXI7/MINIMA>
- Revision used for the reported campaign:
  `796e7721174f9f829b79b3702bf8c2ae9a3d447a`
- Checkpoint filename: `weights/minima_roma.pth`
- Checkpoint SHA-256:
  `17f3923bd780e8f1450792706e0643f70c8864ad11bf5fd4f2b54714bac23538`
- Upstream license: Apache-2.0. MINIMA documents additional terms for some
  alternative feature-extractor components; UMGS uses the RoMa backend.

MINIMA source and weights are not redistributed here. Install the pinned
revision, initialize its submodules as instructed upstream, download the
released `minima_roma.pth`, verify its hash, and pass both paths explicitly.

## MS-Splatting

The controlled external baseline is
<https://github.com/j-gruen/MS-Splatting> at revision
`21bf7123fea39a699af19562dada965fd36fc07e`. Its source and weights are not
redistributed. The repository records only the small Blackwell compatibility
patch used by the experiment; the patch changes optional dependency loading and
does not change the active baseline configuration or numerical method.

## COLMAP and OpenMVS

- COLMAP 3.7: <https://github.com/colmap/colmap>
- OpenMVS 2.3.0: <https://github.com/cdcseacave/openMVS>

These are external system dependencies. OpenMVS is used to construct a
validation-gated geometric proxy, not metric ground truth.
