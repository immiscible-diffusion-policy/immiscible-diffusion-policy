# Source provenance and third-party notices

The core Hungarian assignment was extracted from the author's existing Immiscible Diffusion Policy experiment code. Original experiment files were copied/read, not moved or modified. This release has its own directory and does not include the Git history or remote configuration of the source repositories.

## Algorithm source

The source implementations inspected during extraction were:

- `diffusion_policy/diffusion_policy/policy/diffusion_unet_lowdim_policy.py`
- `diffusion_policy/diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py`
- `RoboTwin/policy/DP/diffusion_policy/policy/diffusion_unet_image_policy.py`
- `RoboTwin/policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/policy/dp3.py`

These are relative paths in the original research workspace, provided as provenance rather than runtime dependencies. The inspected source trees included local changes; a Git commit alone would not identify their full contents. The extracted operation is full-chunk Hungarian assignment. The KNN branch and action-window-only variant are not included. `unbiased=False` replaces `correction=0` for compatibility with PyTorch 1.12 and computes the same population standard deviation. The release adds input validation and device-explicit permutation indices. The test reference copies the original matching formula.

The action-noise assignment principle builds on **Immiscible Diffusion: Accelerating Diffusion Training with Noise Assignment**, Yiheng Li, Huayu Jiang, Atsushi Kodaira, Masayoshi Tomizuka, Kurt Keutzer, and Chenfeng Xu, NeurIPS 2024. No files from that project's repository are bundled.

## Bundled upstream code and patch context

The bundled backends and optional patches contain source from the following MIT-licensed projects:

| Project | Source | Notice |
|---|---|---|
| Diffusion Policy | https://github.com/real-stanford/diffusion_policy | Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab; [license](licenses/diffusion-policy.txt) |
| RoboTwin | https://github.com/RoboTwin-Platform/RoboTwin | Copyright (c) 2025 Tianxing Chen (陈天行); [license](licenses/robotwin.txt) |
| 3D Diffusion Policy | https://github.com/YanjieZe/3D-Diffusion-Policy | Copyright (c) 2024 Yanjie Ze; [license](licenses/dp3.txt) |

Each integration's `base.json` records the exact target revision and SHA-256 hashes of the unmodified patch-target files. These revisions are public integration bases, not claims about the historical source version used to train the paper's checkpoints. Full upstream license texts are retained in `licenses/`.

PyTorch, SciPy, NumPy, and optional PyYAML are dependencies installed separately; their source is not redistributed here.

## Integrated backend source copies

`backends/dp/` and `backends/dp3/` include the required upstream policy, model, dataset, training, and utility source files. Each backend's `UPSTREAM.json` identifies the source revision and original file hashes; its README explains release modifications. No source Git history or remote settings were copied. The DP Push-T renderer also preserves the inline MIT notice for Pymunk (Victor Blomqvist), and the asynchronous vector environment derives from MIT-licensed OpenAI Gym (see `licenses/gym.txt`).

The bundled DP3 configuration uses DDPM/epsilon prediction for both vanilla and Immiscible. The optional patch against external upstream DP3 preserves that upstream revision's DDIM/sample configuration. This distinction is documented in both backend and root READMEs.

## G1 deployment extraction

`deployment/g1/` adapts the author's `g1_dp` image-DP inference and deployment scripts. `deployment/g1/SOURCE.json` (relative to release root) records source paths, source fingerprints, release fingerprints, and external controller fingerprints. The three added DP image-policy/encoder factory files were copied from the locally modified RoboTwin DP tree; they retain the Diffusion Policy/RoboTwin notices above and are covered separately by that extraction manifest rather than the original DP backend's pinned-upstream file list. The optional G1 package has its own namespace, `idp_g1`.

The G1 adapter references the experiment's `openpi_g1_control` arm/hand controller interface and binary Dex3 presets. The controller files, Unitree SDK, camera server, and URDF are not redistributed. No redistributable license for the custom controller modules was found in the inspected checkout; obtain those dependencies separately. The adapter and this release's software checks do not certify hardware compatibility.
