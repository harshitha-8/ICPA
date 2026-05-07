# Skyfall-GS Candidate Notes For Cotton UAV 3D Phenotyping

## What Skyfall-GS Is

Skyfall-GS is a 2025 arXiv framework for synthesizing immersive 3D urban scenes
from multi-view satellite imagery. Its central idea is to use satellite imagery
for coarse large-scale geometry and diffusion-model refinement for closer-view
appearance, then represent the scene with 3D Gaussian Splatting for real-time
exploration.

The important design pattern is:

```text
remote-sensing imagery
  -> coarse geometry
  -> iterative refinement
  -> Gaussian-splat scene
  -> immersive navigation
```

## What Transfers To Cotton

The project is relevant to our cotton work because it addresses a similar
high-level problem: turning top-down/remote imagery into an explorable 3D scene.
The useful transferable ideas are:

- use the aerial/orthomosaic image as a **coarse scene scaffold**;
- refine local regions progressively instead of trying to solve the entire
  field at once;
- separate field-scale structure from close-up local detail;
- use Gaussian splatting for interactive scene navigation;
- evaluate cross-view consistency rather than relying only on pretty images.

For cotton, this suggests a two-level system:

1. field-scale UAV orthomap or 2.5D visibility map;
2. local post-defoliation boll patch reconstruction using overlapping frames.

## What Does Not Transfer Safely

Skyfall-GS uses generative refinement to synthesize plausible urban detail. That
is dangerous for phenotyping if used for cotton boll measurement, because a
diffusion model can invent organ shape that looks believable but is not
measured.

Do not use Skyfall-GS-style generation to compute:

- boll diameter;
- boll volume;
- boll count;
- yield or bale estimate;
- scientific morphology statistics.

If generative refinement is ever used, it should be confined to visualization or
hypothesis generation and clearly excluded from quantitative evaluation.

## Recommended Role In Our Project

Use Skyfall-GS as a **framing inspiration** for hierarchical reconstruction:

```text
UAV orthomap / nadir image
  -> field-level 2.5D scaffold
  -> identify visible post-defoliation boll patches
  -> local multi-view reconstruction with COLMAP, NAS3R, MASt3R, VGGT, or 3DGS
  -> project validated boll masks into local 3D
  -> compute traits only from calibrated geometry
```

The paper can cite or mention this direction only if needed in related work on
remote-image-to-3D scene generation. It should not be treated as an agriculture
baseline.

## Practical Lesson For The Current Figures

The current UAV 2.5D orthomap figure is the safe analogue of Skyfall-GS:

- real UAV texture is used;
- geometry is labeled as a proxy;
- no close-up boll shape is invented;
- the next step is local reconstruction from overlapping views.

This is the right standard for the paper: visually clear, but scientifically
honest.

## Decision

Skyfall-GS is useful for:

- motivating hierarchical remote-imagery-to-3D thinking;
- designing a field scaffold plus local-detail pipeline;
- future interactive Gaussian-splat scene visualization.

Skyfall-GS is not useful as:

- a direct cotton boll estimator;
- a metric reconstruction backend for tiny bolls;
- evidence that single orthomosaics can yield validated boll volume.

## Sources

- Skyfall-GS project/paper listing:
  https://huggingface.co/papers/2510.15869
- Skyfall-GS project page:
  https://skyfall-gs.jayinnn.dev/
- arXiv:
  https://arxiv.org/abs/2510.15869
