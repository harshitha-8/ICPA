# Algorithms

This folder contains implementation-oriented snippets that support the paper.

The intended flow is:

```text
cotton_boll_detector.py
  -> per-image boxes, centers, masks/count priors
run_dataset_counter.py
  -> phase-aware counting across mixed pre/post dataset folders
prepare_reconstruction_inputs.py
  -> sharp-frame selection and camera-path scaffold for 3D reconstruction
geometry_morphology.py
  -> multi-view association helpers, triangulation, diameter, volume, visibility
evaluation_protocol.py
  -> table-ready metrics for robustness experiments
scene_viewer_manifest.py
  -> export reconstructed scenes, camera paths, and boll anchors for 3D inspection
```

The code in this folder should remain small, readable, and paper-facing. Heavy production pipelines can stay under `pipeline/`.
