# Algorithms

This folder contains implementation-oriented snippets that support the paper.

The intended flow is:

```text
cotton_boll_detector.py
  -> per-image boxes, centers, masks/count priors
geometry_morphology.py
  -> multi-view association helpers, triangulation, diameter, volume, visibility
evaluation_protocol.py
  -> table-ready metrics for robustness experiments
```

The code in this folder should remain small, readable, and paper-facing. Heavy production pipelines can stay under `pipeline/`.
