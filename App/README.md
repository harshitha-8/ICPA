# ICPA Cotton 3D App

This folder contains the working local app for the ICPA cotton 3D reconstruction
prototype. It is adapted from the earlier `Cotton-3D-Reconstruction` prototype,
but kept lightweight for this repository:

- indexes the current `/Volumes/T9/ICML` pre/post-defoliation dataset,
- runs the reusable cotton boll detector from `algorithms/cotton_boll_detector.py`,
- builds a fast morphology-aware monocular depth proxy,
- renders a browser-based point-cloud viewer without Gradio or Plotly,
- extracts a gallery of high-confidence cotton-boll candidate crops,
- renders SAM-style cotton-lint masks over the real image for candidate review,
- creates a configurable plot-grid map proxy with row/column cell summaries,
- exports a local `.ply` scene point cloud and a `.csv` proxy measurement table,
- reports length, width, diameter, and volume proxies using a user-specified
  cm-per-pixel scale.

Run from the repository root:

```bash
/Users/harshu/miniconda3/bin/python App/app.py
```

Then open the printed local URL, usually:

```text
http://127.0.0.1:8917
```

The diameter and volume fields are not final biological measurements yet. They
become defensible only after camera calibration, ground sampling distance, or
ground-control scale validation is added to the reconstruction workflow.

Current proxy measurement logic:

- raw detector candidates are filtered into a measurement-ready subset;
- white lint fraction increases extraction confidence;
- green canopy fraction penalizes leaf-heavy boxes;
- detector boxes act as prompts for a deterministic SAM-style lint mask;
- mask length and width are estimated from the largest lint-like connected
  component in each prompted crop;
- candidate diameter is retained from the detector bounding-box width/height as
  a coarse fallback;
- diameter in centimeters is `diameter_px * cm_per_pixel`;
- length and width in centimeters use the same scale assumption;
- volume is reported both as a coarse spherical proxy and a mask-derived
  ellipsoid proxy;
- visibility is the contour area divided by bounding-box area;
- the gallery is sorted by extraction confidence so reviewers can inspect where
  the detector is reliable and where canopy occlusion causes failure.

SAM/SAM2 note: the current app does not claim to run Meta's official Segment
Anything model. It uses the same prompt-first logic as a lightweight local
placeholder: detector boxes prompt a cotton-lint mask. The defensible next step
is to replace `candidate_lint_mask()` with SAM/SAM2 inference and compare mask
IoU, length error, and volume error against a small expert-labeled validation
set.

Current plot mapping logic:

- a `4 x 43` grid is overlaid on the central study area of the selected image;
- measurement-ready candidates are assigned to image-coordinate cells;
- each cell reports boll count, mean diameter proxy, mean volume proxy, and mean
  extraction quality;
- this becomes meter-accurate only after orthomosaic, camera pose, GPS/GCP, or
  plot-boundary calibration is added.
