# ICPA Cotton 3D App

This folder contains the working local app for the ICPA cotton 3D reconstruction
prototype. It is adapted from the earlier `Cotton-3D-Reconstruction` prototype,
but kept lightweight for this repository:

- indexes the current `/Volumes/T9/ICML` pre/post-defoliation dataset,
- runs the reusable cotton boll detector from `algorithms/cotton_boll_detector.py`,
- builds a fast morphology-aware monocular depth proxy,
- renders a browser-based point-cloud viewer without Gradio or Plotly,
- extracts a gallery of high-confidence cotton-boll candidate crops,
- exports a local `.ply` scene point cloud and a `.csv` proxy measurement table,
- reports diameter and volume proxies using a user-specified cm-per-pixel scale.

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

- candidate diameter is estimated from the detector bounding-box width/height;
- diameter in centimeters is `diameter_px * cm_per_pixel`;
- volume is a spherical proxy from that diameter;
- visibility is the contour area divided by bounding-box area;
- the gallery is sorted by visibility and depth score so reviewers can inspect
  where the detector is confident and where canopy occlusion causes failure.
