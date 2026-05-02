# ICPA Cotton 3D App

This folder contains the working local app for the ICPA cotton 3D reconstruction
prototype. It is adapted from the earlier `Cotton-3D-Reconstruction` prototype,
but kept lightweight for this repository:

- indexes the current `/Volumes/T9/ICML` pre/post-defoliation dataset,
- runs the reusable cotton boll detector from `algorithms/cotton_boll_detector.py`,
- builds a fast morphology-aware monocular depth proxy,
- renders a browser-based point-cloud viewer without Gradio or Plotly,
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
