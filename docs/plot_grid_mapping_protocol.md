# Plot Grid Mapping Protocol

## What Is Possible Now

The app can create a paper-style plot map over the selected UAV image:

- red study-site boundary;
- horizontal row dividers;
- yellow column dividers;
- cyan count markers for measurement-ready boll candidates;
- per-cell count, mean diameter proxy, mean volume proxy, and mean extraction
  quality.

This is currently an **image-coordinate plot-grid proxy**. It is useful for
inspection, figure drafts, and comparing pre/post distributions, but it is not
yet a surveyed geospatial map.

## What Is Needed For Metric Mapping

To make the map scientifically metric in meters, at least one is needed:

- a georeferenced orthomosaic;
- calibrated camera poses from COLMAP/VGGT/MASt3R;
- ground-control points or scale markers;
- reliable EXIF GPS/altitude plus camera calibration;
- plot boundary coordinates from field records.

## Current App Assumption

The current prototype uses a fixed `4 x 43` plot grid inspired by the reference
figure. The grid is placed over the central study area of the selected image.
This should be treated as a configurable scaffold until we wire in true plot
boundary coordinates.

## Paper Use

Use this as:

- qualitative figure: spatial distribution of extracted boll candidates;
- supplementary QC: which rows/columns are dense, sparse, or failure-prone;
- pre/post comparison: retained measurement-ready bolls per cell.

Do not use it yet as:

- meter-accurate yield map;
- row-level agronomic recommendation;
- final plot boundary segmentation.
