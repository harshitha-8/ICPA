# Topographical Boll-Density Notes

## Inspiration

The Topographical Explorer post turns flat geospatial heatmaps into interactive
3D elevation blocks. For this cotton project, the useful idea is not country
terrain. The useful idea is to make plot-level phenotyping maps easier to read
by extruding each plot cell according to a measured trait.

## Cotton Adaptation

The local app now renders a client-side topographical landscape where each
`4 x 43` plot cell becomes a block:

- height = measurement-ready boll count;
- color = count and extraction-quality intensity;
- base grid = current image-coordinate plot map.

This gives a fast visual QC layer for:

- boll-density hotspots;
- sparse or failure-prone plot regions;
- pre/post defoliation comparison;
- supplementary visualization.

## Accuracy Boundary

This is currently a plot-grid visualization in image coordinates. It is not a
surveyed geospatial surface. For a metric field map, we need orthomosaic, GCPs,
camera poses, GPS/altitude calibration, or field plot boundary coordinates.

## Source

- Topographical Explorer post: https://x.com/AhmedShahnab/status/2050757234200838182
