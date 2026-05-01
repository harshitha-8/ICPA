# MVP Video Regeneration Notes

Generated MP4/JPG artifacts are intentionally not kept in Git. They are useful for local demos but are redundant because they can be regenerated from the dataset and scripts.

## Basic Scene MVP

Command:

```bash
/Users/harshu/miniconda3/bin/python tools/build_mvp_video.py \
  --manifest outputs/reconstruction_inputs/icml_dataset_sample/reconstruction_images.csv \
  --out-video outputs/videos/icml_dataset_mvp_scene.mp4 \
  --frames-dir outputs/videos/icml_dataset_mvp_frames \
  --fps 6 \
  --hold 6

ffmpeg -y \
  -framerate 6 \
  -i outputs/videos/icml_dataset_mvp_frames/frame_%05d.jpg \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  outputs/videos/icml_dataset_mvp_scene.mp4
```

This video is a scaffold. It shows selected reconstruction frames and a camera-path sketch. It does not claim solved metric 3D reconstruction.

## Cotton Point-Cloud Flythrough

Command:

```bash
/Users/harshu/miniconda3/bin/python tools/build_cotton_pointcloud_video.py \
  --manifest outputs/reconstruction_inputs/icml_dataset_sample/reconstruction_images.csv \
  --out-video outputs/videos/cotton_pointcloud_flythrough.mp4 \
  --frames-dir outputs/videos/cotton_pointcloud_flythrough_frames \
  --out-ply outputs/reconstruction_inputs/icml_dataset_sample/cotton_pointcloud_scaffold.ply \
  --limit 10 \
  --image-width 560 \
  --stride 4 \
  --anchors-per-tile 70 \
  --frames 150 \
  --fps 15

ffmpeg -y \
  -framerate 15 \
  -i outputs/videos/cotton_pointcloud_flythrough_frames/frame_%05d.jpg \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  outputs/videos/cotton_pointcloud_flythrough.mp4
```

This video uses colored point samples and candidate boll anchors. It is a 2.5D visual reconstruction scaffold, not a final metric SfM/Gaussian-splat result. Keep the script and command; do not commit the generated MP4, preview JPG, frames, or PLY unless a final supplementary artifact is explicitly selected.

## Pre/Post Measurement Reconstruction Video

Command:

```bash
/Users/harshu/miniconda3/bin/python tools/build_prepost_measurement_video.py \
  --manifest outputs/reconstruction_inputs/icml_dataset_sample/reconstruction_images.csv \
  --out-video outputs/videos/prepost_measurement_reconstruction.mp4 \
  --frames-dir outputs/videos/prepost_measurement_reconstruction_frames \
  --preview outputs/videos/prepost_measurement_reconstruction_preview.jpg \
  --fps 15

ffmpeg -y \
  -framerate 15 \
  -i outputs/videos/prepost_measurement_reconstruction_frames/frame_%05d.jpg \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  outputs/videos/prepost_measurement_reconstruction.mp4
```

This is the current best teaser artifact: real pre/post UAV frames, monochrome styling, detection-derived boll anchors, a 2.5D reconstruction flythrough, and proxy diameter/volume statistics. The measurement panel is intentionally labeled as proxy because centimeter-scale output requires GSD or field-scale calibration.
