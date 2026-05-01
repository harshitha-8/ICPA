# MVP Video Outputs

Generated MVP scene video:

- `icml_dataset_mvp_scene.mp4`
- `icml_dataset_mvp_preview.jpg`
- `cotton_pointcloud_flythrough.mp4`
- `cotton_pointcloud_flythrough_preview.jpg`

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

The video is an MVP scaffold. It shows selected reconstruction frames and a camera-path sketch. It does not claim solved metric 3D reconstruction yet.

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

This video uses 147,000 colored point samples and 158 candidate boll anchors. It is a 2.5D visual reconstruction scaffold, not a final metric SfM/Gaussian-splat result.
