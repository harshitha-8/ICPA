# UAV-to-Boll 3D Reconstruction Strategy

## Purpose

The current app is strong for counting, pre/post-defoliation scouting, and
row-column mapping. The weak point is not the count; it is metric 3D boll
geometry. The next investigation should therefore be staged:

1. reconstruct the UAV scene or orthomosaic-supported field region;
2. identify the most visible cotton boll clusters in post-defoliation imagery;
3. crop a local region of interest around a small number of well-exposed bolls;
4. reconstruct those local regions with stronger geometry;
5. estimate boll length, width, and volume only after scale calibration.

This keeps the paper accurate: whole-field counting can be reported now, while
boll volume and bale estimation remain calibrated downstream tasks.

## What The Recent 3D Reconstruction Landscape Suggests

### 1. Classical photogrammetry remains the first baseline

COLMAP/SfM-MVS is still the most defensible first baseline for UAV image
geometry because it provides camera poses, sparse points, and dense point clouds.
Cotton scenes are difficult for SfM because cotton lint is repetitive and
low-texture, but the baseline is necessary for reviewer confidence.

Use this for:

- camera pose recovery;
- sparse and dense point-cloud baseline;
- orthomosaic or local scene reconstruction support;
- scale anchoring when GPS/GCP/GSD is available.

Risk:

- repeated white bolls can cause weak feature matching;
- pre-defoliation foliage may dominate reconstruction;
- scale is ambiguous unless GSD, GPS/GCP, or measured references are used.

### 2. 3D Gaussian Splatting is useful for scene review, not automatically measurement

Cotton3DGaussians shows that 3D Gaussian Splatting can reconstruct high-fidelity
cotton plant models and use 2D masks from several views for 3D boll mapping and
trait extraction. The closest idea for this project is not to claim identical
single-plant reconstruction from UAV imagery, but to adapt the logic:

- build a local 3D scene;
- project 2D boll masks into the reconstructed space;
- cluster duplicate boll observations across views;
- estimate traits only where view support and scale are sufficient.

For this project, 3DGS should be a second-stage local reconstruction method,
after selecting a well-exposed post-defoliation patch. It should not be used
as the first claim for bale estimation.

### 3. Feed-forward Gaussian Splatting is promising but experimental

Recent systems such as YoNoSplat are moving toward fast reconstruction from
unstructured image collections, including unposed and uncalibrated inputs. This
direction is attractive for fast MVP experiments, but it should be treated as
research infrastructure rather than a guaranteed agricultural measurement tool.

Use this as:

- a rapid reconstruction candidate if code and GPU support are available;
- a future comparison against COLMAP plus 3DGS;
- a potential route for uncalibrated or weakly calibrated local image sets.

Do not use it yet for:

- final metric boll diameter;
- volume or bale estimation;
- paper claims without validation.

### 4. Social-media prototypes are useful for interface ideas

The football-on-3D-map concept is useful as a design analogy: put events on a
spatial substrate that users understand. In this cotton project, the safe
adaptation is the row-column scouting map, not a synthetic 3D terrain. The app
should keep the map/scouting structure and avoid decorative 3D unless the
geometry is real and calibrated.

Reddit and community posts around Gaussian Splatting repeatedly emphasize two
practical points:

- many pipelines still begin with frame extraction and COLMAP camera poses;
- real metric distance requires known scene scale, such as GCPs, measured
  objects, accurate GPS/RTK, or calibrated camera baselines.

That directly supports the current decision to avoid bale estimation from proxy
volume.

## Recommended Pipeline For The Next Investigation

### Stage A: Dataset audit and image selection

Select post-defoliation frames first, because bolls are more visible. For each
candidate local region, record:

- source folder and frame name;
- phase;
- estimated overlap with neighboring frames;
- visible row/column location;
- number of high-confidence bolls;
- whether bolls are isolated or strongly overlapping.

Select 3-5 local patches for reconstruction, not the full field at first.

### Stage B: Orthomosaic or scene-level reconstruction

Run a classical photogrammetry baseline:

1. image subset selection;
2. feature extraction and matching;
3. camera pose estimation;
4. sparse reconstruction;
5. dense point cloud or mesh;
6. orthomosaic/field coordinate alignment if possible.

Minimum outputs:

- camera poses;
- sparse point cloud;
- dense point cloud if available;
- reprojection error;
- number of registered images;
- scale source.

### Stage C: Local visible-boll crop

Use the app's row-column scouting map and detector overlay to choose a local
patch with strong visibility. The local patch should contain a manageable number
of bolls, ideally isolated enough for mask review.

Candidate criteria:

- post-defoliation preferred;
- high lint fraction;
- low green fraction;
- high extraction quality;
- multiple neighboring frames with the same boll visible;
- minimal motion blur and shadow.

### Stage D: Mask-guided local 3D reconstruction

For the selected patch:

1. crop the same local region across multiple frames;
2. run detector/mask extraction;
3. reconstruct the local patch with COLMAP/MVS;
4. train a local 3DGS if camera poses are stable;
5. project 2D masks into the 3D model;
6. cluster repeated observations of the same boll;
7. estimate geometry only for high-confidence clusters.

This is the correct place to reintroduce SAM-to-3D or mask-to-3D visualization.

### Stage E: Trait and bale-estimation boundary

For individual bolls, measure:

- count;
- visibility;
- mask length;
- mask width;
- calibrated diameter;
- calibrated volume proxy;
- view support;
- reconstruction confidence.

Bale estimation should not be computed directly from the current per-boll
volume proxy. A defensible bale estimate requires:

- calibrated plot area;
- boll count per plot or plant;
- boll weight calibration;
- lint turnout or gin turnout factor;
- moisture and maturity assumptions;
- validation against harvested yield or manual sample weights.

Until these are available, the app should report scouting and proxy morphology,
not bale output.

## Practical Method Ranking

| Rank | Method | Use now? | Why |
|---:|---|---|---|
| 1 | COLMAP/SfM-MVS local patch | Yes | Reviewer-defensible geometry baseline; gives camera poses and point cloud. |
| 2 | Orthomosaic/GCP plot alignment | Yes, if metadata exists | Needed for field-coordinate row/column mapping and scale. |
| 3 | Local 3DGS after stable poses | Next | Useful for visual reconstruction and mask projection in visible patches. |
| 4 | SAM/SAM2 masks on selected bolls | Next | Better mask quality, but must be validated against expert masks. |
| 5 | Feed-forward GS such as YoNoSplat | Exploratory | Promising for fast reconstruction; not yet the core measurement claim. |
| 6 | Decorative WebGL terrain/map views | No for paper claims | Useful for interface inspiration, not trait measurement. |

## Immediate Repo/App Decision

The current app should stay focused on:

- count;
- pre/post scouting;
- row-column map;
- measurement-ready candidates;
- proxy trait table;
- scene PLY export.

The removed SAM-to-3D overlay should return only after Stage D produces a local,
visually stable reconstruction. This avoids an AI-looking figure and protects
the paper from overclaiming.

## Sources Checked

- Cotton3DGaussians, Computers and Electronics in Agriculture, 2025:
  https://www.sciencedirect.com/science/article/pii/S0168169925003990
- YoNoSplat, ICLR 2026, Microsoft Research:
  https://www.microsoft.com/en-us/research/publication/yonosplat-you-only-need-one-model-for-feedforward-3d-gaussian-splatting/
- Metric assessment of 3D Gaussian Splatting for UAV-based reconstruction,
  ISPRS Archives, 2026:
  https://isprs-archives.copernicus.org/articles/XLVIII-2-W12-2026/143/2026/
- Reddit community discussion on metric distances in Gaussian Splatting:
  https://www.reddit.com/r/computervision/comments/1cysg3i
- Reddit community one-script video-to-3DGS pipeline example:
  https://www.reddit.com/r/GaussianSplatting/comments/1su92sf

Social posts and community projects are treated as implementation inspiration,
not as primary scientific evidence.
