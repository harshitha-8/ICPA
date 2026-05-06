"""
Splat-Based Morphology Extraction — ICPA Cotton Boll Pipeline

Extracts boll diameter, girth, volume, and count from 3D Gaussian
primitives rather than raw point clouds. Gaussian ellipsoids fill
texture-less gaps in white cotton, yielding better coverage.
"""

import os
import json
import struct
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import ConvexHull
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def load_gaussian_ply(ply_path: str) -> Dict[str, np.ndarray]:
    """
    Load a 3DGS-format PLY file and return Gaussian attributes.

    Returns dict with keys:
        xyz      (N, 3)  positions
        opacity  (N,)    sigmoid opacities
        scales   (N, 3)  scale values (exp of stored log-scales)
        rotations(N, 4)  quaternions wxyz
        sh_dc    (N, 3)  SH DC coefficients
    """
    with open(ply_path, "rb") as f:
        # Parse header
        header_lines = []
        while True:
            line = f.readline().decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        # Count properties and vertices
        num_vertices = 0
        properties = []
        for line in header_lines:
            if line.startswith("element vertex"):
                num_vertices = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                properties.append((parts[1], parts[2]))  # (type, name)

        # Determine if binary
        is_binary = any("binary" in l for l in header_lines)

        if is_binary:
            # Calculate struct format
            type_map = {"float": "f", "double": "d", "uchar": "B", "int": "i", "uint": "I"}
            fmt = "<" + "".join(type_map.get(p[0], "f") for p in properties)
            record_size = struct.calcsize(fmt)

            data = {}
            for pname in [p[1] for p in properties]:
                data[pname] = []

            for _ in range(num_vertices):
                record = struct.unpack(fmt, f.read(record_size))
                for idx, (_, pname) in enumerate(properties):
                    data[pname].append(record[idx])

            for k in data:
                data[k] = np.array(data[k], dtype=np.float32)
        else:
            # ASCII fallback
            data = {p[1]: [] for p in properties}
            for _ in range(num_vertices):
                values = f.readline().decode().strip().split()
                for idx, (_, pname) in enumerate(properties):
                    data[pname].append(float(values[idx]))
            for k in data:
                data[k] = np.array(data[k], dtype=np.float32)

    # Extract arrays
    xyz = np.column_stack([data["x"], data["y"], data["z"]])

    opacity_raw = data.get("opacity", np.ones(num_vertices))
    # Stored as logit; convert to probability
    opacity = 1.0 / (1.0 + np.exp(-opacity_raw))

    scales = np.column_stack([
        np.exp(data.get("scale_0", np.zeros(num_vertices))),
        np.exp(data.get("scale_1", np.zeros(num_vertices))),
        np.exp(data.get("scale_2", np.zeros(num_vertices))),
    ])

    rotations = np.column_stack([
        data.get("rot_0", np.ones(num_vertices)),
        data.get("rot_1", np.zeros(num_vertices)),
        data.get("rot_2", np.zeros(num_vertices)),
        data.get("rot_3", np.zeros(num_vertices)),
    ])

    sh_dc = np.column_stack([
        data.get("f_dc_0", np.zeros(num_vertices)),
        data.get("f_dc_1", np.zeros(num_vertices)),
        data.get("f_dc_2", np.zeros(num_vertices)),
    ])

    logger.info("Loaded %d Gaussians from %s", num_vertices, ply_path)
    return {
        "xyz": xyz, "opacity": opacity, "scales": scales,
        "rotations": rotations, "sh_dc": sh_dc,
    }


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


class GaussianMorphologyExtractor:
    """
    Extract cotton boll morphology from 3DGS Gaussian primitives.

    Key advantage over point-cloud methods: each Gaussian encodes an
    explicit ellipsoidal volume with scale and rotation, so we can
    compute volume and girth analytically rather than from sparse
    convex hulls.
    """

    def __init__(
        self,
        opacity_threshold: float = 0.3,
        min_scale: float = 0.0005,     # metres — filter dust
        max_scale: float = 0.05,       # metres — filter background
        dbscan_eps: float = 0.008,     # metres
        dbscan_min_samples: int = 5,
        min_gaussians_per_boll: int = 10,
    ):
        self.opacity_threshold = opacity_threshold
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.min_gaussians_per_boll = min_gaussians_per_boll

    def filter_boll_gaussians(self, gaussians: Dict) -> Dict:
        """Filter Gaussians to likely boll candidates by opacity and scale."""
        xyz = gaussians["xyz"]
        opacity = gaussians["opacity"]
        scales = gaussians["scales"]

        # Opacity filter
        op_mask = opacity > self.opacity_threshold

        # Scale filter: mean scale in valid range (not dust, not background)
        mean_scale = scales.mean(axis=1)
        scale_mask = (mean_scale > self.min_scale) & (mean_scale < self.max_scale)

        # Color filter: cotton bolls are white (high SH DC values)
        sh_dc = gaussians["sh_dc"]
        SH_C0 = 0.28209479177387814
        colors = 0.5 + SH_C0 * sh_dc
        brightness = colors.mean(axis=1)
        color_mask = brightness > 0.5  # bright = likely cotton

        combined = op_mask & scale_mask & color_mask
        logger.info("Filtered: %d / %d Gaussians pass boll criteria",
                     combined.sum(), len(xyz))

        return {k: v[combined] for k, v in gaussians.items()}

    def cluster_bolls(self, gaussians: Dict) -> List[Dict]:
        """Cluster filtered Gaussians into individual boll instances."""
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn required for DBSCAN clustering.")
            return []

        xyz = gaussians["xyz"]
        if len(xyz) < self.min_gaussians_per_boll:
            logger.warning("Too few Gaussians (%d) for clustering.", len(xyz))
            return []

        clustering = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
        ).fit(xyz)

        labels = clustering.labels_
        unique = set(labels)
        unique.discard(-1)

        clusters = []
        for label in sorted(unique):
            mask = labels == label
            if mask.sum() >= self.min_gaussians_per_boll:
                clusters.append({k: v[mask] for k, v in gaussians.items()})

        logger.info("Found %d boll clusters from %d Gaussians", len(clusters), len(xyz))
        return clusters

    def measure_boll(self, cluster: Dict) -> Dict:
        """
        Extract morphological measurements from a Gaussian boll cluster.

        Uses Gaussian ellipsoid parameters for volume and girth, which
        is more robust than point-cloud convex hull for sparse data.
        """
        xyz = cluster["xyz"]
        scales = cluster["scales"]
        opacity = cluster["opacity"]
        rotations = cluster["rotations"]
        N = len(xyz)

        centroid = np.average(xyz, weights=opacity, axis=0)
        centered = xyz - centroid

        # --- Diameter via PCA of Gaussian centres ---
        cov = np.cov(centered.T, aweights=opacity)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        projected = centered @ eigenvectors
        diameter = float(projected[:, 0].max() - projected[:, 0].min())

        # Add Gaussian extent: the outermost Gaussians extend by their scale
        max_scale_along_axis = 0
        for i in range(N):
            R = quat_to_rotation_matrix(rotations[i])
            # Project Gaussian semi-axes onto principal axis
            extent = np.abs(R @ scales[i])
            max_scale_along_axis = max(max_scale_along_axis, np.max(extent))
        diameter += 2 * max_scale_along_axis

        # --- Volume from Gaussian ellipsoids ---
        # Each Gaussian contributes an ellipsoidal volume weighted by opacity
        individual_volumes = (4.0 / 3.0) * np.pi * np.prod(scales, axis=1)
        volume = float(np.sum(individual_volumes * opacity))

        # Also compute convex hull volume as cross-check
        try:
            hull = ConvexHull(xyz)
            hull_volume = float(hull.volume)
        except Exception:
            hull_volume = volume

        # --- Girth (equatorial circumference) ---
        mid_axis = (projected[:, 0].max() + projected[:, 0].min()) / 2
        slice_mask = np.abs(projected[:, 0] - mid_axis) < diameter * 0.15
        if slice_mask.sum() >= 3:
            slice_2d = projected[slice_mask, 1:3]
            try:
                hull_2d = ConvexHull(slice_2d)
                girth = float(hull_2d.area)  # in 2D, .area = perimeter

                # Add Gaussian extent contribution
                slice_scales = scales[slice_mask].mean(axis=0)
                girth += 2 * np.pi * np.mean(slice_scales[1:])
            except Exception:
                girth = np.pi * diameter * 0.85
        else:
            girth = np.pi * diameter * 0.85  # approximate as circle

        # --- Compactness & aspect ratio ---
        max_dist = np.max(np.linalg.norm(centered, axis=1))
        sphere_vol = (4.0 / 3.0) * np.pi * max_dist ** 3
        compactness = volume / (sphere_vol + 1e-10)

        aspect_ratio = float(np.sqrt(eigenvalues[0] / (eigenvalues[2] + 1e-10)))

        # --- Mean Gaussian size (characteristic scale) ---
        mean_gaussian_radius = float(np.mean(scales))

        return {
            "num_gaussians": N,
            "centroid": centroid.tolist(),
            "diameter_m": diameter,
            "diameter_mm": diameter * 1000,
            "girth_m": girth,
            "girth_mm": girth * 1000,
            "volume_m3": volume,
            "volume_mm3": volume * 1e9,
            "hull_volume_m3": hull_volume,
            "hull_volume_mm3": hull_volume * 1e9,
            "compactness": float(compactness),
            "aspect_ratio": aspect_ratio,
            "mean_gaussian_radius_m": mean_gaussian_radius,
            "mean_opacity": float(np.mean(opacity)),
            "eigenvalues": eigenvalues.tolist(),
            "principal_axis": eigenvectors[:, 0].tolist(),
        }

    def extract_all(self, ply_path: str) -> List[Dict]:
        """Full pipeline: load → filter → cluster → measure."""
        gaussians = load_gaussian_ply(ply_path)
        filtered = self.filter_boll_gaussians(gaussians)
        clusters = self.cluster_bolls(filtered)

        measurements = []
        for i, cluster in enumerate(clusters):
            m = self.measure_boll(cluster)
            m["boll_id"] = i
            measurements.append(m)

        logger.info("Extracted measurements for %d bolls", len(measurements))
        return measurements


def main():
    """CLI entry for Gaussian-based morphology extraction."""
    import argparse

    repo_root = Path(__file__).parent.parent.parent
    parser = argparse.ArgumentParser(description="Extract boll morphology from 3DGS PLY")
    parser.add_argument("--condition", choices=["pre_defoliation", "post_defoliation", "both"], default="both")
    args = parser.parse_args()

    with open(repo_root / "configs" / "pipeline_config.yaml") as f:
        cfg = yaml.safe_load(f)

    gs_cfg = cfg.get("gaussian_splatting", {})
    morph_cfg = cfg.get("morphology", {})

    extractor = GaussianMorphologyExtractor(
        opacity_threshold=gs_cfg.get("opacity_threshold", 0.3),
        dbscan_eps=morph_cfg.get("dbscan_eps", 0.008),
        dbscan_min_samples=morph_cfg.get("dbscan_min_samples", 5),
        min_gaussians_per_boll=morph_cfg.get("min_points_per_boll", 10),
    )

    conditions = ["pre_defoliation", "post_defoliation"] if args.condition == "both" else [args.condition]

    for cond in conditions:
        ply_path = os.path.join(str(repo_root), "outputs", "gaussian_splats", cond, "point_cloud.ply")
        if not os.path.exists(ply_path):
            logger.warning("No 3DGS PLY for %s at %s — run training first.", cond, ply_path)
            continue

        measurements = extractor.extract_all(ply_path)

        out_path = os.path.join(str(repo_root), "outputs", "metrics", f"morphology_3dgs_{cond}.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(measurements, f, indent=2)

        logger.info("Saved %d boll measurements → %s", len(measurements), out_path)

        # Summary stats
        if measurements:
            diams = [m["diameter_mm"] for m in measurements]
            vols = [m["volume_mm3"] for m in measurements]
            logger.info("  Diameter: %.1f ± %.1f mm (n=%d)",
                        np.mean(diams), np.std(diams), len(diams))
            logger.info("  Volume:   %.1f ± %.1f mm³",
                        np.mean(vols), np.std(vols))


if __name__ == "__main__":
    main()
