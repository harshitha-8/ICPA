"""
Morphological Measurement Extraction — ICPA Cotton Boll Pipeline

Extracts per-boll geometric measurements from 3D point clouds:
diameter, girth, volume, surface curvature, visibility score.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Try importing open3d for point cloud operations
try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    logger.warning("Open3D not installed. Some functions will use NumPy fallbacks.")
    O3D_AVAILABLE = False


class BollMorphologyExtractor:
    """
    Extracts morphological measurements from 3D point cloud clusters
    identified as cotton boll instances.
    """
    
    def __init__(
        self,
        min_points_per_boll: int = 50,
        dbscan_eps: float = 0.005,  # meters
        dbscan_min_samples: int = 10,
        voxel_size: float = 0.001,  # meters, for downsampling
    ):
        self.min_points_per_boll = min_points_per_boll
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.voxel_size = voxel_size
    
    def cluster_bolls(self, points: np.ndarray, labels: np.ndarray = None) -> List[np.ndarray]:
        """
        Cluster boll-labeled 3D points into individual boll instances.
        
        Args:
            points: (N, 3) point cloud
            labels: (N,) semantic labels (optional, pre-filtered to boll points)
        
        Returns:
            List of point clusters, each (n_i, 3)
        """
        if labels is not None:
            boll_mask = labels == 'boll'
            boll_points = points[boll_mask]
        else:
            boll_points = points
        
        if len(boll_points) < self.min_points_per_boll:
            logger.warning(f"Only {len(boll_points)} boll points. Minimum is {self.min_points_per_boll}.")
            return []
        
        if O3D_AVAILABLE:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(boll_points)
            
            # DBSCAN clustering
            cluster_labels = np.array(pcd.cluster_dbscan(
                eps=self.dbscan_eps,
                min_points=self.dbscan_min_samples
            ))
        else:
            # Fallback: simple DBSCAN via sklearn
            try:
                from sklearn.cluster import DBSCAN
                clustering = DBSCAN(
                    eps=self.dbscan_eps,
                    min_samples=self.dbscan_min_samples
                ).fit(boll_points)
                cluster_labels = clustering.labels_
            except ImportError:
                logger.error("Neither Open3D nor sklearn available for clustering.")
                return []
        
        clusters = []
        unique_labels = set(cluster_labels)
        unique_labels.discard(-1)  # remove noise label
        
        for label in sorted(unique_labels):
            cluster = boll_points[cluster_labels == label]
            if len(cluster) >= self.min_points_per_boll:
                clusters.append(cluster)
        
        logger.info(f"Found {len(clusters)} boll clusters from {len(boll_points)} points.")
        return clusters
    
    def measure_boll(self, points: np.ndarray) -> Dict:
        """
        Extract morphological measurements from a single boll point cluster.
        
        Args:
            points: (n, 3) point cloud of a single boll instance
        
        Returns:
            Dict with diameter, girth, volume, curvature, etc.
        """
        n = len(points)
        
        # 1. Principal Component Analysis for oriented bounding box
        centroid = points.mean(axis=0)
        centered = points - centroid
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort by eigenvalue (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Project onto principal axes
        projected = centered @ eigenvectors
        
        # 2. Boll diameter: extent along principal axis
        diameter = float(projected[:, 0].max() - projected[:, 0].min())
        
        # 3. Equatorial girth: circumference at widest cross-section
        # Find the plane perpendicular to principal axis at maximum radial extent
        mid_axis = (projected[:, 0].max() + projected[:, 0].min()) / 2
        slice_mask = np.abs(projected[:, 0] - mid_axis) < diameter * 0.1
        if slice_mask.sum() >= 3:
            slice_points_2d = projected[slice_mask, 1:3]
            # Approximate girth as convex hull perimeter of the slice
            try:
                hull_2d = ConvexHull(slice_points_2d)
                girth = float(hull_2d.area)  # In 2D, ConvexHull.area = perimeter
            except Exception:
                girth = 0.0
        else:
            girth = np.pi * float(np.sqrt(eigenvalues[1] + eigenvalues[2]))
        
        # 4. Volume: convex hull volume
        try:
            hull_3d = ConvexHull(points)
            volume = float(hull_3d.volume)
        except Exception:
            # Fallback: approximate as ellipsoid
            semi_axes = np.sqrt(eigenvalues) * 2  # rough semi-axis lengths
            volume = (4.0 / 3.0) * np.pi * np.prod(semi_axes)
        
        # 5. Surface curvature: mean local curvature via PCA on neighborhoods
        curvatures = self._compute_local_curvatures(points, k=min(20, n-1))
        mean_curvature = float(np.mean(curvatures)) if len(curvatures) > 0 else 0.0
        
        # 6. Compactness: ratio of volume to bounding sphere volume
        max_dist = np.max(np.linalg.norm(centered, axis=1))
        sphere_volume = (4.0 / 3.0) * np.pi * max_dist**3
        compactness = volume / (sphere_volume + 1e-10)
        
        # 7. Aspect ratio
        if eigenvalues[2] > 0:
            aspect_ratio = float(np.sqrt(eigenvalues[0] / eigenvalues[2]))
        else:
            aspect_ratio = float('inf')
        
        return {
            'num_points': n,
            'centroid': centroid.tolist(),
            'diameter_m': diameter,
            'diameter_mm': diameter * 1000,
            'girth_m': girth,
            'girth_mm': girth * 1000,
            'volume_m3': volume,
            'volume_mm3': volume * 1e9,
            'mean_curvature': mean_curvature,
            'compactness': float(compactness),
            'aspect_ratio': aspect_ratio,
            'eigenvalues': eigenvalues.tolist(),
            'principal_axis': eigenvectors[:, 0].tolist(),
        }
    
    def _compute_local_curvatures(self, points: np.ndarray, k: int = 20) -> np.ndarray:
        """Estimate local surface curvature via eigenvalue ratio of local neighborhoods."""
        n = len(points)
        if n < k + 1:
            return np.array([])
        
        # Compute pairwise distances (memory-efficient for moderate point counts)
        if n > 10000:
            # Subsample for large clouds
            indices = np.random.choice(n, 1000, replace=False)
        else:
            indices = np.arange(n)
        
        curvatures = []
        dists = cdist(points[indices], points)
        
        for i, idx in enumerate(indices):
            nn_indices = np.argsort(dists[i])[:k+1]  # include self
            neighborhood = points[nn_indices]
            
            centered_nb = neighborhood - neighborhood.mean(axis=0)
            cov_nb = np.cov(centered_nb.T)
            evals = np.linalg.eigvalsh(cov_nb)
            evals = np.sort(evals)
            
            # Curvature = smallest eigenvalue / sum
            curvature = evals[0] / (evals.sum() + 1e-10)
            curvatures.append(curvature)
        
        return np.array(curvatures)
    
    def compute_visibility_score(
        self,
        boll_points: np.ndarray,
        camera_positions: np.ndarray,
        all_points: np.ndarray = None,
        num_rays: int = 1000
    ) -> Dict:
        """
        Compute visibility score for a boll instance via ray casting.
        
        Args:
            boll_points: (n, 3) boll point cloud
            camera_positions: (C, 3) UAV camera positions
            all_points: (M, 3) full scene point cloud for occlusion testing
            num_rays: number of random rays to cast
        
        Returns:
            Dict with visibility_score, occlusion_index, visible_cameras
        """
        centroid = boll_points.mean(axis=0)
        
        # Simple visibility: fraction of cameras that "see" the boll
        # (i.e., no occluding points between camera and boll centroid)
        visible_cameras = 0
        
        for cam_pos in camera_positions:
            direction = centroid - cam_pos
            distance = np.linalg.norm(direction)
            
            if distance < 0.1:  # too close, skip
                continue
            
            # Simple occlusion check: are there points between camera and boll?
            if all_points is not None and len(all_points) > 0:
                # Project all points onto the camera-boll line
                direction_norm = direction / distance
                to_points = all_points - cam_pos
                projections = np.dot(to_points, direction_norm)
                
                # Points that are between camera and boll and close to the line
                between_mask = (projections > 0.1) & (projections < distance - 0.01)
                if between_mask.any():
                    perp_dists = np.linalg.norm(
                        to_points[between_mask] - np.outer(projections[between_mask], direction_norm),
                        axis=1
                    )
                    # If any point is within boll radius of the line, it's occluded
                    boll_radius = np.max(np.linalg.norm(boll_points - centroid, axis=1))
                    if np.min(perp_dists) < boll_radius * 2:
                        continue
            
            visible_cameras += 1
        
        visibility = visible_cameras / max(len(camera_positions), 1)
        
        return {
            'visibility_score': float(visibility),
            'occlusion_index': float(1.0 - visibility),
            'visible_cameras': visible_cameras,
            'total_cameras': len(camera_positions),
        }
    
    def compare_conditions(
        self,
        pre_measurements: List[Dict],
        post_measurements: List[Dict]
    ) -> Dict:
        """
        Statistical comparison of morphological measurements
        between pre- and post-defoliation conditions.
        """
        from scipy import stats
        
        comparison = {}
        
        metrics_to_compare = [
            'diameter_mm', 'girth_mm', 'volume_mm3',
            'mean_curvature', 'compactness', 'aspect_ratio'
        ]
        
        for metric in metrics_to_compare:
            pre_values = [m[metric] for m in pre_measurements if metric in m]
            post_values = [m[metric] for m in post_measurements if metric in m]
            
            if len(pre_values) >= 2 and len(post_values) >= 2:
                # Welch's t-test (unequal variances)
                t_stat, p_value = stats.ttest_ind(pre_values, post_values, equal_var=False)
                
                comparison[metric] = {
                    'pre_mean': float(np.mean(pre_values)),
                    'pre_std': float(np.std(pre_values)),
                    'pre_n': len(pre_values),
                    'post_mean': float(np.mean(post_values)),
                    'post_std': float(np.std(post_values)),
                    'post_n': len(post_values),
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant_005': p_value < 0.05,
                    'effect_size_cohens_d': float(
                        (np.mean(post_values) - np.mean(pre_values)) /
                        np.sqrt((np.std(pre_values)**2 + np.std(post_values)**2) / 2)
                    )
                }
        
        # Boll count comparison
        comparison['boll_count'] = {
            'pre': len(pre_measurements),
            'post': len(post_measurements),
            'retention_ratio': len(post_measurements) / max(len(pre_measurements), 1)
        }
        
        return comparison


def main():
    """Main morphology extraction pipeline."""
    repo_root = Path(__file__).parent.parent.parent
    
    with open(repo_root / 'configs' / 'pipeline_config.yaml') as f:
        pipeline_config = yaml.safe_load(f)
    
    morph_config = pipeline_config['morphology']
    output_root = os.path.join(
        str(repo_root), 'outputs'
    )
    
    extractor = BollMorphologyExtractor(
        min_points_per_boll=morph_config['min_points_per_boll'],
        dbscan_eps=morph_config['dbscan_eps'],
        dbscan_min_samples=morph_config['dbscan_min_samples'],
    )
    
    # Load point clouds (if available from reconstruction stage)
    for condition in ['pre_defoliation', 'post_defoliation']:
        ply_path = os.path.join(output_root, 'colmap', condition, 'dense', 'fused.ply')
        
        if not os.path.exists(ply_path):
            logger.info(f"Point cloud not found for {condition}: {ply_path}")
            logger.info("Run reconstruction stage first.")
            continue
        
        if O3D_AVAILABLE:
            pcd = o3d.io.read_point_cloud(ply_path)
            points = np.asarray(pcd.points)
        else:
            logger.error("Open3D required for point cloud loading.")
            continue
        
        logger.info(f"\n{condition}: {len(points)} points loaded")
        
        # Cluster bolls (would use semantic labels in full pipeline)
        clusters = extractor.cluster_bolls(points)
        
        # Measure each boll
        measurements = []
        for i, cluster in enumerate(clusters):
            meas = extractor.measure_boll(cluster)
            meas['boll_id'] = i
            meas['condition'] = condition
            measurements.append(meas)
        
        # Save measurements
        meas_path = os.path.join(output_root, 'metrics', f'morphology_{condition}.json')
        os.makedirs(os.path.dirname(meas_path), exist_ok=True)
        with open(meas_path, 'w') as f:
            json.dump(measurements, f, indent=2)
        
        logger.info(f"Saved {len(measurements)} boll measurements → {meas_path}")
    
    logger.info("\nMorphology extraction complete.")


if __name__ == '__main__':
    main()
