"""
Reconstruction Evaluation — ICPA Cotton Boll Pipeline

Computes reconstruction quality metrics: completeness, noise,
boll retention rate, and comparison across methods/conditions.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial import KDTree
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False


def load_point_cloud(path: str) -> np.ndarray:
    """Load point cloud from PLY file."""
    if O3D_AVAILABLE:
        pcd = o3d.io.read_point_cloud(path)
        return np.asarray(pcd.points)
    else:
        # Simple PLY reader for ASCII format
        points = []
        header_done = False
        with open(path, 'r') as f:
            for line in f:
                if 'end_header' in line:
                    header_done = True
                    continue
                if header_done:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        try:
                            points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        except ValueError:
                            continue
        return np.array(points)


def compute_reconstruction_completeness(
    points: np.ndarray,
    field_bounds: Optional[np.ndarray] = None,
    voxel_size: float = 0.05
) -> Dict:
    """
    Compute reconstruction completeness as fraction of voxels occupied.
    
    Args:
        points: (N, 3) point cloud
        field_bounds: (2, 3) min/max bounds of the field. If None, uses data extent.
        voxel_size: size of voxel grid cells in meters
    """
    if len(points) == 0:
        return {'completeness': 0.0, 'num_points': 0}
    
    if field_bounds is None:
        bounds_min = points.min(axis=0)
        bounds_max = points.max(axis=0)
    else:
        bounds_min, bounds_max = field_bounds
    
    extent = bounds_max - bounds_min
    grid_dims = np.ceil(extent / voxel_size).astype(int)
    total_voxels = int(np.prod(grid_dims))
    
    if total_voxels == 0:
        return {'completeness': 0.0, 'num_points': len(points)}
    
    # Voxelize
    voxel_indices = ((points - bounds_min) / voxel_size).astype(int)
    voxel_indices = np.clip(voxel_indices, 0, grid_dims - 1)
    
    # Unique occupied voxels
    unique_voxels = set(map(tuple, voxel_indices))
    occupied = len(unique_voxels)
    
    return {
        'completeness': float(occupied / total_voxels),
        'occupied_voxels': occupied,
        'total_voxels': total_voxels,
        'voxel_size': voxel_size,
        'num_points': len(points),
        'extent_m': extent.tolist(),
        'grid_dims': grid_dims.tolist(),
    }


def compute_noise_level(points: np.ndarray, k: int = 5) -> Dict:
    """
    Estimate noise level as mean k-nearest-neighbor distance.
    """
    if len(points) < k + 1:
        return {'mean_nn_distance': 0.0}
    
    # Subsample for speed
    if len(points) > 50000:
        idx = np.random.choice(len(points), 50000, replace=False)
        sample = points[idx]
    else:
        sample = points
    
    tree = KDTree(sample)
    dists, _ = tree.query(sample, k=k+1)  # k+1 because first is self
    nn_dists = dists[:, 1:]  # exclude self
    
    return {
        'mean_nn_distance': float(np.mean(nn_dists)),
        'median_nn_distance': float(np.median(nn_dists)),
        'std_nn_distance': float(np.std(nn_dists)),
        'p95_nn_distance': float(np.percentile(nn_dists, 95)),
        'sampled_points': len(sample),
    }


def compute_boll_retention_rate(
    boll_count_2d: int,
    boll_count_3d: int
) -> Dict:
    """Compute Boll Retention Rate (BRR)."""
    if boll_count_2d == 0:
        return {'brr': 0.0}
    
    return {
        'brr': float(boll_count_3d / boll_count_2d),
        'boll_count_2d': boll_count_2d,
        'boll_count_3d': boll_count_3d,
        'loss_count': boll_count_2d - boll_count_3d,
        'loss_rate': float(1.0 - boll_count_3d / boll_count_2d),
    }


def compare_reconstructions(
    recon_a_path: str,
    recon_b_path: str,
    voxel_size: float = 0.05
) -> Dict:
    """Compare two reconstructions (e.g., COLMAP vs semantic)."""
    points_a = load_point_cloud(recon_a_path)
    points_b = load_point_cloud(recon_b_path)
    
    comp_a = compute_reconstruction_completeness(points_a, voxel_size=voxel_size)
    comp_b = compute_reconstruction_completeness(points_b, voxel_size=voxel_size)
    noise_a = compute_noise_level(points_a)
    noise_b = compute_noise_level(points_b)
    
    # Chamfer distance between the two reconstructions
    if len(points_a) > 0 and len(points_b) > 0:
        # Subsample for efficiency
        if len(points_a) > 10000:
            idx = np.random.choice(len(points_a), 10000, replace=False)
            pts_a = points_a[idx]
        else:
            pts_a = points_a
        
        if len(points_b) > 10000:
            idx = np.random.choice(len(points_b), 10000, replace=False)
            pts_b = points_b[idx]
        else:
            pts_b = points_b
        
        tree_a = KDTree(pts_a)
        tree_b = KDTree(pts_b)
        
        dist_a_to_b, _ = tree_b.query(pts_a)
        dist_b_to_a, _ = tree_a.query(pts_b)
        
        chamfer = float(np.mean(dist_a_to_b) + np.mean(dist_b_to_a)) / 2
    else:
        chamfer = float('inf')
    
    return {
        'recon_a': {
            'path': recon_a_path,
            'completeness': comp_a,
            'noise': noise_a,
        },
        'recon_b': {
            'path': recon_b_path,
            'completeness': comp_b,
            'noise': noise_b,
        },
        'chamfer_distance': chamfer,
    }


def evaluate_all(output_root: str, voxel_size: float = 0.05) -> Dict:
    """Evaluate all available reconstructions."""
    results = {}
    
    # Check for COLMAP reconstructions
    for condition in ['pre_defoliation', 'post_defoliation']:
        colmap_ply = os.path.join(output_root, 'colmap', condition, 'dense', 'fused.ply')
        semantic_ply = os.path.join(output_root, 'pointclouds', f'semantic_{condition}.ply')
        
        for method, ply_path in [('colmap', colmap_ply), ('semantic', semantic_ply)]:
            key = f"{method}_{condition}"
            
            if os.path.exists(ply_path):
                points = load_point_cloud(ply_path)
                
                results[key] = {
                    'method': method,
                    'condition': condition,
                    'path': ply_path,
                    'completeness': compute_reconstruction_completeness(points, voxel_size=voxel_size),
                    'noise': compute_noise_level(points),
                }
                logger.info(f"{key}: {len(points)} points, completeness={results[key]['completeness']['completeness']:.3f}")
            else:
                logger.info(f"{key}: not found at {ply_path}")
    
    return results


def main():
    repo_root = Path(__file__).parent.parent.parent
    
    with open(repo_root / 'configs' / 'pipeline_config.yaml') as f:
        config = yaml.safe_load(f)
    with open(repo_root / 'configs' / 'dataset_config.yaml') as f:
        dataset_config = yaml.safe_load(f)
    
    output_root = dataset_config['output']['root']
    voxel_size = config['reconstruction']['evaluation']['completeness_threshold']
    
    results = evaluate_all(output_root, voxel_size)
    
    output_path = os.path.join(output_root, 'metrics', 'reconstruction_evaluation.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nEvaluation results saved to {output_path}")


if __name__ == '__main__':
    main()
