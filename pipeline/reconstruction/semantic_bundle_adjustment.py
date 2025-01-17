"""
Semantic Bundle Adjustment — ICPA Cotton Boll Pipeline

Feature-aligned 3D reconstruction using DINOv2 semantic correspondences
for camera pose estimation and dense matching in embedding space.
"""

import os
import sys
import json
import glob
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class SemanticBundleAdjuster:
    """
    Performs structure-from-motion using DINOv2 semantic correspondences
    instead of / in addition to traditional pixel-level matching.
    
    Three modes:
    1. Semantic-only: all correspondences from DINOv2
    2. Hybrid: DINOv2 + SIFT correspondences with configurable weighting
    3. Refinement: COLMAP initialization + semantic feature-metric refinement
    """
    
    def __init__(
        self,
        feature_weight: float = 0.3,
        geometric_weight: float = 0.7,
        num_iterations: int = 1000,
        learning_rate: float = 0.001,
        patch_size: int = 14,
    ):
        self.feature_weight = feature_weight
        self.geometric_weight = geometric_weight
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.patch_size = patch_size
    
    def load_correspondences(
        self,
        feature_dir: str,
        image_paths: List[str],
        threshold: float = 0.7,
        max_pairs: int = None
    ) -> Dict:
        """
        Load precomputed DINOv2 features and compute correspondences
        for all overlapping image pairs.
        """
        from pipeline.feature_alignment.extract_dinov2_features import (
            compute_pairwise_similarity
        )
        
        n = len(image_paths)
        all_correspondences = {}
        pair_count = 0
        
        for i in range(n):
            for j in range(i + 1, min(i + 10, n)):  # consecutive pairs likely overlap
                stem_i = Path(image_paths[i]).stem
                stem_j = Path(image_paths[j]).stem
                
                feat_i_path = os.path.join(feature_dir, f"{stem_i}_dinov2.npz")
                feat_j_path = os.path.join(feature_dir, f"{stem_j}_dinov2.npz")
                
                if not (os.path.exists(feat_i_path) and os.path.exists(feat_j_path)):
                    continue
                
                fi = np.load(feat_i_path)['patch_features'].astype(np.float32)
                fj = np.load(feat_j_path)['patch_features'].astype(np.float32)
                
                result = compute_pairwise_similarity(fi, fj, threshold)
                
                if result['num_correspondences'] > 10:
                    all_correspondences[(i, j)] = result
                    pair_count += 1
                
                if max_pairs and pair_count >= max_pairs:
                    break
            
            if max_pairs and pair_count >= max_pairs:
                break
        
        logger.info(f"Computed correspondences for {pair_count} image pairs.")
        return all_correspondences
    
    def upscale_correspondences(
        self,
        correspondences: List[Dict],
        image_size: Tuple[int, int],
        patch_grid: Tuple[int, int]
    ) -> List[Dict]:
        """
        Convert patch-level correspondences to pixel-level coordinates.
        """
        h_grid, w_grid = patch_grid
        H, W = image_size
        
        scale_x = W / w_grid
        scale_y = H / h_grid
        
        pixel_correspondences = []
        for c in correspondences:
            px_a = (
                (c['a'][0] + 0.5) * scale_x,
                (c['a'][1] + 0.5) * scale_y
            )
            px_b = (
                (c['b'][0] + 0.5) * scale_x,
                (c['b'][1] + 0.5) * scale_y
            )
            pixel_correspondences.append({
                'pt_a': px_a,
                'pt_b': px_b,
                'similarity': c['similarity']
            })
        
        return pixel_correspondences
    
    def estimate_poses_from_semantic(
        self,
        correspondences: Dict,
        image_sizes: List[Tuple[int, int]],
        camera_matrix: np.ndarray = None
    ) -> Dict:
        """
        Estimate camera poses from semantic correspondences using
        essential matrix decomposition.
        
        This is a simplified version; full implementation would use
        incremental SfM with robust estimation.
        """
        import cv2
        
        poses = {}  # image_index -> (R, t)
        poses[0] = (np.eye(3), np.zeros(3))  # reference frame
        
        registered = {0}
        
        # Simple sequential registration
        for (i, j), corr_data in sorted(correspondences.items()):
            if i not in registered and j not in registered:
                continue
            
            if i in registered and j not in registered:
                ref_idx, new_idx = i, j
            elif j in registered and i not in registered:
                ref_idx, new_idx = j, i
            else:
                continue  # both already registered
            
            # Extract pixel correspondences
            matches = corr_data.get('correspondences', [])
            if len(matches) < 8:
                continue
            
            pts_ref = np.array([[m['a'][0], m['a'][1]] for m in matches], dtype=np.float64)
            pts_new = np.array([[m['b'][0], m['b'][1]] for m in matches], dtype=np.float64)
            
            if ref_idx == j:
                pts_ref, pts_new = pts_new, pts_ref
            
            # Estimate camera matrix if not provided
            if camera_matrix is None:
                H, W = image_sizes[0]
                f = max(H, W) * 1.2  # rough focal length estimate
                camera_matrix = np.array([
                    [f, 0, W/2],
                    [0, f, H/2],
                    [0, 0, 1]
                ], dtype=np.float64)
            
            # Essential matrix
            E, mask = cv2.findEssentialMat(
                pts_ref, pts_new, camera_matrix,
                method=cv2.RANSAC, prob=0.999, threshold=1.0
            )
            
            if E is None:
                continue
            
            # Recover pose
            _, R, t, mask_pose = cv2.recoverPose(
                E, pts_ref, pts_new, camera_matrix
            )
            
            # Compose with reference pose
            R_ref, t_ref = poses[ref_idx]
            R_new = R @ R_ref
            t_new = (R @ t_ref.reshape(3, 1) + t).flatten()
            
            poses[new_idx] = (R_new, t_new)
            registered.add(new_idx)
            
            logger.debug(f"Registered image {new_idx} (inliers: {mask.sum()})")
        
        logger.info(f"Registered {len(poses)}/{max(max(k for pair in correspondences for k in pair) + 1, 0)} images")
        
        return {
            'poses': {k: {'R': v[0].tolist(), 't': v[1].tolist()} for k, v in poses.items()},
            'num_registered': len(poses),
            'camera_matrix': camera_matrix.tolist() if camera_matrix is not None else None
        }
    
    def compute_semantic_loss(
        self,
        features_a: np.ndarray,
        features_b: np.ndarray,
        correspondences: List[Dict]
    ) -> float:
        """
        Compute semantic consistency loss for a set of correspondences.
        L_semantic = 1 - mean(cosine_similarity(f_a, f_b)) for all matches.
        """
        if not correspondences:
            return 1.0
        
        sims = [c['similarity'] for c in correspondences]
        return 1.0 - float(np.mean(sims))
    
    def triangulate_points(
        self,
        correspondences: Dict,
        poses: Dict,
        camera_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Triangulate 3D points from semantic correspondences and estimated poses.
        """
        import cv2
        
        all_points_3d = []
        
        for (i, j), corr_data in correspondences.items():
            if str(i) not in poses['poses'] or str(j) not in poses['poses']:
                continue
            
            R_i = np.array(poses['poses'][str(i)]['R'])
            t_i = np.array(poses['poses'][str(i)]['t']).reshape(3, 1)
            R_j = np.array(poses['poses'][str(j)]['R'])
            t_j = np.array(poses['poses'][str(j)]['t']).reshape(3, 1)
            
            K = np.array(camera_matrix) if isinstance(camera_matrix, list) else camera_matrix
            
            P_i = K @ np.hstack([R_i, t_i])
            P_j = K @ np.hstack([R_j, t_j])
            
            matches = corr_data.get('correspondences', [])
            if len(matches) < 2:
                continue
            
            pts_i = np.array([[m['a'][0], m['a'][1]] for m in matches], dtype=np.float64).T
            pts_j = np.array([[m['b'][0], m['b'][1]] for m in matches], dtype=np.float64).T
            
            points_4d = cv2.triangulatePoints(P_i, P_j, pts_i, pts_j)
            points_3d = (points_4d[:3] / points_4d[3:]).T
            
            # Filter invalid points
            valid = np.all(np.isfinite(points_3d), axis=1)
            all_points_3d.append(points_3d[valid])
        
        if all_points_3d:
            return np.vstack(all_points_3d)
        return np.zeros((0, 3))


def main():
    """Run semantic bundle adjustment pipeline."""
    repo_root = Path(__file__).parent.parent.parent
    
    with open(repo_root / 'configs' / 'pipeline_config.yaml') as f:
        pipeline_config = yaml.safe_load(f)
    with open(repo_root / 'configs' / 'dataset_config.yaml') as f:
        dataset_config = yaml.safe_load(f)
    
    sem_config = pipeline_config['reconstruction']['semantic']
    dataset_root = dataset_config['dataset']['root']
    output_root = dataset_config['output']['root']
    
    sba = SemanticBundleAdjuster(
        feature_weight=sem_config['feature_weight'],
        geometric_weight=sem_config['geometric_weight'],
        num_iterations=sem_config['num_iterations'],
        learning_rate=sem_config['learning_rate'],
    )
    
    for condition_key in ['pre_defoliation', 'post_defoliation']:
        condition = dataset_config['dataset'][condition_key]
        
        images = []
        for folder in condition['folders']:
            folder_path = os.path.join(dataset_root, folder)
            if os.path.exists(folder_path):
                imgs = sorted(glob.glob(os.path.join(folder_path, '*.JPG')))
                imgs = [p for p in imgs if not os.path.basename(p).startswith('._')]
                images.extend(imgs)
        
        feature_dir = os.path.join(output_root, 'features', condition_key)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Semantic BA: {condition_key} ({len(images)} images)")
        logger.info(f"{'='*60}")
        
        # Load correspondences
        correspondences = sba.load_correspondences(
            feature_dir, images,
            threshold=pipeline_config['feature_extraction']['matching']['threshold'],
            max_pairs=500
        )
        
        if not correspondences:
            logger.warning(f"No correspondences found for {condition_key}. Run feature extraction first.")
            continue
        
        # Estimate poses
        from PIL import Image as PILImage
        sample_img = PILImage.open(images[0])
        image_sizes = [sample_img.size[::-1]] * len(images)  # (H, W)
        
        poses = sba.estimate_poses_from_semantic(correspondences, image_sizes)
        
        # Save poses
        pose_path = os.path.join(output_root, 'colmap', f'semantic_{condition_key}', 'poses.json')
        os.makedirs(os.path.dirname(pose_path), exist_ok=True)
        with open(pose_path, 'w') as f:
            json.dump(poses, f, indent=2)
        
        # Triangulate
        if poses['camera_matrix']:
            points_3d = sba.triangulate_points(correspondences, poses, poses['camera_matrix'])
            
            if len(points_3d) > 0:
                # Save as PLY
                ply_path = os.path.join(output_root, 'pointclouds', f'semantic_{condition_key}.ply')
                os.makedirs(os.path.dirname(ply_path), exist_ok=True)
                
                # Simple PLY writer
                with open(ply_path, 'w') as f:
                    f.write("ply\n")
                    f.write("format ascii 1.0\n")
                    f.write(f"element vertex {len(points_3d)}\n")
                    f.write("property float x\n")
                    f.write("property float y\n")
                    f.write("property float z\n")
                    f.write("end_header\n")
                    for p in points_3d:
                        f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
                
                logger.info(f"Saved {len(points_3d)} 3D points → {ply_path}")
        
        logger.info(f"Registered {poses['num_registered']} images for {condition_key}")
    
    logger.info("\nSemantic bundle adjustment complete.")


if __name__ == '__main__':
    main()
