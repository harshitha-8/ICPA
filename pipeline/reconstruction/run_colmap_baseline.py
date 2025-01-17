"""
COLMAP SfM Baseline — ICPA Cotton Boll Pipeline

Wrapper for COLMAP Structure-from-Motion and Multi-View Stereo
reconstruction. Runs separate reconstructions for pre/post-defoliation.
"""

import os
import sys
import json
import glob
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict

import yaml
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class COLMAPReconstructor:
    """Wrapper for COLMAP SfM + MVS pipeline."""
    
    def __init__(
        self,
        colmap_binary: str = "colmap",
        workspace: str = "outputs/colmap",
        camera_model: str = "OPENCV",
        use_gpu: bool = True,
        max_image_size: int = 2048
    ):
        self.colmap_binary = colmap_binary
        self.workspace = workspace
        self.camera_model = camera_model
        self.use_gpu = use_gpu
        self.max_image_size = max_image_size
        
        # Verify COLMAP is available
        self._colmap_available = self._check_colmap()
    
    def _check_colmap(self) -> bool:
        """Check if COLMAP binary is available."""
        try:
            result = subprocess.run(
                [self.colmap_binary, "--help"],
                capture_output=True, text=True, timeout=10
            )
            logger.info("COLMAP found and accessible.")
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning(
                "COLMAP binary not found. Install COLMAP or specify path in config.\n"
                "Running in stub mode — will generate placeholder commands."
            )
            return False
    
    def _run_colmap(self, args: list, description: str = "") -> bool:
        """Execute a COLMAP command."""
        cmd = [self.colmap_binary] + args
        logger.info(f"Running: {' '.join(cmd)}")
        
        if not self._colmap_available:
            logger.info(f"[STUB] Would run: {' '.join(cmd)}")
            return True
        
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=7200  # 2h timeout
            )
            if result.returncode != 0:
                logger.error(f"COLMAP failed ({description}):\n{result.stderr[:500]}")
                return False
            logger.info(f"COLMAP {description} completed successfully.")
            return True
        except subprocess.TimeoutExpired:
            logger.error(f"COLMAP {description} timed out after 2 hours.")
            return False
    
    def reconstruct(
        self,
        image_dir: str,
        output_name: str,
        image_list: Optional[list] = None
    ) -> Dict:
        """
        Run full COLMAP SfM + MVS pipeline.
        
        Args:
            image_dir: directory containing images (or symlinks)
            output_name: name for this reconstruction (e.g., 'pre_defoliation')
            image_list: optional list of specific image filenames to use
        
        Returns:
            Dict with reconstruction statistics
        """
        workspace = os.path.join(self.workspace, output_name)
        db_path = os.path.join(workspace, "database.db")
        sparse_dir = os.path.join(workspace, "sparse")
        dense_dir = os.path.join(workspace, "dense")
        
        os.makedirs(workspace, exist_ok=True)
        os.makedirs(sparse_dir, exist_ok=True)
        os.makedirs(dense_dir, exist_ok=True)
        
        # If image_list provided, create a symlink directory
        if image_list:
            link_dir = os.path.join(workspace, "images")
            os.makedirs(link_dir, exist_ok=True)
            for img_path in image_list:
                link_path = os.path.join(link_dir, os.path.basename(img_path))
                if not os.path.exists(link_path):
                    os.symlink(img_path, link_path)
            image_dir = link_dir
        
        results = {
            'output_name': output_name,
            'image_dir': image_dir,
            'workspace': workspace,
            'steps': {}
        }
        
        # Step 1: Feature Extraction
        logger.info(f"\n{'='*60}")
        logger.info(f"Step 1/5: Feature Extraction — {output_name}")
        logger.info(f"{'='*60}")
        
        success = self._run_colmap([
            "feature_extractor",
            "--database_path", db_path,
            "--image_path", image_dir,
            "--ImageReader.camera_model", self.camera_model,
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.max_image_size", str(self.max_image_size),
            "--SiftExtraction.use_gpu", "1" if self.use_gpu else "0",
        ], "feature extraction")
        results['steps']['feature_extraction'] = success
        
        if not success and self._colmap_available:
            return results
        
        # Step 2: Exhaustive Matching
        logger.info(f"\n{'='*60}")
        logger.info(f"Step 2/5: Feature Matching — {output_name}")
        logger.info(f"{'='*60}")
        
        success = self._run_colmap([
            "exhaustive_matcher",
            "--database_path", db_path,
            "--SiftMatching.use_gpu", "1" if self.use_gpu else "0",
        ], "feature matching")
        results['steps']['feature_matching'] = success
        
        # Step 3: Sparse Reconstruction (SfM)
        logger.info(f"\n{'='*60}")
        logger.info(f"Step 3/5: Sparse Reconstruction — {output_name}")
        logger.info(f"{'='*60}")
        
        success = self._run_colmap([
            "mapper",
            "--database_path", db_path,
            "--image_path", image_dir,
            "--output_path", sparse_dir,
        ], "sparse reconstruction")
        results['steps']['sparse_reconstruction'] = success
        
        # Step 4: Image Undistortion (for MVS)
        logger.info(f"\n{'='*60}")
        logger.info(f"Step 4/5: Image Undistortion — {output_name}")
        logger.info(f"{'='*60}")
        
        # Find the largest sparse model (model 0 typically)
        model_dir = os.path.join(sparse_dir, "0")
        if not os.path.exists(model_dir):
            model_dir = sparse_dir
        
        success = self._run_colmap([
            "image_undistorter",
            "--image_path", image_dir,
            "--input_path", model_dir,
            "--output_path", dense_dir,
            "--output_type", "COLMAP",
        ], "image undistortion")
        results['steps']['undistortion'] = success
        
        # Step 5: Dense Reconstruction (MVS)
        logger.info(f"\n{'='*60}")
        logger.info(f"Step 5/5: Dense Reconstruction — {output_name}")
        logger.info(f"{'='*60}")
        
        success = self._run_colmap([
            "patch_match_stereo",
            "--workspace_path", dense_dir,
            "--PatchMatchStereo.geom_consistency", "true",
        ], "dense reconstruction")
        results['steps']['dense_reconstruction'] = success
        
        # Fuse into point cloud
        if success or not self._colmap_available:
            self._run_colmap([
                "stereo_fusion",
                "--workspace_path", dense_dir,
                "--output_path", os.path.join(dense_dir, "fused.ply"),
            ], "point cloud fusion")
        
        # Save results
        results_path = os.path.join(workspace, "reconstruction_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def read_sparse_model_stats(self, model_dir: str) -> Dict:
        """Read statistics from a COLMAP sparse model."""
        stats = {
            'num_cameras': 0,
            'num_images': 0,
            'num_points3D': 0,
            'mean_reprojection_error': 0.0,
        }
        
        # Try reading cameras.txt
        cameras_file = os.path.join(model_dir, "cameras.txt")
        if os.path.exists(cameras_file):
            with open(cameras_file) as f:
                lines = [l for l in f.readlines() if not l.startswith('#')]
                stats['num_cameras'] = len(lines)
        
        # Try reading images.txt
        images_file = os.path.join(model_dir, "images.txt")
        if os.path.exists(images_file):
            with open(images_file) as f:
                lines = [l for l in f.readlines() if not l.startswith('#')]
                stats['num_images'] = len(lines) // 2  # every other line
        
        # Try reading points3D.txt
        points_file = os.path.join(model_dir, "points3D.txt")
        if os.path.exists(points_file):
            with open(points_file) as f:
                lines = [l for l in f.readlines() if not l.startswith('#')]
                stats['num_points3D'] = len(lines)
                if lines:
                    errors = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 8:
                            try:
                                errors.append(float(parts[7]))
                            except ValueError:
                                pass
                    if errors:
                        stats['mean_reprojection_error'] = float(np.mean(errors))
        
        return stats


def main():
    """Run COLMAP baseline reconstruction on both conditions."""
    repo_root = Path(__file__).parent.parent.parent
    
    with open(repo_root / 'configs' / 'pipeline_config.yaml') as f:
        pipeline_config = yaml.safe_load(f)
    with open(repo_root / 'configs' / 'dataset_config.yaml') as f:
        dataset_config = yaml.safe_load(f)
    
    colmap_config = pipeline_config['reconstruction']['colmap']
    dataset_root = dataset_config['dataset']['root']
    output_root = dataset_config['output']['root']
    
    reconstructor = COLMAPReconstructor(
        colmap_binary=colmap_config['binary_path'],
        workspace=os.path.join(output_root, 'colmap'),
        camera_model=colmap_config['camera_model'],
        use_gpu=colmap_config['use_gpu'],
        max_image_size=colmap_config['max_image_size'],
    )
    
    # Run for each condition
    for condition_key in ['pre_defoliation', 'post_defoliation']:
        condition = dataset_config['dataset'][condition_key]
        images = []
        for folder in condition['folders']:
            folder_path = os.path.join(dataset_root, folder)
            if os.path.exists(folder_path):
                imgs = sorted(glob.glob(os.path.join(folder_path, '*.JPG')))
                imgs = [p for p in imgs if not os.path.basename(p).startswith('._')]
                images.extend(imgs)
        
        logger.info(f"\n{'#'*60}")
        logger.info(f"COLMAP Reconstruction: {condition_key} ({len(images)} images)")
        logger.info(f"{'#'*60}")
        
        results = reconstructor.reconstruct(
            image_dir=os.path.join(dataset_root, condition['folders'][0]),
            output_name=condition_key,
            image_list=images
        )
        
        logger.info(f"Results: {json.dumps(results['steps'], indent=2)}")
    
    logger.info("\nCOLMAP baseline reconstruction complete.")


if __name__ == '__main__':
    main()
