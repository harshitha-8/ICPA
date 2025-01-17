"""
SAM 2 Segmentation — ICPA Cotton Boll Pipeline

Zero-shot instance segmentation of cotton bolls using SAM 2 with
automatic mask generation and DINOv2-guided prompt refinement.
"""

import os
import sys
import json
import glob
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class CottonBollSegmenter:
    """
    Zero-shot cotton boll segmentation using SAM 2.
    
    Combines automatic mask generation with DINOv2-guided semantic
    prompting for improved boll detection under occlusion.
    """
    
    def __init__(
        self,
        model_cfg: str = "sam2_hiera_large",
        device: str = "mps",
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.86,
        stability_score_thresh: float = 0.92,
        min_mask_region_area: int = 100
    ):
        self.device = device
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.min_mask_region_area = min_mask_region_area
        
        logger.info(f"Initializing SAM 2 ({model_cfg}) on {device}...")
        
        # NOTE: SAM 2 import and initialization
        # The exact import path depends on the installed version.
        # Users should install via: pip install segment-anything-2
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            
            self.sam2_model = build_sam2(model_cfg, device=device)
            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=self.sam2_model,
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                min_mask_region_area=min_mask_region_area,
            )
            self._sam2_available = True
            logger.info("SAM 2 loaded successfully.")
        except ImportError:
            logger.warning(
                "SAM 2 not installed. Install via: pip install segment-anything-2\n"
                "Running in stub mode for pipeline development."
            )
            self._sam2_available = False
    
    def segment_image(self, image: np.ndarray) -> List[Dict]:
        """
        Generate automatic masks for a single image.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
        
        Returns:
            List of mask dictionaries with keys:
                'segmentation': binary mask (H, W)
                'area': number of pixels
                'bbox': [x, y, w, h]
                'predicted_iou': float
                'stability_score': float
                'crop_box': [x, y, w, h]
        """
        if not self._sam2_available:
            return self._generate_stub_masks(image)
        
        masks = self.mask_generator.generate(image)
        return masks
    
    def _generate_stub_masks(self, image: np.ndarray) -> List[Dict]:
        """Generate placeholder masks for development without SAM 2."""
        H, W = image.shape[:2]
        # Return empty list — actual masks will be generated when SAM 2 is installed
        logger.debug(f"Stub mode: returning empty masks for image ({H}x{W})")
        return []
    
    def classify_masks(
        self,
        masks: List[Dict],
        image: np.ndarray,
        dinov2_features: Optional[np.ndarray] = None,
        boll_prototype: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Classify each mask as boll, leaf, stem, soil, or other using
        DINOv2 feature similarity to prototype embeddings.
        
        Args:
            masks: list of SAM 2 mask dicts
            image: original RGB image
            dinov2_features: (h, w, d) DINOv2 patch features (optional)
            boll_prototype: (d,) mean embedding of cotton boll patches (optional)
        
        Returns:
            Annotated masks with 'label' and 'boll_score' fields
        """
        if dinov2_features is None or boll_prototype is None:
            # Without DINOv2 features, use simple heuristics
            for mask in masks:
                mask['label'] = 'unknown'
                mask['boll_score'] = 0.0
            return masks
        
        h, w, d = dinov2_features.shape
        H, W = image.shape[:2]
        
        # Normalize prototype
        boll_proto_norm = boll_prototype / (np.linalg.norm(boll_prototype) + 1e-8)
        
        for mask in masks:
            seg = mask['segmentation']  # (H, W) boolean
            
            # Downsample mask to feature grid
            seg_small = np.array(
                Image.fromarray(seg.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
            ) > 127
            
            if seg_small.sum() == 0:
                mask['label'] = 'other'
                mask['boll_score'] = 0.0
                continue
            
            # Mean feature within mask
            masked_features = dinov2_features[seg_small]
            mean_feat = masked_features.mean(axis=0).astype(np.float32)
            mean_feat_norm = mean_feat / (np.linalg.norm(mean_feat) + 1e-8)
            
            # Cosine similarity to boll prototype
            boll_score = float(np.dot(mean_feat_norm, boll_proto_norm))
            mask['boll_score'] = boll_score
            
            # Classification thresholds (to be calibrated)
            if boll_score > 0.7:
                mask['label'] = 'boll'
            elif boll_score > 0.4:
                mask['label'] = 'leaf'
            else:
                mask['label'] = 'other'
        
        return masks
    
    def compute_mask_stability(self, masks: List[Dict]) -> Dict:
        """
        Compute aggregated mask stability metrics.
        
        Returns:
            Dict with mean_iou, mean_stability, boll_count, etc.
        """
        if not masks:
            return {'boll_count': 0, 'total_masks': 0}
        
        boll_masks = [m for m in masks if m.get('label') == 'boll']
        
        return {
            'total_masks': len(masks),
            'boll_count': len(boll_masks),
            'mean_predicted_iou': float(np.mean([m['predicted_iou'] for m in masks])),
            'mean_stability_score': float(np.mean([m['stability_score'] for m in masks])),
            'mean_boll_area': float(np.mean([m['area'] for m in boll_masks])) if boll_masks else 0.0,
            'mean_boll_score': float(np.mean([m.get('boll_score', 0) for m in boll_masks])) if boll_masks else 0.0,
        }
    
    def track_instances(
        self,
        masks_t0: List[Dict],
        masks_t1: List[Dict],
        features_t0: Optional[np.ndarray] = None,
        features_t1: Optional[np.ndarray] = None,
        iou_weight: float = 0.5
    ) -> List[Tuple[int, int, float]]:
        """
        Associate boll instances across two consecutive frames using
        IoU and feature similarity.
        
        Returns:
            List of (idx_t0, idx_t1, match_score) tuples
        """
        bolls_t0 = [m for m in masks_t0 if m.get('label') == 'boll']
        bolls_t1 = [m for m in masks_t1 if m.get('label') == 'boll']
        
        if not bolls_t0 or not bolls_t1:
            return []
        
        n0, n1 = len(bolls_t0), len(bolls_t1)
        cost_matrix = np.zeros((n0, n1))
        
        for i, m0 in enumerate(bolls_t0):
            for j, m1 in enumerate(bolls_t1):
                # IoU
                intersection = np.logical_and(m0['segmentation'], m1['segmentation']).sum()
                union = np.logical_or(m0['segmentation'], m1['segmentation']).sum()
                iou = intersection / (union + 1e-8)
                
                # Feature similarity (if available)
                feat_sim = 0.0
                if features_t0 is not None and features_t1 is not None:
                    # Simplified: use boll_score correlation
                    feat_sim = 1.0 - abs(m0.get('boll_score', 0) - m1.get('boll_score', 0))
                
                cost_matrix[i, j] = iou_weight * iou + (1 - iou_weight) * feat_sim
        
        # Hungarian assignment
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(-cost_matrix)  # maximize
            
            matches = []
            for r, c in zip(row_ind, col_ind):
                score = cost_matrix[r, c]
                if score > 0.1:  # minimum match threshold
                    matches.append((int(r), int(c), float(score)))
            
            return matches
        except ImportError:
            logger.warning("scipy not available. Skipping Hungarian assignment.")
            return []


def process_condition(
    segmenter: CottonBollSegmenter,
    image_paths: List[str],
    output_dir: str,
    feature_dir: Optional[str] = None,
    boll_prototype: Optional[np.ndarray] = None
) -> Dict:
    """Process all images for a single condition (pre or post defoliation)."""
    os.makedirs(output_dir, exist_ok=True)
    
    all_metrics = []
    
    for img_path in tqdm(image_paths, desc="Segmenting"):
        filename = Path(img_path).stem
        output_path = os.path.join(output_dir, f"{filename}_masks.npz")
        
        # Skip if already computed
        if os.path.exists(output_path):
            continue
        
        try:
            image = np.array(Image.open(img_path).convert('RGB'))
            
            # Load DINOv2 features if available
            dinov2_features = None
            if feature_dir:
                feat_path = os.path.join(feature_dir, f"{filename}_dinov2.npz")
                if os.path.exists(feat_path):
                    feat_data = np.load(feat_path)
                    dinov2_features = feat_data['patch_features']
            
            # Generate masks
            masks = segmenter.segment_image(image)
            
            # Classify masks
            masks = segmenter.classify_masks(masks, image, dinov2_features, boll_prototype)
            
            # Compute metrics
            metrics = segmenter.compute_mask_stability(masks)
            metrics['filename'] = filename
            all_metrics.append(metrics)
            
            # Save masks (just segmentation arrays and metadata, not full dict)
            mask_data = {
                'num_masks': len(masks),
                'boll_count': metrics['boll_count'],
            }
            
            # Save binary masks as compressed array
            if masks:
                seg_stack = np.stack([m['segmentation'] for m in masks])
                labels = [m.get('label', 'unknown') for m in masks]
                scores = [m.get('boll_score', 0.0) for m in masks]
                
                np.savez_compressed(
                    output_path,
                    masks=seg_stack,
                    labels=np.array(labels),
                    boll_scores=np.array(scores, dtype=np.float32),
                    metadata=json.dumps(mask_data)
                )
            
        except Exception as e:
            logger.error(f"Failed to process {img_path}: {e}")
            continue
    
    # Save aggregated metrics
    metrics_path = os.path.join(output_dir, 'segmentation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    return {
        'total_images': len(image_paths),
        'processed': len(all_metrics),
        'mean_boll_count': float(np.mean([m['boll_count'] for m in all_metrics])) if all_metrics else 0,
    }


def main():
    """Main segmentation pipeline."""
    repo_root = Path(__file__).parent.parent.parent
    
    with open(repo_root / 'configs' / 'pipeline_config.yaml') as f:
        pipeline_config = yaml.safe_load(f)
    with open(repo_root / 'configs' / 'dataset_config.yaml') as f:
        dataset_config = yaml.safe_load(f)
    
    seg_config = pipeline_config['segmentation']
    dataset_root = dataset_config['dataset']['root']
    output_root = dataset_config['output']['root']
    
    # Initialize segmenter
    segmenter = CottonBollSegmenter(
        model_cfg=seg_config['model'],
        device=seg_config['device'],
        points_per_side=seg_config['points_per_side'],
        pred_iou_thresh=seg_config['pred_iou_thresh'],
        stability_score_thresh=seg_config['stability_score_thresh'],
        min_mask_region_area=seg_config['min_mask_region_area'],
    )
    
    # Process each condition
    for condition_key in ['pre_defoliation', 'post_defoliation']:
        condition = dataset_config['dataset'][condition_key]
        images = []
        for folder in condition['folders']:
            folder_path = os.path.join(dataset_root, folder)
            if os.path.exists(folder_path):
                imgs = sorted(glob.glob(os.path.join(folder_path, '*.JPG')))
                imgs = [p for p in imgs if not os.path.basename(p).startswith('._')]
                images.extend(imgs)
        
        output_dir = os.path.join(output_root, 'masks', condition_key)
        feature_dir = os.path.join(output_root, 'features', condition_key)
        
        logger.info(f"\nProcessing {condition_key}: {len(images)} images")
        results = process_condition(
            segmenter, images, output_dir,
            feature_dir=feature_dir if os.path.exists(feature_dir) else None
        )
        logger.info(f"Results: {json.dumps(results, indent=2)}")
    
    logger.info("\nSegmentation complete.")


if __name__ == '__main__':
    main()
