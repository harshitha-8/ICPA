"""
Feature Correspondence Comparison — ICPA Cotton Boll Pipeline

Head-to-head comparison of DINOv2 semantic matching vs
SuperPoint+SuperGlue RGB matching on cotton boll surfaces.
"""

import os
import sys
import json
import glob
import logging
import random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_dinov2_features(feature_path: str) -> np.ndarray:
    """Load precomputed DINOv2 features."""
    data = np.load(feature_path)
    return data['patch_features'].astype(np.float32)


def compute_dinov2_matches(
    features_a: np.ndarray,
    features_b: np.ndarray,
    threshold: float = 0.7,
    max_matches: int = 2000
) -> Dict:
    """
    Compute mutual nearest neighbor correspondences in DINOv2 embedding space.
    """
    h1, w1, d = features_a.shape
    h2, w2, _ = features_b.shape
    
    fa = features_a.reshape(-1, d)
    fb = features_b.reshape(-1, d)
    
    # L2 normalize
    fa_norm = fa / (np.linalg.norm(fa, axis=1, keepdims=True) + 1e-8)
    fb_norm = fb / (np.linalg.norm(fb, axis=1, keepdims=True) + 1e-8)
    
    # Cosine similarity
    sim = fa_norm @ fb_norm.T
    
    # Mutual nearest neighbors
    nn_a2b = np.argmax(sim, axis=1)
    nn_b2a = np.argmax(sim, axis=0)
    
    mutual = np.array([nn_b2a[nn_a2b[i]] == i for i in range(len(nn_a2b))])
    scores = np.array([sim[i, nn_a2b[i]] for i in range(len(nn_a2b))])
    
    valid = mutual & (scores >= threshold)
    valid_idx = np.where(valid)[0]
    
    if len(valid_idx) > max_matches:
        top_k = np.argsort(-scores[valid_idx])[:max_matches]
        valid_idx = valid_idx[top_k]
    
    matches = []
    for idx in valid_idx:
        y1, x1 = divmod(int(idx), w1)
        match_idx = nn_a2b[idx]
        y2, x2 = divmod(int(match_idx), w2)
        matches.append({
            'pt_a': [int(x1), int(y1)],
            'pt_b': [int(x2), int(y2)],
            'score': float(scores[idx])
        })
    
    return {
        'method': 'dinov2',
        'num_matches': len(matches),
        'mean_score': float(np.mean(scores[valid])) if valid.any() else 0,
        'inlier_ratio': float(valid.sum() / len(valid)),
        'matches': matches
    }


def compute_superpoint_matches(
    image_a: np.ndarray,
    image_b: np.ndarray,
    max_matches: int = 2000
) -> Dict:
    """
    Compute correspondences using SuperPoint + SuperGlue (or OpenCV ORB fallback).
    """
    import cv2
    
    # Convert to grayscale
    if len(image_a.shape) == 3:
        gray_a = cv2.cvtColor(image_a, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(image_b, cv2.COLOR_RGB2GRAY)
    else:
        gray_a, gray_b = image_a, image_b
    
    # Try SuperPoint via OpenCV SIFT as accessible baseline
    # (SuperPoint+SuperGlue requires separate installation)
    try:
        sift = cv2.SIFT_create(nfeatures=max_matches)
        kp_a, desc_a = sift.detectAndCompute(gray_a, None)
        kp_b, desc_b = sift.detectAndCompute(gray_b, None)
        
        if desc_a is None or desc_b is None or len(kp_a) < 10 or len(kp_b) < 10:
            return {
                'method': 'sift',
                'num_matches': 0,
                'mean_score': 0.0,
                'inlier_ratio': 0.0,
                'matches': []
            }
        
        # BFMatcher with ratio test
        bf = cv2.BFMatcher()
        raw_matches = bf.knnMatch(desc_a, desc_b, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for m_pair in raw_matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 4:
            return {
                'method': 'sift',
                'num_matches': len(good_matches),
                'mean_score': 0.0,
                'inlier_ratio': 0.0,
                'matches': []
            }
        
        # RANSAC geometric verification
        src_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if mask is None:
            inlier_ratio = 0.0
            inlier_count = 0
        else:
            inlier_count = int(mask.sum())
            inlier_ratio = inlier_count / len(good_matches)
        
        matches = []
        for i, m in enumerate(good_matches):
            matches.append({
                'pt_a': [int(kp_a[m.queryIdx].pt[0]), int(kp_a[m.queryIdx].pt[1])],
                'pt_b': [int(kp_b[m.trainIdx].pt[0]), int(kp_b[m.trainIdx].pt[1])],
                'score': float(1.0 - m.distance / 500.0),
                'inlier': bool(mask[i][0]) if mask is not None else False
            })
        
        return {
            'method': 'sift',
            'num_matches': len(good_matches),
            'num_inliers': inlier_count,
            'mean_score': float(np.mean([m['score'] for m in matches])),
            'inlier_ratio': inlier_ratio,
            'matches': matches
        }
        
    except Exception as e:
        logger.error(f"SIFT matching failed: {e}")
        return {
            'method': 'sift',
            'num_matches': 0,
            'mean_score': 0.0,
            'inlier_ratio': 0.0,
            'matches': [],
            'error': str(e)
        }


def compute_correspondence_stability_index(
    feature_dir: str,
    image_paths: List[str],
    k_threshold: int = 3,
    num_pairs: int = 100,
    threshold: float = 0.7
) -> Dict:
    """
    Compute Correspondence Stability Index (CSI): fraction of matches
    that persist across ≥k views.
    """
    # Sample random triplets of overlapping images
    n = len(image_paths)
    if n < k_threshold:
        return {'csi': 0.0, 'error': f'Need ≥{k_threshold} images'}
    
    # Use consecutive images as likely overlapping
    stability_scores = []
    
    for i in range(0, min(n - k_threshold, num_pairs)):
        group = image_paths[i:i + k_threshold]
        group_features = []
        
        for img_path in group:
            stem = Path(img_path).stem
            feat_path = os.path.join(feature_dir, f"{stem}_dinov2.npz")
            if os.path.exists(feat_path):
                group_features.append(load_dinov2_features(feat_path))
        
        if len(group_features) < k_threshold:
            continue
        
        # Check if matches from first→second also match in first→third
        result_01 = compute_dinov2_matches(group_features[0], group_features[1], threshold)
        result_02 = compute_dinov2_matches(group_features[0], group_features[2], threshold)
        
        if result_01['num_matches'] == 0:
            stability_scores.append(0.0)
            continue
        
        # For each match in 0→1, check if the same source point has a match in 0→2
        pts_in_01 = set(tuple(m['pt_a']) for m in result_01['matches'])
        pts_in_02 = set(tuple(m['pt_a']) for m in result_02['matches'])
        
        persistent = pts_in_01 & pts_in_02
        stability = len(persistent) / max(len(pts_in_01), 1)
        stability_scores.append(stability)
    
    return {
        'csi': float(np.mean(stability_scores)) if stability_scores else 0.0,
        'csi_std': float(np.std(stability_scores)) if stability_scores else 0.0,
        'num_groups_tested': len(stability_scores),
        'k_threshold': k_threshold
    }


def run_comparison(
    dataset_config: Dict,
    pipeline_config: Dict,
    num_pairs: int = 100,
    seed: int = 42
) -> Dict:
    """
    Run full comparison between DINOv2 and SIFT/SuperPoint matching.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    dataset_root = dataset_config['dataset']['root']
    output_root = dataset_config['output']['root']
    target_res = dataset_config['preprocessing']['target_resolution']
    threshold = pipeline_config['feature_extraction']['matching']['threshold']
    
    results = {}
    
    for condition_key in ['pre_defoliation', 'post_defoliation']:
        condition = dataset_config['dataset'][condition_key]
        
        # Collect images
        images = []
        for folder in condition['folders']:
            folder_path = os.path.join(dataset_root, folder)
            if os.path.exists(folder_path):
                imgs = sorted(glob.glob(os.path.join(folder_path, '*.JPG')))
                imgs = [p for p in imgs if not os.path.basename(p).startswith('._')]
                images.extend(imgs)
        
        feature_dir = os.path.join(output_root, 'features', condition_key)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Comparing methods on {condition_key} ({len(images)} images)")
        logger.info(f"{'='*60}")
        
        # Sample pairs of consecutive images (likely overlapping)
        pairs = []
        for i in range(0, min(len(images) - 1, num_pairs)):
            pairs.append((images[i], images[i + 1]))
        
        dinov2_results = []
        sift_results = []
        
        for img_a_path, img_b_path in tqdm(pairs, desc=f"Matching {condition_key}"):
            stem_a = Path(img_a_path).stem
            stem_b = Path(img_b_path).stem
            
            # DINOv2 matching
            feat_a_path = os.path.join(feature_dir, f"{stem_a}_dinov2.npz")
            feat_b_path = os.path.join(feature_dir, f"{stem_b}_dinov2.npz")
            
            if os.path.exists(feat_a_path) and os.path.exists(feat_b_path):
                fa = load_dinov2_features(feat_a_path)
                fb = load_dinov2_features(feat_b_path)
                dino_result = compute_dinov2_matches(fa, fb, threshold)
                dinov2_results.append(dino_result)
            
            # SIFT matching
            try:
                import cv2
                img_a = cv2.imread(img_a_path)
                img_b = cv2.imread(img_b_path)
                
                if img_a is not None and img_b is not None:
                    # Resize for fair comparison
                    h, w = img_a.shape[:2]
                    scale = min(target_res / max(h, w), 1.0)
                    if scale < 1.0:
                        img_a = cv2.resize(img_a, None, fx=scale, fy=scale)
                        img_b = cv2.resize(img_b, None, fx=scale, fy=scale)
                    
                    sift_result = compute_superpoint_matches(img_a, img_b)
                    sift_results.append(sift_result)
            except ImportError:
                pass
        
        # Aggregate results
        condition_results = {
            'condition': condition_key,
            'num_pairs': len(pairs),
        }
        
        if dinov2_results:
            condition_results['dinov2'] = {
                'mean_num_matches': float(np.mean([r['num_matches'] for r in dinov2_results])),
                'mean_inlier_ratio': float(np.mean([r['inlier_ratio'] for r in dinov2_results])),
                'mean_score': float(np.mean([r['mean_score'] for r in dinov2_results])),
                'std_num_matches': float(np.std([r['num_matches'] for r in dinov2_results])),
            }
        
        if sift_results:
            condition_results['sift'] = {
                'mean_num_matches': float(np.mean([r['num_matches'] for r in sift_results])),
                'mean_inlier_ratio': float(np.mean([r['inlier_ratio'] for r in sift_results])),
                'mean_score': float(np.mean([r['mean_score'] for r in sift_results])),
                'std_num_matches': float(np.std([r['num_matches'] for r in sift_results])),
            }
        
        # CSI
        csi = compute_correspondence_stability_index(
            feature_dir, images, k_threshold=3, num_pairs=50, threshold=threshold
        )
        condition_results['csi'] = csi
        
        results[condition_key] = condition_results
        
        logger.info(f"\nDINOv2: {condition_results.get('dinov2', 'N/A')}")
        logger.info(f"SIFT:   {condition_results.get('sift', 'N/A')}")
        logger.info(f"CSI:    {csi}")
    
    return results


def main():
    repo_root = Path(__file__).parent.parent.parent
    
    with open(repo_root / 'configs' / 'pipeline_config.yaml') as f:
        pipeline_config = yaml.safe_load(f)
    with open(repo_root / 'configs' / 'dataset_config.yaml') as f:
        dataset_config = yaml.safe_load(f)
    
    results = run_comparison(
        dataset_config, pipeline_config,
        num_pairs=100,
        seed=pipeline_config['experiment']['seed']
    )
    
    # Save results
    output_path = os.path.join(
        dataset_config['output']['root'], 'metrics', 'feature_comparison.json'
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
