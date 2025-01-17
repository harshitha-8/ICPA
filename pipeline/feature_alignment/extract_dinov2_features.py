"""
DINOv2 Dense Feature Extraction — ICPA Cotton Boll Pipeline

Extracts patch-level DINOv2 embeddings from UAV images for semantic
correspondence matching. Supports ViT-S/14, ViT-B/14, and ViT-L/14 backbones.
"""

import os
import sys
import json
import glob
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# DINOv2 model configurations
DINOV2_CONFIGS = {
    'dinov2_vits14': {'embed_dim': 384, 'patch_size': 14},
    'dinov2_vitb14': {'embed_dim': 768, 'patch_size': 14},
    'dinov2_vitl14': {'embed_dim': 1024, 'patch_size': 14},
}


class DINOv2FeatureExtractor:
    """Extracts dense patch features from images using frozen DINOv2 backbone."""
    
    def __init__(
        self,
        model_name: str = 'dinov2_vitl14',
        device: str = 'mps',
        target_resolution: int = 1024
    ):
        self.model_name = model_name
        self.device = torch.device(device if device != 'mps' or torch.backends.mps.is_available() else 'cpu')
        self.target_resolution = target_resolution
        
        config = DINOV2_CONFIGS[model_name]
        self.embed_dim = config['embed_dim']
        self.patch_size = config['patch_size']
        
        logger.info(f"Loading {model_name} on {self.device}...")
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(target_resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self._make_divisible(target_resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        logger.info(f"Model loaded. Embedding dim: {self.embed_dim}, Patch size: {self.patch_size}")
    
    def _make_divisible(self, size: int) -> int:
        """Round down to nearest multiple of patch_size."""
        return (size // self.patch_size) * self.patch_size
    
    @torch.no_grad()
    def extract_features(self, image: Image.Image) -> dict:
        """
        Extract dense patch features from a single image.
        
        Returns:
            dict with keys:
                'patch_features': np.ndarray of shape (h, w, embed_dim)
                'cls_token': np.ndarray of shape (embed_dim,)
                'patch_grid': tuple (h, w) — spatial dimensions of patch grid
                'image_size': tuple (H, W) — original image size
        """
        orig_size = image.size  # (W, H)
        
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        _, _, H_proc, W_proc = img_tensor.shape
        
        h = H_proc // self.patch_size
        w = W_proc // self.patch_size
        
        # Forward pass to get patch tokens
        output = self.model.forward_features(img_tensor)
        
        # Extract patch tokens (exclude CLS token at position 0)
        patch_tokens = output['x_norm_patchtokens']  # (1, h*w, embed_dim)
        cls_token = output['x_norm_clstoken']  # (1, embed_dim)
        
        # Reshape to spatial grid
        patch_features = patch_tokens.squeeze(0).reshape(h, w, self.embed_dim)
        
        return {
            'patch_features': patch_features.cpu().numpy().astype(np.float16),
            'cls_token': cls_token.squeeze(0).cpu().numpy().astype(np.float32),
            'patch_grid': (h, w),
            'image_size': orig_size,
            'processed_size': (H_proc, W_proc)
        }
    
    def extract_batch(
        self,
        image_paths: List[str],
        output_dir: str,
        batch_size: int = 8
    ) -> List[str]:
        """
        Extract features for a batch of images and save to disk.
        
        Returns:
            List of output file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []
        
        for img_path in tqdm(image_paths, desc="Extracting DINOv2 features"):
            filename = Path(img_path).stem
            output_path = os.path.join(output_dir, f"{filename}_dinov2.npz")
            
            # Skip if already computed
            if os.path.exists(output_path):
                output_paths.append(output_path)
                continue
            
            try:
                image = Image.open(img_path).convert('RGB')
                features = self.extract_features(image)
                
                np.savez_compressed(
                    output_path,
                    patch_features=features['patch_features'],
                    cls_token=features['cls_token'],
                    patch_grid=np.array(features['patch_grid']),
                    image_size=np.array(features['image_size']),
                    processed_size=np.array(features['processed_size'])
                )
                
                output_paths.append(output_path)
                
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")
                continue
        
        logger.info(f"Extracted features for {len(output_paths)} images → {output_dir}")
        return output_paths


def compute_pairwise_similarity(
    features_a: np.ndarray,
    features_b: np.ndarray,
    threshold: float = 0.7
) -> dict:
    """
    Compute dense cosine similarity between two feature maps.
    
    Args:
        features_a: (h1, w1, d) feature map from image A
        features_b: (h2, w2, d) feature map from image B
        threshold: minimum similarity for a valid correspondence
    
    Returns:
        dict with correspondences and similarity statistics
    """
    h1, w1, d = features_a.shape
    h2, w2, d2 = features_b.shape
    assert d == d2, f"Feature dimensions must match: {d} vs {d2}"
    
    # Flatten to (N, d)
    fa = features_a.reshape(-1, d).astype(np.float32)
    fb = features_b.reshape(-1, d).astype(np.float32)
    
    # L2 normalize
    fa = fa / (np.linalg.norm(fa, axis=1, keepdims=True) + 1e-8)
    fb = fb / (np.linalg.norm(fb, axis=1, keepdims=True) + 1e-8)
    
    # Cosine similarity matrix
    sim_matrix = fa @ fb.T  # (N1, N2)
    
    # Mutual nearest neighbors
    nn_a_to_b = np.argmax(sim_matrix, axis=1)  # best match in B for each A
    nn_b_to_a = np.argmax(sim_matrix, axis=0)  # best match in A for each B
    
    # Mutual check
    mutual_mask = np.array([nn_b_to_a[nn_a_to_b[i]] == i for i in range(len(nn_a_to_b))])
    
    # Threshold check
    max_sims = np.array([sim_matrix[i, nn_a_to_b[i]] for i in range(len(nn_a_to_b))])
    valid_mask = mutual_mask & (max_sims >= threshold)
    
    # Build correspondences
    valid_indices = np.where(valid_mask)[0]
    correspondences = []
    for idx in valid_indices:
        y1, x1 = divmod(int(idx), w1)
        match_idx = nn_a_to_b[idx]
        y2, x2 = divmod(int(match_idx), w2)
        correspondences.append({
            'a': (int(x1), int(y1)),
            'b': (int(x2), int(y2)),
            'similarity': float(max_sims[idx])
        })
    
    return {
        'num_correspondences': len(correspondences),
        'correspondences': correspondences,
        'mean_similarity': float(np.mean(max_sims[valid_mask])) if valid_mask.any() else 0.0,
        'median_similarity': float(np.median(max_sims[valid_mask])) if valid_mask.any() else 0.0,
        'inlier_ratio': float(valid_mask.sum() / len(valid_mask)) if len(valid_mask) > 0 else 0.0
    }


def main():
    """Main feature extraction pipeline."""
    # Load configs
    repo_root = Path(__file__).parent.parent.parent
    config_path = repo_root / 'configs' / 'pipeline_config.yaml'
    dataset_config_path = repo_root / 'configs' / 'dataset_config.yaml'
    
    with open(config_path) as f:
        pipeline_config = yaml.safe_load(f)
    with open(dataset_config_path) as f:
        dataset_config = yaml.safe_load(f)
    
    fe_config = pipeline_config['feature_extraction']
    dataset_root = dataset_config['dataset']['root']
    output_root = dataset_config['output']['root']
    
    # Initialize extractor
    extractor = DINOv2FeatureExtractor(
        model_name=fe_config['model'],
        device=fe_config['device'],
        target_resolution=dataset_config['preprocessing']['target_resolution']
    )
    
    # Collect images
    all_images = []
    for condition_key in ['pre_defoliation', 'post_defoliation']:
        condition = dataset_config['dataset'][condition_key]
        for folder in condition['folders']:
            folder_path = os.path.join(dataset_root, folder)
            if os.path.exists(folder_path):
                images = sorted(glob.glob(os.path.join(folder_path, '*.JPG')))
                images = [p for p in images if not os.path.basename(p).startswith('._')]
                all_images.extend([(img, condition_key) for img in images])
    
    logger.info(f"Total images to process: {len(all_images)}")
    
    # Extract features by condition
    for condition in ['pre_defoliation', 'post_defoliation']:
        condition_images = [img for img, cond in all_images if cond == condition]
        output_dir = os.path.join(output_root, 'features', condition)
        
        logger.info(f"\nProcessing {condition}: {len(condition_images)} images")
        extractor.extract_batch(condition_images, output_dir, batch_size=fe_config['batch_size'])
    
    logger.info("\nFeature extraction complete.")


if __name__ == '__main__':
    main()
