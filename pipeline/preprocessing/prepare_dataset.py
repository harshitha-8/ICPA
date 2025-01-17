"""
Dataset Preparation — ICPA Cotton Boll Semantic 3D Reconstruction

Organizes raw UAV imagery into pre/post-defoliation splits,
extracts EXIF metadata, validates images, and creates manifest files.
"""

import os
import sys
import json
import glob
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import yaml
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    """Load dataset configuration from YAML."""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'configs', 'dataset_config.yaml'
        )
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def collect_images(folder: str, extensions: list = None) -> list:
    """Collect all valid image paths from a folder, excluding macOS resource forks."""
    if extensions is None:
        extensions = ['.JPG', '.jpg', '.jpeg', '.png']
    
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(folder, f'*{ext}')))
    
    # Filter out macOS resource fork files
    images = [p for p in images if not os.path.basename(p).startswith('._')]
    images.sort()
    return images


def extract_basic_metadata(image_path: str) -> dict:
    """Extract basic image metadata without EXIF dependency."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            mode = img.mode
    except Exception as e:
        logger.warning(f"Cannot open {image_path}: {e}")
        return None
    
    stat = os.stat(image_path)
    filename = os.path.basename(image_path)
    
    # Parse DJI filename for timestamp: DJI_YYYYMMDDHHMMSS_NNNN_D.JPG
    timestamp = None
    seq_num = None
    if filename.startswith('DJI_'):
        parts = filename.replace('.JPG', '').replace('.jpg', '').split('_')
        if len(parts) >= 3:
            try:
                timestamp = datetime.strptime(parts[1], '%Y%m%d%H%M%S').isoformat()
                seq_num = int(parts[2])
            except (ValueError, IndexError):
                pass
    
    return {
        'filename': filename,
        'path': image_path,
        'width': width,
        'height': height,
        'mode': mode,
        'size_bytes': stat.st_size,
        'timestamp': timestamp,
        'sequence_number': seq_num
    }


def validate_image(image_path: str) -> bool:
    """Validate that an image can be opened and decoded."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def create_manifest(images: list, condition: str, output_path: str) -> dict:
    """Create a manifest file with metadata for a set of images."""
    manifest = {
        'condition': condition,
        'created': datetime.now().isoformat(),
        'total_images': 0,
        'valid_images': 0,
        'invalid_images': [],
        'images': []
    }
    
    logger.info(f"Processing {len(images)} images for {condition}...")
    
    for img_path in tqdm(images, desc=f"Scanning {condition}"):
        meta = extract_basic_metadata(img_path)
        if meta is None:
            manifest['invalid_images'].append(img_path)
            continue
        
        manifest['images'].append(meta)
        manifest['total_images'] += 1
    
    manifest['valid_images'] = manifest['total_images']
    
    # Summary statistics
    if manifest['images']:
        sizes = [m['size_bytes'] for m in manifest['images']]
        widths = [m['width'] for m in manifest['images']]
        heights = [m['height'] for m in manifest['images']]
        
        manifest['summary'] = {
            'mean_size_mb': sum(sizes) / len(sizes) / 1e6,
            'total_size_gb': sum(sizes) / 1e9,
            'resolution': f"{widths[0]}x{heights[0]}",
            'unique_resolutions': list(set(f"{w}x{h}" for w, h in zip(widths, heights))),
            'timestamp_range': {
                'earliest': min((m['timestamp'] for m in manifest['images'] if m['timestamp']), default=None),
                'latest': max((m['timestamp'] for m in manifest['images'] if m['timestamp']), default=None)
            }
        }
    
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Manifest saved: {output_path} ({manifest['valid_images']} valid images)")
    return manifest


def prepare_dataset(config_path: str = None):
    """Main dataset preparation pipeline."""
    config = load_config(config_path)
    dataset_root = config['dataset']['root']
    output_root = config['output']['root']
    
    os.makedirs(output_root, exist_ok=True)
    
    # Collect pre-defoliation images
    pre_images = []
    for folder in config['dataset']['pre_defoliation']['folders']:
        folder_path = os.path.join(dataset_root, folder)
        if os.path.exists(folder_path):
            imgs = collect_images(folder_path)
            pre_images.extend(imgs)
            logger.info(f"Found {len(imgs)} images in {folder}")
        else:
            logger.warning(f"Folder not found: {folder_path}")
    
    # Collect post-defoliation images
    post_images = []
    for folder in config['dataset']['post_defoliation']['folders']:
        folder_path = os.path.join(dataset_root, folder)
        if os.path.exists(folder_path):
            imgs = collect_images(folder_path)
            post_images.extend(imgs)
            logger.info(f"Found {len(imgs)} images in {folder}")
        else:
            logger.warning(f"Folder not found: {folder_path}")
    
    logger.info(f"\nTotal pre-defoliation images: {len(pre_images)}")
    logger.info(f"Total post-defoliation images: {len(post_images)}")
    logger.info(f"Grand total: {len(pre_images) + len(post_images)}")
    
    # Create manifests
    pre_manifest = create_manifest(
        pre_images, 'pre_defoliation',
        os.path.join(output_root, 'manifest_pre_defoliation.json')
    )
    
    post_manifest = create_manifest(
        post_images, 'post_defoliation',
        os.path.join(output_root, 'manifest_post_defoliation.json')
    )
    
    # Create combined summary
    summary = {
        'dataset': 'ICPA Cotton Boll UAV Imagery',
        'capture_date': config['dataset']['capture_date'],
        'sensor': config['dataset']['sensor'],
        'pre_defoliation': {
            'total_images': pre_manifest['valid_images'],
            'folders': config['dataset']['pre_defoliation']['folders']
        },
        'post_defoliation': {
            'total_images': post_manifest['valid_images'],
            'folders': config['dataset']['post_defoliation']['folders']
        },
        'grand_total': pre_manifest['valid_images'] + post_manifest['valid_images']
    }
    
    summary_path = os.path.join(output_root, 'dataset_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nDataset summary saved: {summary_path}")
    return summary


if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    summary = prepare_dataset(config_path)
    print(json.dumps(summary, indent=2))
