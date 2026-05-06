"""
3DGS Pipeline Orchestrator — ICPA Cotton Boll Pipeline

End-to-end runner: COLMAP check → 3DGS train → SPZ export →
Gaussian morphology → web viewer copy.
"""

import os
import sys
import json
import shutil
import logging
import argparse
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def check_colmap_output(colmap_dir: str) -> bool:
    """Check that COLMAP sparse reconstruction exists."""
    bin_dir = os.path.join(colmap_dir, "sparse", "0")
    txt_dir = os.path.join(colmap_dir, "sparse")

    if os.path.exists(os.path.join(bin_dir, "points3D.bin")):
        return True
    if os.path.exists(os.path.join(txt_dir, "points3D.txt")):
        return True

    # Check for dense/fused.ply as alternative
    fused = os.path.join(colmap_dir, "dense", "fused.ply")
    return os.path.exists(fused)


def prepare_web_export(repo_root: str, condition: str, spz_path: str, morphology_path: str):
    """Copy SPZ, morphology JSON, and web viewer to export directory."""
    export_dir = os.path.join(repo_root, "outputs", "web_export")
    os.makedirs(export_dir, exist_ok=True)

    # Copy web viewer files
    viewer_src = os.path.join(repo_root, "pipeline", "visualization", "web_viewer")
    for fname in ["index.html", "viewer.js", "styles.css"]:
        src = os.path.join(viewer_src, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(export_dir, fname))

    # Copy morphology data
    if os.path.exists(morphology_path):
        shutil.copy2(morphology_path, os.path.join(export_dir, f"morphology_{condition}.json"))

    logger.info("Web export ready → %s", export_dir)
    logger.info("Serve with: python -m http.server 8765 --directory %s", export_dir)


def run_pipeline(condition: str, repo_root: str, config: dict, device: str = "mps"):
    """Run the full 3DGS pipeline for one condition."""
    logger.info("=" * 60)
    logger.info("3DGS Pipeline: %s", condition)
    logger.info("=" * 60)

    colmap_dir = os.path.join(repo_root, "outputs", "colmap", condition)
    image_dir = os.path.join(repo_root, "data", condition)
    gs_output = os.path.join(repo_root, "outputs", "gaussian_splats", condition)
    spz_path = os.path.join(repo_root, "outputs", "web_export", f"{condition}.spz")
    morph_path = os.path.join(repo_root, "outputs", "metrics", f"morphology_3dgs_{condition}.json")

    # Step 1: Check COLMAP
    logger.info("[1/4] Checking COLMAP reconstruction …")
    if not check_colmap_output(colmap_dir):
        logger.error("COLMAP output not found at %s", colmap_dir)
        logger.error("Run: python pipeline/reconstruction/run_colmap_baseline.py first")
        return False

    # Step 2: Train 3DGS
    logger.info("[2/4] Training 3D Gaussian Splatting …")
    from pipeline.gaussian_splatting.train_gaussian_splat import GaussianSplatTrainer

    gs_config = config.get("gaussian_splatting", {})
    trainer = GaussianSplatTrainer(
        colmap_dir=colmap_dir,
        image_dir=image_dir,
        output_dir=gs_output,
        config=gs_config,
        device=device,
    )
    ply_path = trainer.train()
    if not ply_path:
        logger.error("3DGS training failed.")
        return False

    # Step 3: Export to SPZ
    logger.info("[3/4] Exporting to SPZ v4 …")
    from pipeline.gaussian_splatting.export_spz import export_ply_to_spz
    from pipeline.gaussian_splatting.spz_encoder import CoordinateSystem

    spz_cfg = gs_config.get("spz_export", {})
    export_ply_to_spz(
        ply_path=ply_path,
        spz_path=spz_path,
        sh_degree=gs_config.get("sh_degree", 0),
        fractional_bits=spz_cfg.get("fractional_bits", 12),
        sh1_bits=spz_cfg.get("sh1_bits", 5),
        sh_rest_bits=spz_cfg.get("shrest_bits", 4),
        source_coord=CoordinateSystem.RDF,
    )

    # Step 4: Extract morphology
    logger.info("[4/4] Extracting Gaussian-based morphology …")
    from pipeline.gaussian_splatting.splat_morphology import GaussianMorphologyExtractor

    morph_cfg = config.get("morphology", {})
    extractor = GaussianMorphologyExtractor(
        opacity_threshold=gs_config.get("opacity_threshold", 0.3),
        dbscan_eps=morph_cfg.get("dbscan_eps", 0.008),
        dbscan_min_samples=morph_cfg.get("dbscan_min_samples", 5),
        min_gaussians_per_boll=morph_cfg.get("min_points_per_boll", 10),
    )
    measurements = extractor.extract_all(ply_path)

    os.makedirs(os.path.dirname(morph_path), exist_ok=True)
    with open(morph_path, "w") as f:
        json.dump(measurements, f, indent=2)

    # Prepare web export
    prepare_web_export(repo_root, condition, spz_path, morph_path)

    logger.info("✓ Pipeline complete for %s", condition)
    logger.info("  PLY:  %s", ply_path)
    logger.info("  SPZ:  %s", spz_path)
    logger.info("  Morph: %s", morph_path)
    return True


def main():
    repo_root = str(Path(__file__).parent.parent.parent)

    parser = argparse.ArgumentParser(description="Run full 3DGS pipeline")
    parser.add_argument("--condition", choices=["pre_defoliation", "post_defoliation", "both"], default="both")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    args = parser.parse_args()

    with open(os.path.join(repo_root, "configs", "pipeline_config.yaml")) as f:
        config = yaml.safe_load(f)

    if args.iterations:
        config.setdefault("gaussian_splatting", {})["num_iterations"] = args.iterations

    device = args.device or config.get("gaussian_splatting", {}).get(
        "device", config.get("feature_extraction", {}).get("device", "mps"))

    conditions = ["pre_defoliation", "post_defoliation"] if args.condition == "both" else [args.condition]

    results = {}
    for cond in conditions:
        results[cond] = run_pipeline(cond, repo_root, config, device)

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    for cond, ok in results.items():
        logger.info("  %s: %s", cond, "✓" if ok else "✗")


if __name__ == "__main__":
    main()
