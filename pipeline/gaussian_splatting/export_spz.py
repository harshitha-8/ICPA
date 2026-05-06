"""
SPZ Export Orchestrator — ICPA Cotton Boll Pipeline

Converts a trained 3DGS PLY to SPZ v4 format for web delivery.
Uses the pure-Python encoder (spz_encoder.py) by default; can
optionally shell out to the nianticlabs/spz CLI binary if available.
"""

import os
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from pipeline.gaussian_splatting.spz_encoder import (
    save_spz, CoordinateSystem, decode_spz_header,
)
from pipeline.gaussian_splatting.splat_morphology import load_gaussian_ply

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def export_ply_to_spz(
    ply_path: str,
    spz_path: str,
    sh_degree: int = 0,
    fractional_bits: int = 12,
    sh1_bits: int = 5,
    sh_rest_bits: int = 4,
    source_coord: int = CoordinateSystem.RDF,
    cli_binary: Optional[str] = None,
) -> str:
    """
    Convert a 3DGS PLY to SPZ v4.

    Args:
        ply_path: input Gaussian PLY
        spz_path: output SPZ path
        cli_binary: optional path to compiled spz CLI binary

    Returns:
        Path to the saved SPZ file.
    """
    # Strategy A: use compiled CLI if available
    if cli_binary and os.path.exists(cli_binary):
        logger.info("Using SPZ CLI binary: %s", cli_binary)
        try:
            result = subprocess.run(
                [cli_binary, "encode", ply_path, "-o", spz_path],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                logger.info("CLI export succeeded → %s", spz_path)
                return spz_path
            else:
                logger.warning("CLI failed: %s — falling back to Python encoder", result.stderr)
        except Exception as e:
            logger.warning("CLI error: %s — falling back to Python encoder", e)

    # Strategy B: pure-Python encoder
    logger.info("Using pure-Python SPZ v4 encoder")
    gaussians = load_gaussian_ply(ply_path)

    xyz = gaussians["xyz"]
    opacity = gaussians["opacity"]
    scales = gaussians["scales"]
    rotations = gaussians["rotations"]
    sh_dc = gaussians["sh_dc"]

    # Build SH coefficients array (DC only for now)
    sh_coeffs = sh_dc[:, np.newaxis, :]  # (N, 1, 3)

    # Scales: convert back to log-scale for the encoder
    log_scales = np.log(np.clip(scales, 1e-7, None))

    result_path = save_spz(
        output_path=spz_path,
        positions=xyz,
        opacities=opacity,
        sh_coeffs=sh_coeffs,
        scales=log_scales,
        rotations=rotations,
        sh_degree=sh_degree,
        fractional_bits=fractional_bits,
        source_coord_system=source_coord,
        sh1_bits=sh1_bits,
        sh_rest_bits=sh_rest_bits,
    )

    # Verify
    with open(result_path, "rb") as f:
        header = decode_spz_header(f.read())
    logger.info("SPZ verification: %s", json.dumps(header, indent=2))

    # Compare sizes
    ply_size = os.path.getsize(ply_path)
    spz_size = os.path.getsize(result_path)
    logger.info(
        "Compression: PLY=%.2f MB → SPZ=%.2f MB (%.1f× smaller)",
        ply_size / 1e6, spz_size / 1e6, ply_size / max(spz_size, 1),
    )

    return result_path


def main():
    """CLI: convert trained 3DGS PLY to SPZ."""
    repo_root = Path(__file__).parent.parent.parent

    parser = argparse.ArgumentParser(description="Export 3DGS PLY to SPZ v4")
    parser.add_argument("--condition", choices=["pre_defoliation", "post_defoliation", "both"], default="both")
    parser.add_argument("--ply", type=str, default=None, help="Override PLY path")
    parser.add_argument("--output", type=str, default=None, help="Override SPZ output path")
    parser.add_argument("--cli-binary", type=str, default=None, help="Path to spz CLI binary")
    args = parser.parse_args()

    with open(repo_root / "configs" / "pipeline_config.yaml") as f:
        cfg = yaml.safe_load(f)

    gs_cfg = cfg.get("gaussian_splatting", {})
    spz_cfg = gs_cfg.get("spz_export", {})

    conditions = ["pre_defoliation", "post_defoliation"] if args.condition == "both" else [args.condition]

    for cond in conditions:
        ply_path = args.ply or os.path.join(
            str(repo_root), "outputs", "gaussian_splats", cond, "point_cloud.ply")
        spz_path = args.output or os.path.join(
            str(repo_root), "outputs", "web_export", f"{cond}.spz")

        if not os.path.exists(ply_path):
            logger.warning("PLY not found: %s — run training first.", ply_path)
            continue

        logger.info("=" * 60)
        logger.info("SPZ Export: %s", cond)

        export_ply_to_spz(
            ply_path=ply_path,
            spz_path=spz_path,
            sh_degree=spz_cfg.get("sh_degree", 0),
            fractional_bits=spz_cfg.get("fractional_bits", 12),
            sh1_bits=spz_cfg.get("sh1_bits", 5),
            sh_rest_bits=spz_cfg.get("shrest_bits", 4),
            source_coord=CoordinateSystem.RDF,
            cli_binary=args.cli_binary,
        )

    logger.info("SPZ export complete.")


if __name__ == "__main__":
    main()
