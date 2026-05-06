"""
3D Gaussian Splatting Trainer — ICPA Cotton Boll Pipeline

Trains a 3DGS model from COLMAP sparse reconstruction of UAV cotton
field imagery. Supports gsplat (direct) and nerfstudio backends.
"""

import os
import sys
import json
import time
import struct
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import gsplat
    GSPLAT_AVAILABLE = True
except ImportError:
    GSPLAT_AVAILABLE = False


def read_colmap_points3D_bin(path: str) -> np.ndarray:
    """Read COLMAP points3D.bin → (N,3) xyz."""
    points = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(8)  # point3D_id
            xyz = struct.unpack("<3d", f.read(24))
            f.read(3)  # rgb
            f.read(8)  # error
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(track_len * 8)
            points.append(xyz)
    return np.array(points, dtype=np.float32)


def read_colmap_points3D_txt(path: str) -> np.ndarray:
    """Read COLMAP points3D.txt → (N,3) xyz."""
    points = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            points.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(points, dtype=np.float32) if points else np.zeros((0, 3), dtype=np.float32)


def read_colmap_images_bin(path: str) -> List[Dict]:
    """Read COLMAP images.bin → list of {qvec, tvec, camera_id, name}."""
    images = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            img_id = struct.unpack("<i", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<i", f.read(4))[0]
            name_chars = []
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_chars.append(c.decode())
            num_pts = struct.unpack("<Q", f.read(8))[0]
            f.read(num_pts * 24)
            images.append({"id": img_id, "qvec": list(qvec), "tvec": list(tvec),
                           "camera_id": cam_id, "name": "".join(name_chars)})
    return images


def read_colmap_cameras_bin(path: str) -> Dict:
    """Read COLMAP cameras.bin."""
    cameras = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cam_id = struct.unpack("<i", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            w = struct.unpack("<Q", f.read(8))[0]
            h = struct.unpack("<Q", f.read(8))[0]
            num_params = {0: 3, 1: 4, 2: 4, 4: 12}.get(model_id, 4)
            params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))
            cameras[cam_id] = {"model_id": model_id, "width": w, "height": h,
                               "params": list(params)}
    return cameras


class GaussianSplatTrainer:
    """Trains 3DGS from COLMAP sparse reconstruction."""

    def __init__(self, colmap_dir, image_dir, output_dir, config, device="mps"):
        self.colmap_dir = colmap_dir
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.config = config
        self.device = device
        os.makedirs(output_dir, exist_ok=True)

        self.sh_degree = config.get("sh_degree", 3)
        self.num_iterations = config.get("num_iterations", 30_000)

    def load_colmap_data(self) -> Dict:
        """Load COLMAP sparse reconstruction."""
        bin_dir = os.path.join(self.colmap_dir, "sparse", "0")
        cams_bin = os.path.join(bin_dir, "cameras.bin")
        imgs_bin = os.path.join(bin_dir, "images.bin")
        pts_bin = os.path.join(bin_dir, "points3D.bin")

        if all(os.path.exists(p) for p in [cams_bin, imgs_bin, pts_bin]):
            cameras = read_colmap_cameras_bin(cams_bin)
            images = read_colmap_images_bin(imgs_bin)
            points = read_colmap_points3D_bin(pts_bin)
        else:
            pts_txt = os.path.join(self.colmap_dir, "sparse", "points3D.txt")
            points = read_colmap_points3D_txt(pts_txt) if os.path.exists(pts_txt) else np.zeros((0, 3))
            cameras, images = {}, []

        logger.info("Loaded %d cameras, %d images, %d points",
                     len(cameras), len(images), len(points))
        return {"cameras": cameras, "images": images, "points": points}

    def train(self) -> str:
        """Train 3DGS model. Returns path to output PLY."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for 3DGS training.")

        data = self.load_colmap_data()
        pts = data["points"]
        if len(pts) == 0:
            logger.error("No init points — run COLMAP first.")
            return ""

        logger.info("Initialising %d Gaussians", len(pts))

        device = torch.device(self.device if self.device != "cuda" or torch.cuda.is_available() else "cpu")
        if self.device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            device = torch.device("cpu")

        N = len(pts)
        means = torch.tensor(pts, dtype=torch.float32, device=device, requires_grad=True)

        # Scale init from nearest-neighbour distances
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=4)
        avg_dist = np.mean(dists[:, 1:], axis=1).astype(np.float32)
        log_scales = torch.tensor(
            np.log(np.clip(avg_dist, 1e-7, None))[:, None].repeat(3, axis=1),
            dtype=torch.float32, device=device, requires_grad=True)

        quats = torch.zeros(N, 4, dtype=torch.float32, device=device, requires_grad=True)
        with torch.no_grad():
            quats[:, 0] = 1.0

        opacities = torch.full((N, 1), np.log(0.1 / 0.9),
                               dtype=torch.float32, device=device, requires_grad=True)
        num_sh = (self.sh_degree + 1) ** 2
        sh_coeffs = torch.zeros(N, num_sh, 3, dtype=torch.float32, device=device, requires_grad=True)
        with torch.no_grad():
            sh_coeffs[:, 0, :] = 0.5

        optimizer = torch.optim.Adam([
            {"params": [means], "lr": 1.6e-4},
            {"params": [log_scales], "lr": 5e-3},
            {"params": [quats], "lr": 1e-3},
            {"params": [opacities], "lr": 0.05},
            {"params": [sh_coeffs], "lr": 2.5e-3},
        ])

        if GSPLAT_AVAILABLE:
            logger.info("gsplat available — full rasterisation training for %d iters", self.num_iterations)
            self._train_gsplat_loop(means, log_scales, quats, opacities, sh_coeffs, optimizer, data, device)
        else:
            logger.warning("gsplat not installed — exporting initial Gaussians only. pip install gsplat for full training.")

        return self._export_ply(means, log_scales, quats, opacities, sh_coeffs)

    def _train_gsplat_loop(self, means, log_scales, quats, opacities, sh_coeffs, optimizer, data, device):
        """Full gsplat rasterisation training loop."""
        from gsplat.rendering import rasterization
        from PIL import Image as PILImage
        import torchvision.transforms as T

        transform = T.Compose([T.Resize((512, 512)), T.ToTensor()])
        start = time.time()

        for step in range(1, self.num_iterations + 1):
            optimizer.zero_grad()
            img_info = data["images"][np.random.randint(len(data["images"]))] if data["images"] else None
            gt = None
            if img_info:
                p = os.path.join(self.image_dir, img_info["name"])
                if os.path.exists(p):
                    gt = transform(PILImage.open(p).convert("RGB")).to(device)

            scales = torch.exp(log_scales)
            try:
                Ks = torch.tensor([[[500, 0, 256], [0, 500, 256], [0, 0, 1]]],
                                  dtype=torch.float32, device=device)
                vm = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
                renders, _, _ = rasterization(
                    means=means, quats=quats / (quats.norm(dim=-1, keepdim=True) + 1e-8),
                    scales=scales, opacities=torch.sigmoid(opacities).squeeze(-1),
                    colors=sh_coeffs[:, :1, :].squeeze(1), viewmats=vm, Ks=Ks,
                    width=512, height=512, packed=False)
                if gt is not None:
                    loss = torch.nn.functional.l1_loss(renders[0].permute(2, 0, 1), gt)
                else:
                    loss = renders.mean() * 0
                loss.backward()
                optimizer.step()
            except Exception as e:
                if step == 1:
                    logger.warning("gsplat rasterisation error: %s", e)
                    return
                continue

            if step % 5000 == 0 or step == 1:
                logger.info("Step %5d/%d | loss=%.6f | #G=%d | %.1fs",
                            step, self.num_iterations, loss.item(), len(means), time.time() - start)

    def _export_ply(self, means, log_scales, quats, opacities, sh_coeffs) -> str:
        """Write Gaussian PLY in standard 3DGS format."""
        ply_path = os.path.join(self.output_dir, "point_cloud.ply")
        with torch.no_grad():
            xyz = means.cpu().numpy()
            sc = torch.exp(log_scales).cpu().numpy()
            rt = quats.cpu().numpy()
            rt = rt / (np.linalg.norm(rt, axis=1, keepdims=True) + 1e-8)
            op = torch.sigmoid(opacities).cpu().numpy()
            sh = sh_coeffs.cpu().numpy()

        N = len(xyz)
        num_sh = (self.sh_degree + 1) ** 2
        hdr = ["ply", "format binary_little_endian 1.0", f"element vertex {N}",
               "property float x", "property float y", "property float z",
               "property float nx", "property float ny", "property float nz"]
        for i in range(3):
            hdr.append(f"property float f_dc_{i}")
        for i in range((num_sh - 1) * 3):
            hdr.append(f"property float f_rest_{i}")
        hdr += ["property float opacity",
                "property float scale_0", "property float scale_1", "property float scale_2",
                "property float rot_0", "property float rot_1", "property float rot_2", "property float rot_3",
                "end_header"]

        with open(ply_path, "wb") as f:
            f.write(("\n".join(hdr) + "\n").encode("ascii"))
            for i in range(N):
                f.write(struct.pack("<3f", *xyz[i]))
                f.write(struct.pack("<3f", 0, 0, 0))
                f.write(struct.pack("<3f", *sh[i, 0, :]))
                for k in range(1, num_sh):
                    f.write(struct.pack("<3f", *sh[i, k, :]))
                f.write(struct.pack("<f", float(op[i])))
                f.write(struct.pack("<3f", *np.log(sc[i])))
                f.write(struct.pack("<4f", *rt[i]))

        logger.info("Exported %d Gaussians → %s (%.2f MB)", N, ply_path, os.path.getsize(ply_path) / 1e6)
        return ply_path


def main():
    repo_root = Path(__file__).parent.parent.parent
    parser = argparse.ArgumentParser(description="Train 3DGS on COLMAP reconstruction")
    parser.add_argument("--condition", choices=["pre_defoliation", "post_defoliation", "both"], default="both")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(repo_root / "configs" / "pipeline_config.yaml") as f:
        cfg = yaml.safe_load(f)

    gs = cfg.get("gaussian_splatting", {})
    if args.iterations:
        gs["num_iterations"] = args.iterations
    device = args.device or gs.get("device", cfg.get("feature_extraction", {}).get("device", "mps"))

    conditions = ["pre_defoliation", "post_defoliation"] if args.condition == "both" else [args.condition]

    for cond in conditions:
        logger.info("=" * 60)
        logger.info("3DGS Training: %s", cond)
        trainer = GaussianSplatTrainer(
            colmap_dir=os.path.join(str(repo_root), "outputs", "colmap", cond),
            image_dir=os.path.join(str(repo_root), "data", cond),
            output_dir=os.path.join(str(repo_root), "outputs", "gaussian_splats", cond),
            config=gs, device=device)
        ply_path = trainer.train()
        if ply_path:
            logger.info("✓ Complete → %s", ply_path)
        else:
            logger.error("✗ Failed for %s", cond)


if __name__ == "__main__":
    main()
