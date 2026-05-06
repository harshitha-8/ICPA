"""
Pure-Python SPZ v4 Encoder — ICPA Cotton Boll Pipeline

Encodes 3D Gaussian Splat data into the SPZ v4 format (Niantic Labs)
without requiring the C++ build. Uses zstandard for ZSTD compression.

SPZ v4 spec:
  - 32-byte NgspFileHeader (plaintext)
  - Table of Contents (N × 16 bytes)
  - N independent ZSTD-compressed attribute streams
  - Streams: positions, alphas, colors, scales, rotations, SH coefficients
"""

import os
import struct
import logging
from typing import Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False
    logger.warning("zstandard not installed. pip install zstandard")

# SPZ v4 constants
SPZ_MAGIC = 0x5053474E  # "NGSP" in little-endian
SPZ_VERSION = 4
NUM_STREAMS_BASE = 6  # positions, alphas, colors, scales, rotations, SH


class CoordinateSystem:
    """Coordinate system conventions."""
    UNSPECIFIED = 0
    RDF = 1   # COLMAP: Right-Down-Forward
    RUB = 2   # OpenGL/three.js: Right-Up-Back
    LUF = 3   # GLB: Left-Up-Forward
    RUF = 4   # Unity: Right-Up-Forward


def _rdf_to_rub(positions: np.ndarray) -> np.ndarray:
    """Convert COLMAP (RDF) to OpenGL/three.js (RUB) coordinates."""
    # RDF → RUB: x stays, y flips, z flips
    converted = positions.copy()
    converted[:, 1] = -positions[:, 1]  # y → -y
    converted[:, 2] = -positions[:, 2]  # z → -z
    return converted


def _quantize_positions(positions: np.ndarray, fractional_bits: int = 12) -> bytes:
    """
    Quantize positions to 24-bit fixed-point signed integers.

    Each coordinate is stored as 3 bytes (little-endian), giving 24-bit
    signed range [-2^23, 2^23-1]. The fractional_bits field controls
    precision vs range trade-off.
    """
    scale = float(1 << fractional_bits)
    quantized = np.round(positions * scale).astype(np.int32)

    # Clamp to 24-bit signed range
    min_val = -(1 << 23)
    max_val = (1 << 23) - 1
    quantized = np.clip(quantized, min_val, max_val)

    # Pack each coordinate as 3 bytes (little-endian)
    data = bytearray()
    for i in range(len(quantized)):
        for j in range(3):
            val = int(quantized[i, j]) & 0xFFFFFF  # mask to 24 bits
            data.extend(struct.pack("<I", val)[:3])  # take lower 3 bytes
    return bytes(data)


def _quantize_alphas(opacities: np.ndarray) -> bytes:
    """Quantize opacities to uint8 (inverse sigmoid space)."""
    # Clip to valid sigmoid range
    opacities = np.clip(opacities, 1e-6, 1 - 1e-6)
    # Convert to logit space, then quantize to uint8
    logits = np.log(opacities / (1 - opacities))
    # Map logit range [-10, 10] to [0, 255]
    normalized = (logits + 10.0) / 20.0
    normalized = np.clip(normalized, 0, 1)
    quantized = (normalized * 255).astype(np.uint8)
    return quantized.tobytes()


def _quantize_colors(sh_dc: np.ndarray) -> bytes:
    """Quantize SH DC coefficients (RGB color) to uint8 per channel."""
    # SH DC to linear color: c = 0.5 + SH_C0 * sh_dc
    SH_C0 = 0.28209479177387814
    colors = 0.5 + SH_C0 * sh_dc
    colors = np.clip(colors, 0, 1)
    quantized = (colors * 255).astype(np.uint8)
    # Flatten: R0 G0 B0 R1 G1 B1 ...
    return quantized.tobytes()


def _quantize_scales(log_scales: np.ndarray) -> bytes:
    """Quantize log-scales to uint8. Maps range [-10, 10] → [0, 255]."""
    normalized = (log_scales + 10.0) / 20.0
    normalized = np.clip(normalized, 0, 1)
    quantized = (normalized * 255).astype(np.uint8)
    return quantized.tobytes()


def _quantize_rotations(quats: np.ndarray) -> bytes:
    """
    Quantize quaternions to compact representation.
    Store the 3 smallest components as uint8, plus a 2-bit index for
    the largest component. For simplicity, we use 4 × uint8 here.
    """
    # Normalise
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / (norms + 1e-10)
    # Ensure w > 0 (canonical form)
    mask = quats[:, 0] < 0
    quats[mask] *= -1
    # Map [-1, 1] → [0, 255]
    normalized = (quats + 1.0) / 2.0
    quantized = (normalized * 255).astype(np.uint8)
    return quantized.tobytes()


def _quantize_sh(sh_rest: np.ndarray, bits: int = 5) -> bytes:
    """
    Quantize higher-order SH coefficients.

    Args:
        sh_rest: (N, num_rest_coeffs, 3) SH coefficients beyond DC
        bits: quantization bits (1-8)
    """
    if sh_rest.size == 0:
        return b""
    max_val = (1 << bits) - 1
    flat = sh_rest.reshape(-1)
    # Map typical SH range [-2, 2] → [0, max_val]
    normalized = (flat + 2.0) / 4.0
    normalized = np.clip(normalized, 0, 1)
    quantized = (normalized * max_val).astype(np.uint8)
    return quantized.tobytes()


def encode_spz(
    positions: np.ndarray,
    opacities: np.ndarray,
    sh_coeffs: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    sh_degree: int = 0,
    fractional_bits: int = 12,
    source_coord_system: int = CoordinateSystem.RDF,
    sh1_bits: int = 5,
    sh_rest_bits: int = 4,
    antialiased: bool = False,
) -> bytes:
    """
    Encode Gaussian splat data to SPZ v4 format.

    Args:
        positions: (N, 3) float32 xyz
        opacities: (N,) float32 in [0, 1]
        sh_coeffs: (N, num_sh, 3) float32 spherical harmonics
        scales: (N, 3) float32 log-scales
        rotations: (N, 4) float32 quaternions (wxyz)
        sh_degree: SH degree (0-4)
        fractional_bits: bits for position fixed-point fractional part
        source_coord_system: coordinate system of input data
        sh1_bits: quantization bits for SH degree 1
        sh_rest_bits: quantization bits for SH degree 2+

    Returns:
        bytes: complete SPZ v4 file content
    """
    if not ZSTD_AVAILABLE:
        raise RuntimeError("zstandard required: pip install zstandard")

    N = len(positions)
    logger.info("Encoding %d Gaussians to SPZ v4 (SH degree %d)", N, sh_degree)

    # Coordinate conversion
    if source_coord_system == CoordinateSystem.RDF:
        positions = _rdf_to_rub(positions)

    # Quantize each attribute stream
    pos_data = _quantize_positions(positions, fractional_bits)
    alpha_data = _quantize_alphas(opacities)
    color_data = _quantize_colors(sh_coeffs[:, 0, :])  # DC term
    scale_data = _quantize_scales(scales)
    rot_data = _quantize_rotations(rotations)

    # SH higher-order terms
    if sh_degree > 0 and sh_coeffs.shape[1] > 1:
        sh_rest = sh_coeffs[:, 1:, :]
        sh_data = _quantize_sh(sh_rest, sh1_bits)
    else:
        sh_data = b""

    # Compress each stream with ZSTD
    compressor = zstd.ZstdCompressor(level=3)
    streams_raw = [pos_data, alpha_data, color_data, scale_data, rot_data, sh_data]
    streams_compressed = []
    for s in streams_raw:
        if len(s) > 0:
            streams_compressed.append(compressor.compress(s))
        else:
            streams_compressed.append(b"")

    num_streams = len(streams_compressed)

    # Build TOC: N × [compressedSize u64, uncompressedSize u64]
    toc = bytearray()
    for i, (raw, comp) in enumerate(zip(streams_raw, streams_compressed)):
        toc.extend(struct.pack("<Q", len(comp)))
        toc.extend(struct.pack("<Q", len(raw)))

    # Flags
    flags = 0
    if antialiased:
        flags |= 0x1

    # Header (32 bytes)
    toc_byte_offset = 32  # no extensions
    header = struct.pack(
        "<I I I B B B B I 12s",
        SPZ_MAGIC,           # magic
        SPZ_VERSION,         # version
        N,                   # numPoints
        sh_degree,           # shDegree
        fractional_bits,     # fractionalBits
        flags,               # flags
        num_streams,         # numStreams
        toc_byte_offset,     # tocByteOffset
        b"\x00" * 12,        # reserved
    )
    assert len(header) == 32

    # Assemble file
    output = bytearray()
    output.extend(header)
    output.extend(toc)
    for comp in streams_compressed:
        output.extend(comp)

    logger.info(
        "SPZ encoded: %d Gaussians, %.2f KB (%.1f× compression vs raw)",
        N, len(output) / 1024,
        sum(len(s) for s in streams_raw) / max(len(output), 1),
    )

    return bytes(output)


def decode_spz_header(data: bytes) -> dict:
    """Decode SPZ v4 header for inspection."""
    if len(data) < 32:
        raise ValueError("File too small for SPZ header")

    magic, version, num_points, sh_degree, frac_bits, flags, num_streams, toc_offset = \
        struct.unpack("<I I I B B B B I", data[:20])

    if magic != SPZ_MAGIC:
        raise ValueError(f"Invalid magic: 0x{magic:08X} (expected 0x{SPZ_MAGIC:08X})")

    return {
        "magic": f"0x{magic:08X}",
        "version": version,
        "num_points": num_points,
        "sh_degree": sh_degree,
        "fractional_bits": frac_bits,
        "flags": flags,
        "antialiased": bool(flags & 0x1),
        "has_extensions": bool(flags & 0x2),
        "num_streams": num_streams,
        "toc_byte_offset": toc_offset,
        "file_size_bytes": len(data),
    }


def save_spz(
    output_path: str,
    positions: np.ndarray,
    opacities: np.ndarray,
    sh_coeffs: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    **kwargs,
) -> str:
    """Encode and save SPZ file. Returns output path."""
    data = encode_spz(positions, opacities, sh_coeffs, scales, rotations, **kwargs)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(data)
    logger.info("Saved SPZ → %s (%.2f KB)", output_path, len(data) / 1024)
    return output_path


# ---------------------------------------------------------------------------
#  Self-test
# ---------------------------------------------------------------------------

def _self_test():
    """Smoke test: encode synthetic Gaussians and verify header."""
    N = 1000
    np.random.seed(42)
    positions = np.random.randn(N, 3).astype(np.float32)
    opacities = np.random.rand(N).astype(np.float32)
    sh_coeffs = np.random.randn(N, 1, 3).astype(np.float32) * 0.1
    scales = np.random.randn(N, 3).astype(np.float32)
    rotations = np.random.randn(N, 4).astype(np.float32)
    rotations /= np.linalg.norm(rotations, axis=1, keepdims=True)

    data = encode_spz(
        positions, opacities, sh_coeffs, scales, rotations,
        sh_degree=0, fractional_bits=12,
    )

    hdr = decode_spz_header(data)
    assert hdr["version"] == 4, f"Expected v4, got {hdr['version']}"
    assert hdr["num_points"] == N, f"Expected {N} points, got {hdr['num_points']}"
    assert hdr["num_streams"] == 6

    # Check size is much smaller than raw float data
    raw_size = N * (3 * 4 + 1 * 4 + 3 * 4 + 3 * 4 + 4 * 4)  # floats
    assert len(data) < raw_size, f"SPZ ({len(data)}) should be smaller than raw ({raw_size})"

    logger.info("✓ Self-test passed: %d Gaussians → %d bytes (%.1f× compression)",
                N, len(data), raw_size / len(data))


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        _self_test()
    else:
        print("Usage: python spz_encoder.py --test")
        print("Or import and use encode_spz() / save_spz() programmatically.")
