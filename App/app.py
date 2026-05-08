#!/usr/bin/env python3
"""Local browser app for cotton 3D reconstruction and boll proxy measurement."""

from __future__ import annotations

import json
import socket
import base64
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from reconstruction_core import dataset_payload, reconstruct_dataset_image

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
DEFAULT_PORT = 8917
LOCAL_BOLL_DIR = REPO_ROOT / "outputs" / "metrics" / "local_boll_2p5d_reconstruction"


def encode_file(path: Path, mime: str) -> str:
    if not path.exists():
        return ""
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def local_boll_payload() -> dict[str, object]:
    summary_path = LOCAL_BOLL_DIR / "local_boll_2p5d_summary.csv"
    rows: list[dict[str, str]] = []
    if summary_path.exists():
        with summary_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    if rows and rows[0].get("crop_path"):
        crop_path = Path(rows[0]["crop_path"])
        best_crop = crop_path if crop_path.is_absolute() else REPO_ROOT / crop_path
    else:
        best_crop = LOCAL_BOLL_DIR / "crops" / "001_post_DJI_20250929124623_0166_D_cand333.jpg"
    return {
        "scientific_boundary": "Interactive single-crop 2.5D proxy; not calibrated boll 3D or volume.",
        "gallery_image": encode_file(LOCAL_BOLL_DIR / "local_boll_2p5d_gallery.png", "image/png"),
        "best_view_image": encode_file(LOCAL_BOLL_DIR / "best_local_boll_2p5d_view.png", "image/png"),
        "best_crop_image": encode_file(best_crop, "image/jpeg"),
        "rotation_video": "/asset/local-boll-video",
        "summary": rows[:12],
    }


def find_free_port(start: int = DEFAULT_PORT, end: int = 8999) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found between {start} and {end}.")


class CottonAppHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(INDEX_HTML)
            return
        if parsed.path == "/api/images":
            self._send_json(dataset_payload())
            return
        if parsed.path == "/api/local-boll":
            self._send_json(local_boll_payload())
            return
        if parsed.path == "/asset/local-boll-video":
            self._send_file(LOCAL_BOLL_DIR / "best_local_boll_2p5d_rotation.mp4", "video/mp4")
            return
        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/reconstruct":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = reconstruct_dataset_image(
                phase=payload.get("phase", "pre"),
                label=payload.get("label"),
                max_points=int(payload.get("max_points", 12000)),
                gsd_cm_per_px=float(payload.get("gsd_cm_per_px", 0.25)),
            )
            self._send_json(result)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, mime: str) -> None:
        if not path.exists():
            self.send_error(404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ICPA Cotton Phenotyping Workspace</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f8f4;
      --panel: #ffffff;
      --ink: #182116;
      --muted: #65715f;
      --line: #dfe7dc;
      --accent: #2f7d4f;
      --accent-dark: #225f3c;
      --soft: #eef5ec;
      --warn: #8a5a18;
      --danger: #a43d3d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main { max-width: 1440px; margin: 0 auto; padding: 18px; }
    header {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 16px;
      align-items: end;
      border-bottom: 1px solid var(--line);
      padding: 8px 0 16px;
      margin-bottom: 14px;
    }
    h1 { margin: 0 0 6px; font-size: 25px; line-height: 1.15; letter-spacing: 0; }
    h2 { margin: 0 0 10px; font-size: 17px; letter-spacing: 0; }
    p { margin: 0; color: var(--muted); line-height: 1.45; }
    .byline { color: var(--muted); font-size: 13px; text-align: right; }
    .status { margin-top: 4px; color: var(--muted); font-size: 12px; }
    .layout {
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      gap: 14px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }
    .controls { position: sticky; top: 14px; }
    label { display: block; font-size: 13px; color: var(--muted); margin: 12px 0 6px; }
    select, input {
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      background: #fff;
      color: var(--ink);
      font-size: 14px;
    }
    button {
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      font-weight: 700;
      cursor: pointer;
    }
    #runButton { width: 100%; min-height: 42px; margin-top: 14px; }
    button:disabled { opacity: 0.6; cursor: progress; }
    .tabs {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 12px;
    }
    .tab {
      min-height: 38px;
      background: #fff;
      color: var(--accent-dark);
      border: 1px solid var(--line);
    }
    .tab.active { background: var(--accent); color: #fff; border-color: var(--accent); }
    .page { display: none; }
    .page.active { display: block; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfcfa;
      min-height: 74px;
    }
    .metric strong { display: block; font-size: 21px; line-height: 1.15; }
    .metric span { color: var(--muted); font-size: 12px; }
    .image-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }
    .two-col {
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(280px, 0.65fr);
      gap: 12px;
      align-items: start;
    }
    .figure-title { font-size: 13px; font-weight: 700; color: var(--muted); margin: 0 0 7px; }
    img {
      display: block;
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      aspect-ratio: 4 / 3;
      object-fit: contain;
    }
    .map-image { aspect-ratio: 16 / 10; }
    .crop-gallery {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 10px;
      margin-top: 8px;
    }
    .crop-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfa;
      overflow: hidden;
    }
    .crop-card img {
      border: 0;
      border-radius: 0;
      aspect-ratio: 1 / 1;
      object-fit: cover;
    }
    .crop-meta { padding: 7px 8px 8px; font-size: 12px; color: var(--muted); line-height: 1.35; }
    .crop-meta strong { display: block; color: var(--ink); font-size: 13px; margin-bottom: 2px; }
    .table-wrap {
      max-height: 380px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid var(--line); padding: 8px; text-align: right; white-space: nowrap; }
    th:first-child, td:first-child { text-align: left; }
    th { position: sticky; top: 0; background: #f7faf6; color: var(--muted); }
    .note {
      border: 1px solid #ead8ad;
      background: #fff9e9;
      color: var(--warn);
      border-radius: 8px;
      padding: 12px;
      line-height: 1.45;
      font-size: 13px;
      margin-bottom: 12px;
    }
    .export-box {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfa;
      overflow-wrap: anywhere;
      color: var(--muted);
      line-height: 1.45;
    }
    .empty {
      border: 1px dashed var(--line);
      border-radius: 8px;
      padding: 18px;
      color: var(--muted);
      background: #fbfcfa;
    }
    .error { color: var(--danger); margin-top: 10px; font-size: 13px; }
    .boll3d-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) minmax(360px, 0.9fr);
      gap: 12px;
      align-items: start;
    }
    .interactive-stage {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      overflow: hidden;
      position: relative;
    }
    #boll3dCanvas {
      display: block;
      width: 100%;
      height: 520px;
      cursor: grab;
      touch-action: none;
      background: #fff;
    }
    #boll3dCanvas:active { cursor: grabbing; }
    .small-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    video {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      display: block;
    }
    @media (max-width: 980px) {
      header, .layout, .two-col, .boll3d-grid, .small-grid { grid-template-columns: 1fr; }
      .byline { text-align: left; }
      .controls { position: static; }
      .tabs, .metrics, .image-grid { grid-template-columns: 1fr; }
      .crop-gallery { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>ICPA Cotton Phenotyping Workspace</h1>
      <p>Pre/post-defoliation scouting, boll counting, plot-grid mapping, and proxy trait review.</p>
    </div>
    <div class="byline">
      <strong>Created by Harshitha Manjunatha</strong>
      <div class="status" id="datasetStatus">Loading dataset index...</div>
    </div>
  </header>

  <div class="layout">
    <section class="panel controls">
      <h2>Run Setup</h2>
      <label for="phase">Phase</label>
      <select id="phase">
        <option value="post" selected>Post-defoliation</option>
        <option value="pre">Pre-defoliation</option>
      </select>

      <label for="imageSelect">Dataset image</label>
      <select id="imageSelect"></select>

      <label for="gsd">Proxy scale, cm per pixel</label>
      <input id="gsd" type="number" min="0.001" max="5" step="0.001" value="0.250" />

      <button id="runButton">Run Cotton Analysis</button>
      <div class="error" id="error"></div>
      <div class="note" style="margin-top:12px">
        Bale estimation is not produced from the current volume proxy. A defensible bale estimate needs calibrated field area, boll mass or lint-turnout calibration, and validation against harvested yield.
      </div>
    </section>

    <section class="panel">
      <nav class="tabs" aria-label="Workspace pages">
        <button class="tab active" data-page="overview">Overview</button>
        <button class="tab" data-page="scouting">Scouting Map</button>
        <button class="tab" data-page="measurements">Measurements</button>
        <button class="tab" data-page="local3d">Local Boll 3D</button>
        <button class="tab" data-page="exports">Exports & Notes</button>
      </nav>

      <section id="overview" class="page active">
        <div class="metrics">
          <div class="metric"><strong id="countMetric">-</strong><span>adjusted boll count</span></div>
          <div class="metric"><strong id="rawMetric">-</strong><span>raw candidates</span></div>
          <div class="metric"><strong id="usableMetric">-</strong><span>measurement-ready candidates</span></div>
          <div class="metric"><strong id="diamMetric">-</strong><span>median mask length x width, cm</span></div>
          <div class="metric"><strong id="volMetric">-</strong><span>median ellipsoid volume proxy, cm3</span></div>
        </div>
        <div class="image-grid">
          <div><div class="figure-title">Input frame</div><img id="inputImage" alt="input frame" /></div>
          <div><div class="figure-title">Raw detector overlay</div><img id="overlayImage" alt="detector overlay" /></div>
          <div><div class="figure-title">Measurement-ready extraction</div><img id="extractionImage" alt="extraction overlay" /></div>
        </div>
      </section>

      <section id="scouting" class="page">
        <div class="two-col">
          <div>
            <div class="figure-title">Row-column scouting map</div>
            <img class="map-image" id="plotMapImage" alt="plot grid map" />
          </div>
          <div>
            <h2>Scouting Interpretation</h2>
            <p>The useful idea from geospatial 3D sports-map visualizations is spatial organization: place the activity on a map that matches the real site. For this cotton work, the current safe version is a plot-grid scouting map, not a fake terrain surface.</p>
            <div class="note">This grid is image-coordinate scouting. It becomes a field-coordinate map only after orthomosaic, camera pose, GPS/GCP, or plot-boundary calibration.</div>
          </div>
        </div>
        <div class="figure-title" style="margin-top:12px">Highest-count plot cells</div>
        <div class="table-wrap"><table id="plotCellTable"></table></div>
      </section>

      <section id="measurements" class="page">
        <div class="note">Length, width, diameter, and volume remain proxy traits until scale calibration and physical boll measurements are added. The count and scouting layers are currently more reliable than bale or volume estimation.</div>
        <div class="figure-title">Extracted cotton-boll candidates, strongest 36 by confidence</div>
        <div class="crop-gallery" id="cropGallery"></div>
        <div class="figure-title" style="margin-top:12px">Top 75 proxy measurement candidates</div>
        <div class="table-wrap"><table id="measureTable"></table></div>
      </section>

      <section id="local3d" class="page">
        <div class="note">This page shows local cotton boll/cluster 2.5D proxies from real UAV crops. Drag the interactive view to rotate the selected crop. These views are for target selection and visual review; calibrated volume still requires multi-view geometry and scale validation.</div>
        <div class="boll3d-grid">
          <div>
            <div class="figure-title">Ranked local cotton-cluster reconstructions, 3 rows x 4 columns</div>
            <img class="map-image" id="localBollGallery" alt="local cotton boll 2.5D reconstruction gallery" />
          </div>
          <div>
            <div class="figure-title">Interactive selected crop, drag to rotate</div>
            <div class="interactive-stage"><canvas id="boll3dCanvas" width="760" height="520"></canvas></div>
          </div>
        </div>
        <div class="small-grid" style="margin-top:12px">
          <div>
            <div class="figure-title">Best static view</div>
            <img id="localBollBestView" alt="best local cotton boll 2.5D view" />
          </div>
          <div>
            <div class="figure-title">Rotation video</div>
            <video id="localBollVideo" controls muted loop playsinline></video>
          </div>
        </div>
      </section>

      <section id="exports" class="page">
        <div class="two-col">
          <div>
            <div class="figure-title">Morphology depth proxy</div>
            <img id="depthImage" alt="depth field" />
          </div>
          <div>
            <h2>Exports</h2>
            <div id="exportText" class="export-box">Run analysis to create local PLY and CSV outputs.</div>
            <div class="note" style="margin-top:12px">The SAM-to-3D highlighted overlay has been removed for now. It should return only after calibrated reconstruction or a stronger multi-view geometry step makes the 3D evidence defensible.</div>
          </div>
        </div>
      </section>
    </section>
  </div>
</main>

<script>
let dataset = {pre: [], post: []};
let localBoll = null;
let bollPoints = [];
let bollTexture = null;
let bollRotation = {x: -0.68, y: 0.72};
let dragState = null;

const phase = document.getElementById("phase");
const imageSelect = document.getElementById("imageSelect");
const runButton = document.getElementById("runButton");

async function loadDataset() {
  const res = await fetch("/api/images");
  dataset = await res.json();
  document.getElementById("datasetStatus").textContent = `${dataset.pre.length} pre frames, ${dataset.post.length} post frames indexed`;
  updateChoices();
  loadLocalBoll();
}

async function loadLocalBoll() {
  const res = await fetch("/api/local-boll");
  localBoll = await res.json();
  document.getElementById("localBollGallery").src = localBoll.gallery_image;
  document.getElementById("localBollBestView").src = localBoll.best_view_image;
  document.getElementById("localBollVideo").src = localBoll.rotation_video;
  buildInteractiveBoll(localBoll.best_crop_image);
}

function updateChoices() {
  imageSelect.innerHTML = "";
  for (const item of dataset[phase.value] || []) {
    const opt = document.createElement("option");
    opt.value = item.label;
    opt.textContent = item.label;
    imageSelect.appendChild(opt);
  }
}

function showPage(pageId) {
  document.querySelectorAll(".page").forEach(page => page.classList.toggle("active", page.id === pageId));
  document.querySelectorAll(".tab").forEach(tab => tab.classList.toggle("active", tab.dataset.page === pageId));
}

async function run() {
  runButton.disabled = true;
  runButton.textContent = "Running...";
  document.getElementById("error").textContent = "";
  try {
    const res = await fetch("/api/reconstruct", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        phase: phase.value,
        label: imageSelect.value,
        max_points: 8000,
        gsd_cm_per_px: Number(document.getElementById("gsd").value)
      })
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    renderResult(data);
  } catch (err) {
    document.getElementById("error").textContent = err.message;
  } finally {
    runButton.disabled = false;
    runButton.textContent = "Run Cotton Analysis";
  }
}

function renderResult(data) {
  const s = data.summary;
  document.getElementById("countMetric").textContent = s.adjusted_count;
  document.getElementById("rawMetric").textContent = s.raw_candidates;
  document.getElementById("usableMetric").textContent = s.measurement_candidates;
  document.getElementById("diamMetric").textContent = `${s.median_length_cm_proxy} x ${s.median_width_cm_proxy}`;
  document.getElementById("volMetric").textContent = s.median_ellipsoid_volume_cm3_proxy;
  document.getElementById("inputImage").src = data.input_image;
  document.getElementById("overlayImage").src = data.annotated_image;
  document.getElementById("extractionImage").src = data.extraction_overlay_image;
  document.getElementById("plotMapImage").src = data.plot_map_image;
  document.getElementById("depthImage").src = data.depth_image;
  document.getElementById("exportText").innerHTML = `<strong>Scene PLY:</strong><br>${s.ply}<br><br><strong>CSV:</strong><br>${s.measurements_csv}<br><br><strong>Scene points:</strong> ${s.point_count}`;
  renderCrops(data.boll_crops);
  renderTable(data.measurements);
  renderPlotCells(data.plot_cells);
}

function renderCrops(rows) {
  const gallery = document.getElementById("cropGallery");
  gallery.innerHTML = rows.map(r => `
    <div class="crop-card">
      <img src="${r.crop_image}" alt="extracted cotton boll ${r.id}" />
      <div class="crop-meta">
        <strong>#${r.id} | ${r.length_cm_proxy} x ${r.width_cm_proxy} cm</strong>
        vol proxy ${r.ellipsoid_volume_cm3_proxy} cm3<br>
        q ${r.extraction_quality} | mask ${r.mask_area_px}px
      </div>
    </div>
  `).join("");
}

function renderTable(rows) {
  const table = document.getElementById("measureTable");
  const cols = ["id", "length_cm_proxy", "width_cm_proxy", "diameter_cm_proxy", "ellipsoid_volume_cm3_proxy", "extraction_quality", "lint_fraction", "green_fraction", "visibility_proxy"];
  table.innerHTML = `<thead><tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr></thead>` +
    `<tbody>${rows.map(r => `<tr>${cols.map(c => `<td>${r[c]}</td>`).join("")}</tr>`).join("")}</tbody>`;
}

function renderPlotCells(rows) {
  const table = document.getElementById("plotCellTable");
  const cols = ["row", "column", "boll_count", "mean_diameter_cm_proxy", "mean_volume_cm3_proxy", "mean_extraction_quality"];
  table.innerHTML = `<thead><tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr></thead>` +
    `<tbody>${rows.map(r => `<tr>${cols.map(c => `<td>${r[c]}</td>`).join("")}</tr>`).join("")}</tbody>`;
}

document.querySelectorAll(".tab").forEach(tab => tab.addEventListener("click", () => showPage(tab.dataset.page)));
phase.addEventListener("change", updateChoices);
runButton.addEventListener("click", run);
loadDataset();

function buildInteractiveBoll(src) {
  if (!src) return;
  const img = new Image();
  img.onload = () => {
    const sample = document.createElement("canvas");
    const size = 184;
    sample.width = size;
    sample.height = size;
    const sctx = sample.getContext("2d");
    sctx.imageSmoothingQuality = "high";
    sctx.drawImage(img, 0, 0, size, size);
    bollTexture = sample;
    const data = sctx.getImageData(0, 0, size, size).data;
    bollPoints = [];
    const step = 3;
    for (let y = 0; y < size - step; y += step) {
      for (let x = 0; x < size - step; x += step) {
        const i = (y * size + x) * 4;
        const r = data[i], g = data[i + 1], b = data[i + 2];
        const maxc = Math.max(r, g, b) / 255;
        const minc = Math.min(r, g, b) / 255;
        const sat = maxc > 0 ? (maxc - minc) / maxc : 0;
        const exg = 2 * g - r - b;
        const lint = Math.max(0, Math.min(1, (maxc - 0.48) / 0.38)) * Math.max(0, Math.min(1, (0.58 - sat) / 0.58)) * Math.max(0, Math.min(1, (90 - exg) / 150));
        const gray = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
        const edge = localContrast(data, size, x, y);
        let z = 0.22 * lint + 0.10 * Math.sqrt(Math.max(lint, 0)) + 0.035 * edge + 0.025 * gray;
        z = Math.min(0.32, z);
        if (maxc < 0.22 && lint < 0.05) z *= 0.35;
        bollPoints.push({
          x: (x / (size - 1) - 0.5) * 2.0,
          y: (0.5 - y / (size - 1)) * 2.0,
          z,
          sx: x,
          sy: y,
          sw: step,
          r, g, b,
          lint
        });
      }
    }
    drawInteractiveBoll();
  };
  img.src = src;
}

function localContrast(data, size, x, y) {
  const i = (y * size + x) * 4;
  const center = (data[i] + data[i + 1] + data[i + 2]) / 3;
  const x2 = Math.min(size - 1, x + 2);
  const y2 = Math.min(size - 1, y + 2);
  const ix = (y * size + x2) * 4;
  const iy = (y2 * size + x) * 4;
  const gx = Math.abs(center - (data[ix] + data[ix + 1] + data[ix + 2]) / 3);
  const gy = Math.abs(center - (data[iy] + data[iy + 1] + data[iy + 2]) / 3);
  return Math.min(1, Math.sqrt(gx * gx + gy * gy) / 110);
}

function drawInteractiveBoll() {
  const canvas = document.getElementById("boll3dCanvas");
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(640, Math.floor(rect.width * dpr));
  canvas.height = Math.max(430, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);
  const cx = w / 2;
  const cy = h / 2 + 10;
  const scale = Math.min(w, h) * 0.36;
  const sinY = Math.sin(bollRotation.y), cosY = Math.cos(bollRotation.y);
  const sinX = Math.sin(bollRotation.x), cosX = Math.cos(bollRotation.x);
  const projected = bollPoints.map(p => {
    const x1 = p.x * cosY + p.z * sinY;
    const z1 = -p.x * sinY + p.z * cosY;
    const y1 = p.y * cosX - z1 * sinX;
    const z2 = p.y * sinX + z1 * cosX;
    const perspective = 1.0 / (1.0 + 0.38 * z2);
    return {...p, px: cx + x1 * scale * perspective, py: cy - y1 * scale * perspective, depth: z2, perspective};
  }).sort((a, b) => a.depth - b.depth);
  ctx.shadowColor = "rgba(0,0,0,0.08)";
  ctx.shadowBlur = 10;
  ctx.shadowOffsetY = 8;
  ctx.fillStyle = "rgba(73, 65, 52, 0.12)";
  ctx.beginPath();
  ctx.ellipse(cx, cy + scale * 0.70, scale * 0.78, scale * 0.12, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.shadowColor = "transparent";
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  for (const p of projected) {
    const shade = 0.92 + 0.16 * Math.max(0, p.depth + 0.15);
    const tile = Math.max(2.2, scale * p.sw / 184 * 1.7 * p.perspective);
    ctx.globalAlpha = 0.93 + 0.07 * p.lint;
    ctx.drawImage(bollTexture, p.sx, p.sy, p.sw, p.sw, p.px - tile / 2, p.py - tile / 2, tile, tile);
    if (shade < 0.99) {
      ctx.fillStyle = `rgba(45, 38, 31, ${Math.min(0.12, 1.0 - shade)})`;
      ctx.fillRect(p.px - tile / 2, p.py - tile / 2, tile, tile);
    }
  }
  ctx.globalAlpha = 1;
  ctx.fillStyle = "#65715f";
  ctx.font = "12px Inter, system-ui, sans-serif";
  ctx.fillText("Drag to rotate real UAV crop texture | monocular 2.5D proxy", 14, h - 16);
}

const canvas = document.getElementById("boll3dCanvas");
canvas.addEventListener("pointerdown", e => {
  dragState = {x: e.clientX, y: e.clientY, rx: bollRotation.x, ry: bollRotation.y};
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointermove", e => {
  if (!dragState) return;
  bollRotation.y = dragState.ry + (e.clientX - dragState.x) * 0.012;
  bollRotation.x = Math.max(-1.25, Math.min(0.15, dragState.rx + (e.clientY - dragState.y) * 0.010));
  drawInteractiveBoll();
});
canvas.addEventListener("pointerup", () => { dragState = null; });
window.addEventListener("resize", () => { if (bollPoints.length) drawInteractiveBoll(); });
</script>
</body>
</html>
"""


def main() -> None:
    port = find_free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), CottonAppHandler)
    print(f"ICPA Cotton 3D app: http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
