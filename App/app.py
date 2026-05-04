#!/usr/bin/env python3
"""Local browser app for cotton 3D reconstruction and boll proxy measurement."""

from __future__ import annotations

import json
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from reconstruction_core import dataset_payload, reconstruct_dataset_image

APP_DIR = Path(__file__).resolve().parent
DEFAULT_PORT = 8917


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


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ICPA Cotton 3D App</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7f4;
      --panel: #ffffff;
      --ink: #162013;
      --muted: #687165;
      --line: #dfe6dc;
      --accent: #247a50;
      --accent-2: #146d8f;
      --danger: #a43d3d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main { padding: 18px; max-width: 1500px; margin: 0 auto; }
    header {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: end;
      padding: 12px 0 18px;
      border-bottom: 1px solid var(--line);
      margin-bottom: 16px;
    }
    h1 { font-size: 24px; line-height: 1.15; margin: 0 0 6px; letter-spacing: 0; }
    p { margin: 0; color: var(--muted); line-height: 1.45; }
    .status { color: var(--muted); font-size: 13px; text-align: right; }
    .layout {
      display: grid;
      grid-template-columns: 330px minmax(0, 1fr);
      gap: 16px;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }
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
      width: 100%;
      min-height: 42px;
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      font-weight: 700;
      margin-top: 14px;
      cursor: pointer;
    }
    button:disabled { opacity: 0.58; cursor: progress; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfcfa;
      min-height: 72px;
    }
    .metric strong { display: block; font-size: 20px; line-height: 1.2; }
    .metric span { color: var(--muted); font-size: 12px; }
    .grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 12px;
    }
    .panel-title { font-size: 13px; font-weight: 700; color: var(--muted); margin-bottom: 8px; }
    img, canvas {
      display: block;
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      aspect-ratio: 4 / 3;
      object-fit: contain;
    }
    canvas { cursor: grab; }
    .map-panel { grid-column: span 2; }
    .viewer { grid-column: span 3; }
    .terrain-panel { grid-column: span 3; }
    #topoCanvas {
      aspect-ratio: 16 / 7;
      cursor: default;
    }
    .crop-gallery {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 10px;
      margin: 8px 0 14px;
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
    .crop-meta {
      padding: 7px 8px 8px;
      font-size: 12px;
      color: var(--muted);
      line-height: 1.35;
    }
    .crop-meta strong {
      display: block;
      color: var(--ink);
      font-size: 13px;
      margin-bottom: 2px;
    }
    .table-wrap {
      max-height: 320px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid var(--line); padding: 8px; text-align: right; white-space: nowrap; }
    th:first-child, td:first-child { text-align: left; }
    th { position: sticky; top: 0; background: #f7faf6; color: var(--muted); }
    .error { color: var(--danger); margin-top: 10px; font-size: 13px; }
    @media (max-width: 980px) {
      .layout, header { grid-template-columns: 1fr; }
      .grid, .metrics { grid-template-columns: 1fr; }
      .crop-gallery { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .viewer, .map-panel, .terrain-panel { grid-column: auto; }
      .status { text-align: left; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>ICPA Cotton 3D Reconstruction Workspace</h1>
      <p>Dataset frame selection, detector overlay, morphology-depth preview, point-cloud navigation, and proxy boll measurements.</p>
    </div>
    <div class="status" id="datasetStatus">Loading dataset index...</div>
  </header>

  <div class="layout">
    <section>
      <label for="phase">Phase</label>
      <select id="phase">
        <option value="post" selected>Post-defoliation</option>
        <option value="pre">Pre-defoliation</option>
      </select>

      <label for="imageSelect">Dataset image</label>
      <select id="imageSelect"></select>

      <label for="maxPoints">Point cloud density</label>
      <input id="maxPoints" type="number" min="2000" max="50000" step="1000" value="12000" />

      <label for="gsd">Scale assumption, cm per pixel</label>
      <input id="gsd" type="number" min="0.001" max="5" step="0.001" value="0.250" />

      <button id="runButton">Generate Reconstruction</button>
      <div class="error" id="error"></div>
    </section>

    <section>
      <div class="metrics">
        <div class="metric"><strong id="countMetric">-</strong><span>adjusted boll count</span></div>
        <div class="metric"><strong id="rawMetric">-</strong><span>raw candidates</span></div>
        <div class="metric"><strong id="usableMetric">-</strong><span>measurement-ready candidates</span></div>
        <div class="metric"><strong id="diamMetric">-</strong><span>median diameter proxy, cm</span></div>
        <div class="metric"><strong id="volMetric">-</strong><span>median volume proxy, cm3</span></div>
      </div>

      <div class="grid">
        <div><div class="panel-title">Input frame</div><img id="inputImage" alt="input frame" /></div>
        <div><div class="panel-title">Raw detector overlay</div><img id="overlayImage" alt="detector overlay" /></div>
        <div><div class="panel-title">Measurement-ready extraction</div><img id="extractionImage" alt="extraction overlay" /></div>
        <div class="map-panel"><div class="panel-title">Plot grid map proxy</div><img id="plotMapImage" alt="plot grid map" /></div>
        <div class="terrain-panel"><div class="panel-title">Topographical boll-density landscape</div><canvas id="topoCanvas" width="980" height="430"></canvas></div>
        <div><div class="panel-title">Morphology depth</div><img id="depthImage" alt="depth field" /></div>
        <div class="viewer"><div class="panel-title">Interactive point-cloud view</div><canvas id="cloudCanvas" width="920" height="520"></canvas></div>
        <div>
          <div class="panel-title">Exports</div>
          <p id="exportText">Run reconstruction to create local PLY and CSV outputs.</p>
        </div>
      </div>

      <div class="panel-title">Extracted cotton-boll candidates, strongest 36 by extraction confidence</div>
      <div class="crop-gallery" id="cropGallery"></div>

      <div class="panel-title">Boll proxy measurements, top 75 candidates by extraction confidence</div>
      <div class="table-wrap"><table id="measureTable"></table></div>

      <div class="panel-title">Plot-cell map summary, highest-count cells</div>
      <div class="table-wrap"><table id="plotCellTable"></table></div>
    </section>
  </div>
</main>

<script>
let dataset = {pre: [], post: []};
let cloud = [];
let angleX = -0.9;
let angleZ = 0.35;
let dragging = false;
let last = [0, 0];

const phase = document.getElementById("phase");
const imageSelect = document.getElementById("imageSelect");
const runButton = document.getElementById("runButton");
const canvas = document.getElementById("cloudCanvas");
const ctx = canvas.getContext("2d");
const topoCanvas = document.getElementById("topoCanvas");
const topoCtx = topoCanvas.getContext("2d");

async function loadDataset() {
  const res = await fetch("/api/images");
  dataset = await res.json();
  document.getElementById("datasetStatus").textContent = `${dataset.pre.length} pre frames, ${dataset.post.length} post frames indexed`;
  updateChoices();
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
        max_points: Number(document.getElementById("maxPoints").value),
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
    runButton.textContent = "Generate Reconstruction";
  }
}

function renderResult(data) {
  const s = data.summary;
  document.getElementById("countMetric").textContent = s.adjusted_count;
  document.getElementById("rawMetric").textContent = s.raw_candidates;
  document.getElementById("usableMetric").textContent = s.measurement_candidates;
  document.getElementById("diamMetric").textContent = s.median_diameter_cm_proxy;
  document.getElementById("volMetric").textContent = s.median_volume_cm3_proxy;
  document.getElementById("inputImage").src = data.input_image;
  document.getElementById("overlayImage").src = data.annotated_image;
  document.getElementById("extractionImage").src = data.extraction_overlay_image;
  document.getElementById("plotMapImage").src = data.plot_map_image;
  document.getElementById("depthImage").src = data.depth_image;
  document.getElementById("exportText").innerHTML = `<strong>PLY:</strong><br>${s.ply}<br><br><strong>CSV:</strong><br>${s.measurements_csv}`;
  cloud = data.points;
  renderCrops(data.boll_crops);
  renderTable(data.measurements);
  renderPlotCells(data.plot_cells);
  drawTopoLandscape(data.plot_cells);
  drawCloud();
}

function renderCrops(rows) {
  const gallery = document.getElementById("cropGallery");
  gallery.innerHTML = rows.map(r => `
    <div class="crop-card">
      <img src="${r.crop_image}" alt="extracted cotton boll ${r.id}" />
      <div class="crop-meta">
        <strong>#${r.id} | ${r.diameter_cm_proxy} cm</strong>
        vol ${r.volume_cm3_proxy} cm3<br>
        q ${r.extraction_quality} | lint ${r.lint_fraction}<br>
        green ${r.green_fraction} | depth ${r.depth_score}
      </div>
    </div>
  `).join("");
}

function renderTable(rows) {
  const table = document.getElementById("measureTable");
  const cols = ["id", "diameter_px", "diameter_cm_proxy", "volume_cm3_proxy", "extraction_quality", "lint_fraction", "green_fraction", "visibility_proxy", "depth_score"];
  table.innerHTML = `<thead><tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr></thead>` +
    `<tbody>${rows.map(r => `<tr>${cols.map(c => `<td>${r[c]}</td>`).join("")}</tr>`).join("")}</tbody>`;
}

function renderPlotCells(rows) {
  const table = document.getElementById("plotCellTable");
  const cols = ["row", "column", "boll_count", "mean_diameter_cm_proxy", "mean_volume_cm3_proxy", "mean_extraction_quality"];
  table.innerHTML = `<thead><tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr></thead>` +
    `<tbody>${rows.map(r => `<tr>${cols.map(c => `<td>${r[c]}</td>`).join("")}</tr>`).join("")}</tbody>`;
}

function drawTopoLandscape(rows) {
  const w = topoCanvas.width;
  const h = topoCanvas.height;
  topoCtx.clearRect(0, 0, w, h);
  topoCtx.fillStyle = "#fbfcfa";
  topoCtx.fillRect(0, 0, w, h);

  const gridRows = 4;
  const gridCols = 43;
  const byCell = new Map(rows.map(r => [`${r.row}-${r.column}`, r]));
  const maxCount = Math.max(1, ...rows.map(r => Number(r.boll_count) || 0));
  const tileW = Math.min(18, (w - 90) / (gridCols * 0.62));
  const tileH = tileW * 0.48;
  const originX = 44;
  const originY = 218;

  drawTopoLabel("Height = measurement-ready boll count per plot cell", 18, 28);
  drawTopoLabel("Image-coordinate proxy; needs orthomosaic/GCP/camera poses for meter-accurate mapping", 18, h - 18, "#687165");

  for (let r = gridRows; r >= 1; r--) {
    for (let c = gridCols; c >= 1; c--) {
      const cell = byCell.get(`${r}-${c}`) || { boll_count: 0, mean_extraction_quality: 0 };
      const count = Number(cell.boll_count) || 0;
      const q = Number(cell.mean_extraction_quality) || 0;
      const blockH = count === 0 ? 1.5 : 4 + (count / maxCount) * 82;
      const sx = originX + (c - 1) * tileW * 0.62 + (r - 1) * tileW * 0.34;
      const sy = originY + (r - 1) * tileH * 2.0 - (c - 1) * tileH * 0.08;
      const color = topoColor(count / maxCount, q);
      drawBlock(sx, sy, tileW, tileH, blockH, color);
    }
  }
}

function topoColor(t, q) {
  const low = [211, 222, 196];
  const high = [28, 154, 176];
  const boost = Math.max(0.45, Math.min(1, 0.55 + q));
  return low.map((v, i) => Math.round((v + (high[i] - v) * t) * boost));
}

function drawBlock(x, y, tw, th, bh, color) {
  const top = `rgb(${color[0]},${color[1]},${color[2]})`;
  const sideA = `rgb(${Math.max(0, color[0] - 34)},${Math.max(0, color[1] - 42)},${Math.max(0, color[2] - 36)})`;
  const sideB = `rgb(${Math.max(0, color[0] - 58)},${Math.max(0, color[1] - 60)},${Math.max(0, color[2] - 54)})`;
  const pTop = [[x, y - bh], [x + tw / 2, y - th - bh], [x + tw, y - bh], [x + tw / 2, y + th - bh]];
  const pRight = [[x + tw, y - bh], [x + tw / 2, y + th - bh], [x + tw / 2, y + th], [x + tw, y]];
  const pLeft = [[x, y - bh], [x + tw / 2, y + th - bh], [x + tw / 2, y + th], [x, y]];
  fillPoly(pLeft, sideA);
  fillPoly(pRight, sideB);
  fillPoly(pTop, top);
  strokePoly(pTop, "rgba(22,32,19,0.22)");
}

function fillPoly(points, fill) {
  topoCtx.beginPath();
  topoCtx.moveTo(points[0][0], points[0][1]);
  for (const p of points.slice(1)) topoCtx.lineTo(p[0], p[1]);
  topoCtx.closePath();
  topoCtx.fillStyle = fill;
  topoCtx.fill();
}

function strokePoly(points, stroke) {
  topoCtx.beginPath();
  topoCtx.moveTo(points[0][0], points[0][1]);
  for (const p of points.slice(1)) topoCtx.lineTo(p[0], p[1]);
  topoCtx.closePath();
  topoCtx.strokeStyle = stroke;
  topoCtx.lineWidth = 0.7;
  topoCtx.stroke();
}

function drawTopoLabel(text, x, y, color = "#162013") {
  topoCtx.fillStyle = color;
  topoCtx.font = "15px Inter, system-ui, sans-serif";
  topoCtx.fillText(text, x, y);
}

function drawCloud() {
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);
  const ca = Math.cos(angleZ), sa = Math.sin(angleZ);
  const cb = Math.cos(angleX), sb = Math.sin(angleX);
  const projected = cloud.map(p => {
    let x = p[0], y = p[1], z = p[2];
    let x1 = ca * x - sa * y;
    let y1 = sa * x + ca * y;
    let y2 = cb * y1 - sb * z;
    let z2 = sb * y1 + cb * z;
    return [w * 0.5 + x1 * w * 0.86, h * 0.58 - y2 * w * 0.86, z2, p[3], p[4], p[5]];
  }).sort((a, b) => a[2] - b[2]);
  for (const p of projected) {
    ctx.fillStyle = `rgb(${p[3]},${p[4]},${p[5]})`;
    ctx.fillRect(p[0], p[1], 1.8, 1.8);
  }
}

canvas.addEventListener("mousedown", e => { dragging = true; last = [e.clientX, e.clientY]; canvas.style.cursor = "grabbing"; });
window.addEventListener("mouseup", () => { dragging = false; canvas.style.cursor = "grab"; });
window.addEventListener("mousemove", e => {
  if (!dragging) return;
  angleZ += (e.clientX - last[0]) * 0.006;
  angleX += (e.clientY - last[1]) * 0.006;
  last = [e.clientX, e.clientY];
  drawCloud();
});

phase.addEventListener("change", updateChoices);
runButton.addEventListener("click", run);
loadDataset();
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
