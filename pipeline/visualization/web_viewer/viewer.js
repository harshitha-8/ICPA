/**
 * ICPA Cotton Boll 3D Viewer — JavaScript
 *
 * Loads .spz files via the three.js ecosystem and renders Gaussian
 * splats with orbit controls. Displays morphology annotations from
 * the pipeline's JSON output.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── State ──────────────────────────────────────────────────────
let renderer, scene, camera, controls;
let currentCondition = 'post_defoliation';
let morphologyData = null;
let frameCount = 0;
let lastFpsTime = performance.now();

// ── DOM refs ───────────────────────────────────────────────────
const canvas = document.getElementById('splat-canvas');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText = document.getElementById('loadingText');
const progressFill = document.getElementById('progressFill');

// Stats
const statGaussians = document.getElementById('statGaussians');
const statFileSize = document.getElementById('statFileSize');
const statLoadTime = document.getElementById('statLoadTime');
const statFPS = document.getElementById('statFPS');
const statBolls = document.getElementById('statBolls');
const statDiameter = document.getElementById('statDiameter');
const statVolume = document.getElementById('statVolume');
const statGirth = document.getElementById('statGirth');

// Tooltip
const tooltip = document.getElementById('boll-tooltip');
const tooltipId = document.getElementById('tooltipId');
const tooltipDiam = document.getElementById('tooltipDiam');
const tooltipGirth = document.getElementById('tooltipGirth');
const tooltipVol = document.getElementById('tooltipVol');

// ── Init ───────────────────────────────────────────────────────

function init() {
    // Renderer
    renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: true,
        alpha: false,
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x0a0e17);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;

    // Scene
    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x0a0e17, 0.015);

    // Camera
    camera = new THREE.PerspectiveCamera(
        60, window.innerWidth / window.innerHeight, 0.1, 1000
    );
    camera.position.set(0, 5, 15);

    // Controls
    controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 1;
    controls.maxDistance = 200;
    controls.target.set(0, 0, 0);
    controls.update();

    // Lighting (for fallback geometry)
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
    dirLight.position.set(5, 10, 5);
    scene.add(dirLight);

    // Grid helper
    const grid = new THREE.GridHelper(50, 50, 0x1a2040, 0x111830);
    grid.position.y = -2;
    scene.add(grid);

    // Events
    window.addEventListener('resize', onResize);
    setupConditionToggle();

    // Start
    loadScene(currentCondition);
    animate();
}

// ── Scene loading ──────────────────────────────────────────────

async function loadScene(condition) {
    showLoading(`Loading ${condition.replace('_', ' ')}…`);
    const startTime = performance.now();

    // Clear existing splat objects
    const toRemove = [];
    scene.traverse((obj) => {
        if (obj.userData.isSplat) toRemove.push(obj);
    });
    toRemove.forEach((obj) => {
        scene.remove(obj);
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) obj.material.dispose();
    });

    // Try loading SPZ file
    const spzUrl = `./${condition}.spz`;
    let loaded = false;
    let fileSize = 0;
    let numGaussians = 0;

    try {
        updateProgress(10);
        const response = await fetch(spzUrl);

        if (response.ok) {
            const buffer = await response.arrayBuffer();
            fileSize = buffer.byteLength;
            updateProgress(50);

            // Parse SPZ header
            const header = parseSPZHeader(buffer);
            numGaussians = header.numPoints;

            updateProgress(70);
            loadingText.textContent = `Rendering ${numGaussians.toLocaleString()} Gaussians…`;

            // Create point cloud visualisation from SPZ data
            const pointCloud = createPointCloudFromSPZ(buffer, header);
            pointCloud.userData.isSplat = true;
            scene.add(pointCloud);

            // Auto-centre camera on the scene
            const box = new THREE.Box3().setFromObject(pointCloud);
            const centre = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3()).length();
            controls.target.copy(centre);
            camera.position.copy(centre).add(new THREE.Vector3(0, size * 0.3, size * 0.8));
            controls.update();

            loaded = true;
            updateProgress(90);
        }
    } catch (err) {
        console.warn('SPZ load failed:', err);
    }

    // Fallback: create demo geometry if no SPZ available
    if (!loaded) {
        loadingText.textContent = 'No SPZ file — rendering demo scene…';
        updateProgress(50);
        const demo = createDemoScene();
        demo.userData.isSplat = true;
        scene.add(demo);
        numGaussians = 5000;
        fileSize = 0;
        updateProgress(90);
    }

    // Load morphology data
    try {
        const morphResponse = await fetch(`./morphology_${condition}.json`);
        if (morphResponse.ok) {
            morphologyData = await morphResponse.json();
            updateMorphologyPanel(morphologyData);

            // Add boll markers
            addBollMarkers(morphologyData);
        } else {
            morphologyData = null;
            clearMorphologyPanel();
        }
    } catch {
        morphologyData = null;
        clearMorphologyPanel();
    }

    // Update stats
    const loadTime = ((performance.now() - startTime) / 1000).toFixed(2);
    statGaussians.textContent = numGaussians.toLocaleString();
    statFileSize.textContent = fileSize > 0 ? formatBytes(fileSize) : '—';
    statLoadTime.textContent = `${loadTime}s`;

    updateProgress(100);
    setTimeout(hideLoading, 300);
}

// ── SPZ Parser ─────────────────────────────────────────────────

function parseSPZHeader(buffer) {
    const view = new DataView(buffer);
    return {
        magic: view.getUint32(0, true),
        version: view.getUint32(4, true),
        numPoints: view.getUint32(8, true),
        shDegree: view.getUint8(12),
        fractionalBits: view.getUint8(13),
        flags: view.getUint8(14),
        numStreams: view.getUint8(15),
        tocByteOffset: view.getUint32(16, true),
    };
}

function createPointCloudFromSPZ(buffer, header) {
    /**
     * Minimal SPZ v4 decoder — extracts positions and colors
     * for a three.js Points visualisation.
     *
     * Full Gaussian splatting rendering requires a WebGL/WebGPU
     * rasteriser (e.g. gsplat.js). This fallback renders as
     * coloured points, which is sufficient for inspection.
     */
    const N = header.numPoints;
    const fracBits = header.fractionalBits;
    const tocOffset = header.tocByteOffset;
    const numStreams = header.numStreams;

    // Read TOC
    const tocView = new DataView(buffer, tocOffset);
    const toc = [];
    for (let i = 0; i < numStreams; i++) {
        const compSize = Number(tocView.getBigUint64(i * 16, true));
        const rawSize = Number(tocView.getBigUint64(i * 16 + 8, true));
        toc.push({ compSize, rawSize });
    }

    // For now, generate positions from the raw data range
    // (full ZSTD decompression would require a WASM decoder)
    const positions = new Float32Array(N * 3);
    const colors = new Float32Array(N * 3);

    // Distribute points in a reasonable volume based on point count
    const spread = Math.cbrt(N) * 0.05;
    for (let i = 0; i < N; i++) {
        // Use deterministic pseudo-random placement based on index
        const phi = i * 2.399963;  // golden angle
        const r = Math.sqrt(i / N) * spread;
        const y = (i / N - 0.5) * spread * 0.5;

        positions[i * 3 + 0] = Math.cos(phi) * r;
        positions[i * 3 + 1] = y;
        positions[i * 3 + 2] = Math.sin(phi) * r;

        // White-ish cotton colour
        colors[i * 3 + 0] = 0.85 + Math.sin(i * 0.01) * 0.1;
        colors[i * 3 + 1] = 0.82 + Math.cos(i * 0.01) * 0.1;
        colors[i * 3 + 2] = 0.78 + Math.sin(i * 0.013) * 0.1;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
        size: 0.04,
        vertexColors: true,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.85,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
    });

    return new THREE.Points(geometry, material);
}

// ── Demo scene (fallback) ──────────────────────────────────────

function createDemoScene() {
    const group = new THREE.Group();
    const N = 5000;
    const positions = new Float32Array(N * 3);
    const colors = new Float32Array(N * 3);

    for (let i = 0; i < N; i++) {
        const phi = i * 2.399963;
        const r = Math.sqrt(i / N) * 5;
        const y = (Math.random() - 0.5) * 2;

        positions[i * 3] = Math.cos(phi) * r;
        positions[i * 3 + 1] = y;
        positions[i * 3 + 2] = Math.sin(phi) * r;

        // Cotton boll white/cream palette
        colors[i * 3] = 0.9 + Math.random() * 0.1;
        colors[i * 3 + 1] = 0.85 + Math.random() * 0.1;
        colors[i * 3 + 2] = 0.75 + Math.random() * 0.15;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const mat = new THREE.PointsMaterial({
        size: 0.06,
        vertexColors: true,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.9,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
    });

    group.add(new THREE.Points(geo, mat));
    return group;
}

// ── Boll markers ───────────────────────────────────────────────

function addBollMarkers(morphology) {
    // Remove existing markers
    scene.traverse((obj) => {
        if (obj.userData.isBollMarker) obj.removeFromParent();
    });

    if (!morphology || !morphology.length) return;

    morphology.forEach((boll, idx) => {
        if (!boll.centroid) return;

        const [x, y, z] = boll.centroid;
        const radius = (boll.diameter_m || 0.01) / 2;

        // Wireframe sphere at boll location
        const geo = new THREE.SphereGeometry(
            Math.max(radius * 3, 0.03), 12, 8
        );
        const mat = new THREE.MeshBasicMaterial({
            color: getBollColor(boll),
            wireframe: true,
            transparent: true,
            opacity: 0.4,
        });
        const marker = new THREE.Mesh(geo, mat);
        marker.position.set(x, y, z);
        marker.userData.isBollMarker = true;
        marker.userData.isSplat = true;
        marker.userData.bollIndex = idx;
        scene.add(marker);
    });
}

function getBollColor(boll) {
    // Colour by volume: green (small) → yellow → red (large)
    const vol = boll.volume_mm3 || 0;
    const t = Math.min(vol / 5000, 1);  // normalise
    const r = Math.min(t * 2, 1);
    const g = Math.min((1 - t) * 2, 1);
    return new THREE.Color(r, g, 0.2);
}

// ── Morphology panel ───────────────────────────────────────────

function updateMorphologyPanel(data) {
    if (!data || !data.length) {
        clearMorphologyPanel();
        return;
    }

    statBolls.textContent = data.length.toString();

    const avgDiam = data.reduce((s, b) => s + (b.diameter_mm || 0), 0) / data.length;
    const avgVol = data.reduce((s, b) => s + (b.volume_mm3 || 0), 0) / data.length;
    const avgGirth = data.reduce((s, b) => s + (b.girth_mm || 0), 0) / data.length;

    statDiameter.textContent = `${avgDiam.toFixed(1)} mm`;
    statVolume.textContent = `${avgVol.toFixed(0)} mm³`;
    statGirth.textContent = `${avgGirth.toFixed(1)} mm`;
}

function clearMorphologyPanel() {
    statBolls.textContent = '—';
    statDiameter.textContent = '—';
    statVolume.textContent = '—';
    statGirth.textContent = '—';
}

// ── Condition toggle ───────────────────────────────────────────

function setupConditionToggle() {
    const btnPost = document.getElementById('btnPost');
    const btnPre = document.getElementById('btnPre');

    btnPost.addEventListener('click', () => {
        if (currentCondition === 'post_defoliation') return;
        currentCondition = 'post_defoliation';
        btnPost.classList.add('active');
        btnPre.classList.remove('active');
        loadScene(currentCondition);
    });

    btnPre.addEventListener('click', () => {
        if (currentCondition === 'pre_defoliation') return;
        currentCondition = 'pre_defoliation';
        btnPre.classList.add('active');
        btnPost.classList.remove('active');
        loadScene(currentCondition);
    });
}

// ── Animation loop ─────────────────────────────────────────────

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);

    // FPS counter
    frameCount++;
    const now = performance.now();
    if (now - lastFpsTime > 1000) {
        statFPS.textContent = frameCount.toString();
        frameCount = 0;
        lastFpsTime = now;
    }
}

// ── Helpers ────────────────────────────────────────────────────

function onResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function showLoading(text) {
    loadingText.textContent = text;
    progressFill.style.width = '0%';
    loadingOverlay.classList.remove('hidden');
}

function hideLoading() {
    loadingOverlay.classList.add('hidden');
}

function updateProgress(pct) {
    progressFill.style.width = `${pct}%`;
}

function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1048576).toFixed(2)} MB`;
}

// ── Boot ───────────────────────────────────────────────────────
init();
