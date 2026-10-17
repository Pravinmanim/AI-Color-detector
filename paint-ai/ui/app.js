/**
 * Paint AI Studio — Frontend Client
 * Handles visual color selection, Kubelka-Munk recipe formulation,
 * batch mass calculations, and closed-loop corrections.
 */

// Global State
let currentTarget = {
    L: 55.3,
    a: 9.2,
    b: -65.3,
    hex: '#3b82f6',
    rgb: [59, 130, 246],
    name: 'Dodger Blue'
};

let currentRecipe = null;
let currentBatchGrams = 100.0;

const API_BASE = ""; // Relative path works seamlessly on any host:port

// ─── Initialization ───
document.addEventListener('DOMContentLoaded', () => {
    setupDragAndDrop();
    // Initialize default target display
    updateTargetDisplay(currentTarget.L, currentTarget.a, currentTarget.b, currentTarget.hex, currentTarget.rgb, currentTarget.name);
});

// ─── Tab Switching ───
function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));

    const activeBtn = document.querySelector(`[onclick*="${tabId}"]`);
    if (activeBtn) activeBtn.classList.add('active');

    const panel = document.getElementById(tabId);
    if (panel) panel.classList.add('active');
}

// ─── Visual Color Picker Handlers ───
let colorDebounce = null;
function onColorPickerChange(event) {
    const hex = event.target.value;
    document.getElementById('chip-hex').innerText = `HEX: ${hex.toUpperCase()}`;
    
    // Parse RGB
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    document.getElementById('chip-rgb').innerText = `RGB: ${r}, ${g}, ${b}`;

    clearTimeout(colorDebounce);
    colorDebounce = setTimeout(async () => {
        try {
            const res = await fetch(`${API_BASE}/api/rgb_to_lab`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ r, g, b })
            });
            if (res.ok) {
                const data = await res.json();
                updateTargetDisplay(data.L, data.a, data.b, hex, [r, g, b], 'Custom Pick');
            }
        } catch (e) {
            console.error("Color conversion error:", e);
        }
    }, 80);
}

async function setPresetColor(hex, name) {
    document.getElementById('native-color-picker').value = hex;
    document.getElementById('chip-hex').innerText = `HEX: ${hex.toUpperCase()}`;

    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    document.getElementById('chip-rgb').innerText = `RGB: ${r}, ${g}, ${b}`;

    try {
        const res = await fetch(`${API_BASE}/api/rgb_to_lab`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ r, g, b })
        });
        if (res.ok) {
            const data = await res.json();
            updateTargetDisplay(data.L, data.a, data.b, hex, [r, g, b], name);
            showToast(`Selected preset: ${name}`);
        }
    } catch (e) {
        console.error("Error setting preset:", e);
    }
}

// ─── Text / Hex / Name Resolution ───
function handleInputKey(event) {
    if (event.key === 'Enter') {
        resolveColorInput();
    }
}

async function resolveColorInput() {
    const input = document.getElementById('color-input').value.trim();
    if (!input) return showToast("Please enter a color name or hex code");

    const btn = document.getElementById('analyze-btn');
    btn.innerText = "Analyzing...";
    btn.disabled = true;

    const formData = new FormData();
    formData.append('name_or_hex', input);

    try {
        const res = await fetch(`${API_BASE}/api/resolve_name`, {
            method: 'POST',
            body: formData
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Colour not found");
        }

        const data = await res.json();
        updateTargetDisplay(data.L, data.a, data.b, data.hex, data.rgb, data.name || input);
        
        // Sync native picker
        if (data.hex) {
            document.getElementById('native-color-picker').value = data.hex;
            document.getElementById('chip-hex').innerText = `HEX: ${data.hex.toUpperCase()}`;
            document.getElementById('chip-rgb').innerText = `RGB: ${data.rgb.join(', ')}`;
        }

        showToast(`Resolved: ${data.name || input} (${data.source})`);
    } catch (e) {
        showToast(`Error: ${e.message}`);
    } finally {
        btn.innerText = "Analyze";
        btn.disabled = false;
    }
}

// ─── Image Eyedropper & Drag/Drop ───
function setupDragAndDrop() {
    const dropZone = document.getElementById('drop-zone');
    if (!dropZone) return;

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            handleImageFile(files[0]);
        }
    });
}

function handleImageUpload(event) {
    const file = event.target.files[0];
    if (file) {
        handleImageFile(file);
    }
}

async function handleImageFile(file) {
    if (!file.type.startsWith('image/')) {
        return showToast("Please upload an image file (PNG, JPG, etc.)");
    }

    // Show preview immediately
    const reader = new FileReader();
    reader.onload = (e) => {
        const previewContainer = document.getElementById('image-preview-container');
        const previewImg = document.getElementById('image-preview');
        previewImg.src = e.target.result;
        previewContainer.classList.remove('hidden');
    };
    reader.readAsDataURL(file);

    const overlay = document.getElementById('image-analyzing-overlay');
    overlay.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch(`${API_BASE}/api/extract_color`, {
            method: 'POST',
            body: formData
        });

        if (!res.ok) throw new Error("Could not extract color from image");
        const data = await res.json();

        const extractedName = data.name || "Extracted from Image";
        updateTargetDisplay(data.L, data.a, data.b, data.hex, data.rgb, extractedName);
        document.getElementById('native-color-picker').value = data.hex;
        document.getElementById('chip-hex').innerText = `HEX: ${data.hex.toUpperCase()}`;
        document.getElementById('chip-rgb').innerText = `RGB: ${data.rgb.join(', ')}`;

        showToast("Extracted dominant color from image!");
    } catch (e) {
        showToast(e.message);
    } finally {
        overlay.classList.add('hidden');
    }
}

// ─── Target Display Updater ───
function updateTargetDisplay(L, a, b, hex, rgb, name) {
    currentTarget = { L, a, b, hex, rgb, name };

    const swatch = document.getElementById('target-swatch');
    if (swatch) swatch.style.backgroundColor = hex;

    // Ambient glow aura
    const aura = document.getElementById('card-aura');
    if (aura) aura.style.background = `radial-gradient(circle, ${hex} 0%, transparent 70%)`;

    const nameEl = document.getElementById('target-name-display');
    if (nameEl) nameEl.innerText = name || hex;

    const hexEl = document.getElementById('target-hex-display');
    if (hexEl) hexEl.innerText = hex.toUpperCase();

    const labEl = document.getElementById('target-lab-display');
    if (labEl) labEl.innerText = `L*: ${L.toFixed(1)} | a*: ${a.toFixed(1)} | b*: ${b.toFixed(1)}`;

    // Sync Interactive RGB Sliders
    if (rgb && rgb.length === 3) {
        const [rVal, gVal, bVal] = rgb;
        const setVal = (id, v) => { const el = document.getElementById(id); if (el) el.value = v; };
        setVal('channel-r', rVal);
        setVal('channel-g', gVal);
        setVal('channel-b', bVal);
        setVal('num-r', rVal);
        setVal('num-g', gVal);
        setVal('num-b', bVal);
    }

    // Determine Tone Badge
    const toneBadge = document.getElementById('color-tone-badge');
    if (toneBadge) {
        const chroma = Math.sqrt(a * a + b * b);
        if (L >= 88) {
            toneBadge.innerText = "Pure Light Tint";
            toneBadge.style.color = "#ffffff";
            toneBadge.style.borderColor = "rgba(255, 255, 255, 0.4)";
        } else if (L <= 22) {
            toneBadge.innerText = "Deep Shadow Shade";
            toneBadge.style.color = "#94a3b8";
        } else if (chroma < 10) {
            toneBadge.innerText = "Neutral Earth Tone";
            toneBadge.style.color = "#cbd5e1";
        } else if (a > 12 && b > 12) {
            toneBadge.innerText = "Warm Terracotta / Gold";
            toneBadge.style.color = "#f59e0b";
        } else if (b < -10) {
            toneBadge.innerText = "Cool Blue Tone";
            toneBadge.style.color = "#38bdf8";
        } else if (a < -8 && b > 8) {
            toneBadge.innerText = "Natural Green Tone";
            toneBadge.style.color = "#34d399";
        } else {
            toneBadge.innerText = "Vibrant Studio Tone";
            toneBadge.style.color = "#a5b4fc";
        }
    }

    // Reset results if target changed
    const resSec = document.getElementById('results-section');
    if (resSec) resSec.classList.add('hidden');
    const corrSec = document.getElementById('correction-results');
    if (corrSec) corrSec.classList.add('hidden');
}

// ─── Formulate Recipe (Inverse Solver) ───
async function generateRecipe() {
    if (!currentTarget) return;

    const btn = document.getElementById('generate-btn');
    const btnText = document.getElementById('generate-btn-text');
    btnText.innerText = "Optimizing Physics Model...";
    btn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/api/predict_recipe`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                L: currentTarget.L,
                a: currentTarget.a,
                b: currentTarget.b
            })
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Recipe optimization failed");
        }

        const data = await res.json();
        currentRecipe = data;

        displayResults(data);
        showToast("Recipe calculated successfully!");
    } catch (e) {
        showToast(`Error: ${e.message}`);
    } finally {
        btnText.innerText = "Formulate AI Recipe";
        btn.disabled = false;
    }
}

// ─── Display Results & Swatch Comparison ───
function displayResults(data) {
    const resultsSec = document.getElementById('results-section');
    resultsSec.classList.remove('hidden');

    // 1. Swatch Comparison
    const targetHex = data.target_hex || currentTarget.hex;
    const predHex = data.predicted_hex || currentTarget.hex;

    document.getElementById('cmp-target-swatch').style.backgroundColor = targetHex;
    document.getElementById('cmp-target-hex').innerText = targetHex.toUpperCase();

    document.getElementById('cmp-pred-swatch').style.backgroundColor = predHex;
    document.getElementById('cmp-pred-hex').innerText = predHex.toUpperCase();

    // 2. Delta E00 and Quality Rating
    const de00 = data.predicted_deltaE00;
    const de00El = document.getElementById('de00-val');
    de00El.innerText = de00.toFixed(2);

    const qualityEl = document.getElementById('quality-label');
    const statusEl = document.getElementById('status-val');
    const pill = document.getElementById('delta-pill');

    if (de00 < 1.0) {
        qualityEl.innerText = "🌟 Imperceptible Match";
        qualityEl.style.color = "#10b981";
        pill.style.borderColor = "rgba(16, 185, 129, 0.4)";
        pill.style.background = "rgba(16, 185, 129, 0.15)";
        statusEl.innerText = "Production Quality (ΔE₀₀ < 1.0)";
    } else if (de00 < 2.0) {
        qualityEl.innerText = "✅ Excellent Commercial Match";
        qualityEl.style.color = "#38bdf8";
        pill.style.borderColor = "rgba(56, 189, 248, 0.4)";
        pill.style.background = "rgba(56, 189, 248, 0.15)";
        statusEl.innerText = "Within Commercial Tolerance (ΔE₀₀ < 2.0)";
    } else if (de00 <= 2.5) {
        qualityEl.innerText = "⚠️ Acceptable Match";
        qualityEl.style.color = "#f59e0b";
        pill.style.borderColor = "rgba(245, 158, 11, 0.4)";
        pill.style.background = "rgba(245, 158, 11, 0.15)";
        statusEl.innerText = "Slight Visual Shift (ΔE₀₀ ≤ 2.5)";
    } else {
        qualityEl.innerText = "🚫 Outside 4-Pigment Gamut";
        qualityEl.style.color = "#f43f5e";
        pill.style.borderColor = "rgba(244, 63, 94, 0.4)";
        pill.style.background = "rgba(244, 63, 94, 0.15)";
        statusEl.innerText = "Closest Achievable Match";
    }

    // 3. Render Recipe Bars & Mass Breakdown
    renderRecipeBars(data.recipe_RBYW);

    // Scroll smoothly to results
    resultsSec.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ─── Batch Gram Calculations ───
function setBatchTotal(grams) {
    currentBatchGrams = grams;
    document.getElementById('batch-total-input').value = grams;

    document.querySelectorAll('.size-btn').forEach(btn => {
        btn.classList.toggle('active', btn.innerText.replace(' kg', '000').replace('g', '') == grams);
    });

    if (currentRecipe) {
        renderRecipeBars(currentRecipe.recipe_RBYW);
    }
}

function updateBatchMasses() {
    const inputVal = parseFloat(document.getElementById('batch-total-input').value);
    if (!isNaN(inputVal) && inputVal > 0) {
        currentBatchGrams = inputVal;
        if (currentRecipe) {
            renderRecipeBars(currentRecipe.recipe_RBYW);
        }
    }
}

function renderRecipeBars(recipe) {
    const [rPct, bPct, yPct, wPct] = recipe;
    const grid = document.getElementById('recipe-grid');
    grid.innerHTML = '';

    const pigments = [
        { name: 'Cadmium Red', pct: rPct, fillClass: 'fill-red', dotClass: 'dot-red' },
        { name: 'Phthalo Blue', pct: bPct, fillClass: 'fill-blue', dotClass: 'dot-blue' },
        { name: 'Primary Yellow', pct: yPct, fillClass: 'fill-yellow', dotClass: 'dot-yellow' },
        { name: 'Titanium White', pct: wPct, fillClass: 'fill-white', dotClass: 'dot-white' },
    ];

    pigments.forEach(item => {
        const grams = (item.pct / 100.0) * currentBatchGrams;
        const row = document.createElement('div');
        row.className = 'recipe-bar-item';
        row.innerHTML = `
            <div class="bar-item-header">
                <div class="pigment-tag">
                    <span class="pigment-dot ${item.dotClass}"></span>
                    <span>${item.name}</span>
                </div>
                <div class="mass-stats">
                    <span class="mass-grams">${grams.toFixed(2)} g</span>
                    <span class="mass-pct">(${item.pct.toFixed(1)}%)</span>
                </div>
            </div>
            <div class="bar-track">
                <div class="bar-fill ${item.fillClass}" style="width: 0%"></div>
            </div>
        `;
        grid.appendChild(row);

        // Animate bar width
        setTimeout(() => {
            row.querySelector('.bar-fill').style.width = `${Math.min(100, Math.max(0, item.pct))}%`;
        }, 40);
    });
}

// ─── Closed-Loop Batch Correction ───
function prefillCorrectionWithRecipe() {
    if (!currentRecipe) return;
    const [rPct, bPct, yPct, wPct] = currentRecipe.recipe_RBYW;
    document.getElementById('corr-r').value = ((rPct / 100.0) * currentBatchGrams).toFixed(1);
    document.getElementById('corr-b').value = ((bPct / 100.0) * currentBatchGrams).toFixed(1);
    document.getElementById('corr-y').value = ((yPct / 100.0) * currentBatchGrams).toFixed(1);
    document.getElementById('corr-w').value = ((wPct / 100.0) * currentBatchGrams).toFixed(1);
    showToast("Filled with suggested recipe amounts. Adjust with what you measured!");
}

async function calculateCorrection() {
    if (!currentTarget) return;

    const r = parseFloat(document.getElementById('corr-r').value) || 0;
    const b = parseFloat(document.getElementById('corr-b').value) || 0;
    const y = parseFloat(document.getElementById('corr-y').value) || 0;
    const w = parseFloat(document.getElementById('corr-w').value) || 0;

    const total = r + b + y + w;
    if (total <= 0) {
        return showToast("Please enter the grams currently mixed in your container");
    }

    const btn = document.getElementById('correct-btn');
    btn.innerText = "Calculating Non-Destructive Additions...";
    btn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/api/correct_recipe`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                current_R_g: r,
                current_B_g: b,
                current_Y_g: y,
                current_W_g: w,
                target_L: currentTarget.L,
                target_a: currentTarget.a,
                target_b: currentTarget.b
            })
        });

        if (!res.ok) throw new Error("Correction solver failed");

        const data = await res.json();
        displayCorrectionResults(data);
        showToast("Calculated optimal pigment additions!");
    } catch (e) {
        showToast(e.message);
    } finally {
        btn.innerText = "Calculate Additions (Add-Only)";
        btn.disabled = false;
    }
}

function displayCorrectionResults(data) {
    const resultsContainer = document.getElementById('correction-results');
    resultsContainer.classList.remove('hidden');

    const adds = data.additions_g_RBYW;
    const labels = [
        { name: 'Red', val: adds[0], dot: 'dot-red' },
        { name: 'Blue', val: adds[1], dot: 'dot-blue' },
        { name: 'Yellow', val: adds[2], dot: 'dot-yellow' },
        { name: 'White', val: adds[3], dot: 'dot-white' }
    ];

    const totalAdded = adds.reduce((a, b) => a + b, 0);

    let html = `
        <div class="additions-title">
            <span>✨ Recommended Additions (Total: +${totalAdded.toFixed(2)} g)</span>
        </div>
        <div class="additions-grid">
    `;

    labels.forEach(item => {
        const isAdd = item.val >= 0.05;
        html += `
            <div class="addition-card">
                <div class="addition-pigment">
                    <span class="dot ${item.dot}"></span> ${item.name}
                </div>
                <div class="addition-amount ${isAdd ? 'has-add' : ''}">
                    ${isAdd ? `+${item.val.toFixed(2)} g` : '0.00 g'}
                </div>
            </div>
        `;
    });

    html += `</div>`;
    resultsContainer.innerHTML = html;
}

// ─── Live Webcam Eyedropper ───
let webcamStream = null;
let webcamInterval = null;

async function startWebcam() {
    const video = document.getElementById('webcam-video');
    const placeholder = document.getElementById('camera-placeholder');
    const toggleBtn = document.getElementById('btn-toggle-cam');
    const sampleBtn = document.getElementById('btn-sample-cam');
    const readoutRow = document.getElementById('cam-readout-row');
    const calibControls = document.getElementById('cam-calibration-controls');

    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "environment" }
        });
        video.srcObject = webcamStream;
        placeholder.style.display = 'none';
        toggleBtn.style.display = 'inline-block';
        sampleBtn.style.display = 'inline-block';
        readoutRow.style.display = 'flex';
        if (calibControls) calibControls.style.display = 'flex';

        // Start live reticle sampling loop
        startLiveSamplingLoop();
        showToast("Webcam active! Center the color in the reticle.");
    } catch (err) {
        console.error("Camera access error:", err);
        showToast(`Camera error: ${err.message || 'Permission denied'}`);
    }
}

function toggleWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    if (webcamInterval) {
        clearInterval(webcamInterval);
        webcamInterval = null;
    }

    const video = document.getElementById('webcam-video');
    video.srcObject = null;
    document.getElementById('camera-placeholder').style.display = 'flex';
    document.getElementById('btn-toggle-cam').style.display = 'none';
    document.getElementById('btn-sample-cam').style.display = 'none';
    document.getElementById('cam-readout-row').style.display = 'none';
    const calibControls = document.getElementById('cam-calibration-controls');
    if (calibControls) calibControls.style.display = 'none';
    showToast("Webcam stopped.");
}

// ─── Color Calibration Engine ───
let whitePoint = { r: 255, g: 255, b: 255 };
let isWhiteCalibrated = false;
let exposureMultiplier = 1.25;
let saturationMultiplier = 1.15;
let latestRawSample = [128, 128, 128];

function onExposureChange(event) {
    exposureMultiplier = parseFloat(event.target.value);
    document.getElementById('exposure-val').innerText = `${exposureMultiplier.toFixed(2)}x`;
}

function onSaturationChange(event) {
    saturationMultiplier = parseFloat(event.target.value);
    document.getElementById('saturation-val').innerText = `${saturationMultiplier.toFixed(2)}x`;
}

function applyColorCalibration(rawR, rawG, rawB) {
    // 1. Von Kries chromatic adaptation against calibrated white reference
    let r = rawR * (255.0 / whitePoint.r) * exposureMultiplier;
    let g = rawG * (255.0 / whitePoint.g) * exposureMultiplier;
    let b = rawB * (255.0 / whitePoint.b) * exposureMultiplier;

    // 2. Saturation / Vibrancy compensation (counteracts gloomy camera desaturation)
    const gray = 0.299 * r + 0.587 * g + 0.114 * b;
    r = gray + (r - gray) * saturationMultiplier;
    g = gray + (g - gray) * saturationMultiplier;
    b = gray + (b - gray) * saturationMultiplier;

    return [
        Math.min(255, Math.max(0, Math.round(r))),
        Math.min(255, Math.max(0, Math.round(g))),
        Math.min(255, Math.max(0, Math.round(b)))
    ];
}

function calibrateWhitePoint() {
    if (!latestRawSample) return showToast("Start camera and point at white paper first.");

    const [rawR, rawG, rawB] = latestRawSample;
    whitePoint = {
        r: Math.max(rawR, 35),
        g: Math.max(rawG, 35),
        b: Math.max(rawB, 35)
    };
    isWhiteCalibrated = true;

    const badgeText = document.getElementById('calibration-text');
    if (badgeText) badgeText.innerText = `White Reference Calibrated (${whitePoint.r}, ${whitePoint.g}, ${whitePoint.b})`;

    showToast("White Point calibrated! Gloominess eliminated.");
}

function resetWhitePoint() {
    whitePoint = { r: 255, g: 255, b: 255 };
    isWhiteCalibrated = false;
    exposureMultiplier = 1.25;
    saturationMultiplier = 1.15;
    document.getElementById('cam-exposure-slider').value = 1.25;
    document.getElementById('cam-saturation-slider').value = 1.15;
    document.getElementById('exposure-val').innerText = '1.25x';
    document.getElementById('saturation-val').innerText = '1.15x';

    const badgeText = document.getElementById('calibration-text');
    if (badgeText) badgeText.innerText = "Lighting: Auto Compensated";

    showToast("Calibration reset to standard auto.");
}

function startLiveSamplingLoop() {
    if (webcamInterval) clearInterval(webcamInterval);

    const video = document.getElementById('webcam-video');
    const canvas = document.getElementById('webcam-canvas');
    const ctx = canvas.getContext('2d');

    webcamInterval = setInterval(() => {
        if (!video.videoWidth || !video.videoHeight) return;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Center 24x24 box
        const cx = Math.floor(canvas.width / 2);
        const cy = Math.floor(canvas.height / 2);
        const boxSize = 24;
        const imgData = ctx.getImageData(cx - boxSize / 2, cy - boxSize / 2, boxSize, boxSize);

        let totalR = 0, totalG = 0, totalB = 0;
        const totalPixels = boxSize * boxSize;
        for (let i = 0; i < imgData.data.length; i += 4) {
            totalR += imgData.data[i];
            totalG += imgData.data[i + 1];
            totalB += imgData.data[i + 2];
        }

        const rawR = Math.round(totalR / totalPixels);
        const rawG = Math.round(totalG / totalPixels);
        const rawB = Math.round(totalB / totalPixels);
        latestRawSample = [rawR, rawG, rawB];

        // Apply real-time calibration
        const [r, g, b] = applyColorCalibration(rawR, rawG, rawB);
        const hex = `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`.toUpperCase();

        document.getElementById('cam-rgb-chip').innerText = `RGB: (${r}, ${g}, ${b})`;
        document.getElementById('cam-hex-chip').innerText = `HEX: ${hex}`;
        
        if (!window.lastNameFetch || window.lastNameFetch !== hex) {
            window.lastNameFetch = hex;
            fetch(`${API_BASE}/api/rgb_to_lab`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ r, g, b })
            })
            .then(res => res.json())
            .then(data => {
                const nameEl = document.getElementById('cam-name-chip');
                if (nameEl && data.name) {
                    nameEl.innerText = `Name: ${data.name}`;
                }
            })
            .catch(err => console.error(err));
        }
    }, 100);
}

async function sampleWebcamColor() {
    if (!latestRawSample) {
        return showToast("Please start the webcam first.");
    }

    const [r, g, b] = applyColorCalibration(latestRawSample[0], latestRawSample[1], latestRawSample[2]);
    const hex = `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;

    try {
        const res = await fetch(`${API_BASE}/api/rgb_to_lab`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ r, g, b })
        });
        if (res.ok) {
            const data = await res.json();
            let sampleName = data.name || "Camera Sample";
            if (isWhiteCalibrated) sampleName += " (Calibrated)";
            updateTargetDisplay(data.L, data.a, data.b, hex, [r, g, b], sampleName);
            document.getElementById('native-color-picker').value = hex;
            document.getElementById('chip-hex').innerText = `HEX: ${hex.toUpperCase()}`;
            document.getElementById('chip-rgb').innerText = `RGB: ${r}, ${g}, ${b}`;
            showToast(`Captured Color: RGB(${r}, ${g}, ${b}) -> ${hex.toUpperCase()}`);
        }
    } catch (e) {
        console.error("Color sample error:", e);
    }
}

// ─── Interactive Card RGB Channel Handlers ───
let channelDebounce = null;

function onChannelChange() {
    const r = parseInt(document.getElementById('channel-r').value, 10);
    const g = parseInt(document.getElementById('channel-g').value, 10);
    const b = parseInt(document.getElementById('channel-b').value, 10);

    document.getElementById('num-r').value = r;
    document.getElementById('num-g').value = g;
    document.getElementById('num-b').value = b;

    syncRGBToDisplay(r, g, b);
}

function onChannelNumChange() {
    let r = parseInt(document.getElementById('num-r').value, 10) || 0;
    let g = parseInt(document.getElementById('num-g').value, 10) || 0;
    let b = parseInt(document.getElementById('num-b').value, 10) || 0;

    r = Math.min(255, Math.max(0, r));
    g = Math.min(255, Math.max(0, g));
    b = Math.min(255, Math.max(0, b));

    document.getElementById('channel-r').value = r;
    document.getElementById('channel-g').value = g;
    document.getElementById('channel-b').value = b;

    syncRGBToDisplay(r, g, b);
}

function syncRGBToDisplay(r, g, b) {
    const hex = `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
    document.getElementById('native-color-picker').value = hex;
    document.getElementById('chip-hex').innerText = `HEX: ${hex.toUpperCase()}`;
    document.getElementById('chip-rgb').innerText = `RGB: ${r}, ${g}, ${b}`;

    clearTimeout(channelDebounce);
    channelDebounce = setTimeout(async () => {
        try {
            const res = await fetch(`${API_BASE}/api/rgb_to_lab`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ r, g, b })
            });
            if (res.ok) {
                const data = await res.json();
                updateTargetDisplay(data.L, data.a, data.b, hex, [r, g, b], "Interactive Custom");
            }
        } catch (e) {
            console.error("Sync error:", e);
        }
    }, 60);
}

function autoBrightenGloomyColor() {
    if (!currentTarget || !currentTarget.rgb) return;
    let [r, g, b] = currentTarget.rgb;

    // Detect if color is gloomy (low brightness)
    const maxVal = Math.max(r, g, b, 1);
    if (maxVal < 220) {
        // Boost towards full reflectance
        const factor = Math.min(255 / maxVal, 1.6);
        r = Math.min(255, Math.round(r * factor));
        g = Math.min(255, Math.round(g * factor));
        b = Math.min(255, Math.round(b * factor));

        syncRGBToDisplay(r, g, b);
        showToast("☀️ Boosted exposure to eliminate gloomy indoor lighting!");
    } else {
        showToast("Color is already at high reflectance.");
    }
}

function copyHex() {
    if (!currentTarget || !currentTarget.hex) return;
    navigator.clipboard.writeText(currentTarget.hex.toUpperCase()).then(() => {
        showToast(`Copied ${currentTarget.hex.toUpperCase()} to clipboard!`);
    }).catch(() => {
        showToast(`Hex: ${currentTarget.hex.toUpperCase()}`);
    });
}

// ─── Toast Helper ───
let toastTimer = null;
function showToast(message, duration = 3500) {
    const toast = document.getElementById('toast');
    if (!toast) return;

    toast.innerText = message;
    toast.classList.remove('hidden');

    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
        toast.classList.add('hidden');
    }, duration);
}

