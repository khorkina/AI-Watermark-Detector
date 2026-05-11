"""
AI Watermark Detector — "Is This AI?"
Hacker / cryptography aesthetic. Port 5000.

Detection signals:
  1. Noise Level Estimation        (Laplacian sigma)          — 30 %
  2. Error Level Analysis (ELA)                               — 20 %
  3. Source Format Forensics                                  — 15 %
  4. DCT Block Structure                                      — 10 %
  5. SynthID Carrier Phase Coherence  (reverse-SynthID)       — 25 %
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AI Watermark Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MAX_FILE_SIZE_MB = 10
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "webp"]

# ══════════════════════════════════════════════════════════════════════════════
# HACKER CSS
# ══════════════════════════════════════════════════════════════════════════════
HACKER_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');
html,body,[class*="css"]{background-color:#000!important;color:#c8ffc8!important;font-family:'Share Tech Mono','Courier New',monospace!important;}
body::before{content:"";position:fixed;top:0;left:0;width:100%;height:100%;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,255,80,.012) 2px,rgba(0,255,80,.012) 4px);pointer-events:none;z-index:9999;}
.stApp,[data-testid="stAppViewContainer"]{background:#000!important;}
[data-testid="stHeader"]{background:transparent!important;}
section[data-testid="stMain"]>div{background:#000!important;}
[data-testid="stTabs"] button{color:#00cc55!important;font-family:'Share Tech Mono',monospace!important;font-size:.9rem!important;letter-spacing:2px!important;background:transparent!important;border-bottom:2px solid transparent!important;}
[data-testid="stTabs"] button[aria-selected="true"]{color:#00ff88!important;border-bottom:2px solid #00ff88!important;text-shadow:0 0 10px #00ff88,0 0 20px #00ff4488!important;}
[data-testid="stTabs"] [role="tablist"]{border-bottom:1px solid #00ff2222!important;background:transparent!important;}
[data-testid="stFileUploader"]{border:1px solid #00ff4433!important;background:#000e04!important;box-shadow:0 0 20px #00ff441a,inset 0 0 30px #00ff440a!important;}
[data-testid="stFileUploadDropzone"]{background:#000e04!important;border:2px dashed #00ff4455!important;}
[data-testid="stFileUploadDropzone"]:hover{border-color:#00ff88!important;box-shadow:0 0 25px #00ff4433!important;}
[data-testid="stProgressBar"]>div{background:#001a0a!important;border:1px solid #00ff4422!important;}
[data-testid="stProgressBar"]>div>div{background:linear-gradient(90deg,#005522,#00ff88)!important;box-shadow:0 0 8px #00ff8888!important;}
[data-testid="stExpander"]{border:1px solid #00ff2222!important;background:#000a04!important;}
details summary{color:#00ff88!important;font-family:'Share Tech Mono',monospace!important;}
hr{border-color:#00ff2222!important;box-shadow:0 0 6px #00ff2211!important;}
h1,h2,h3{font-family:'Orbitron','Share Tech Mono',monospace!important;text-shadow:0 0 10px #00ff8877,0 0 30px #00ff4433!important;letter-spacing:2px!important;}
h1{color:#ffffff!important;}h2{color:#00ff88!important;}h3{color:#c0ffc0!important;font-size:1.05rem!important;}
p,li{color:#a0e8a0!important;font-family:'Share Tech Mono',monospace!important;}
strong{color:#00ff88!important;}
code{background:#001a0a!important;color:#00ff88!important;border:1px solid #00ff2233!important;border-radius:2px!important;font-family:'Share Tech Mono',monospace!important;}
pre,.stCode{background:#000d04!important;border:1px solid #00ff2233!important;border-left:3px solid #00ff88!important;box-shadow:0 0 20px #00ff221a!important;}
a{color:#00ff88!important;}a:hover{text-shadow:0 0 8px #00ff88!important;}
table{border-collapse:collapse!important;font-family:'Share Tech Mono',monospace!important;}
th{background:#001a08!important;color:#00ff88!important;border:1px solid #00ff2233!important;padding:6px 12px!important;}
td{border:1px solid #00ff2222!important;color:#88cc88!important;padding:5px 12px!important;}
tr:hover td{background:#001008!important;}
figcaption,[data-testid="stCaptionContainer"]{color:#336633!important;font-size:.73rem!important;letter-spacing:1px!important;font-family:'Share Tech Mono',monospace!important;}
[data-testid="stAlert"]{font-family:'Share Tech Mono',monospace!important;}
</style>
"""

# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL 1 — Noise Level
# ══════════════════════════════════════════════════════════════════════════════
def estimate_noise_level(img: Image.Image) -> dict:
    from scipy.ndimage import convolve
    gray = np.array(img.convert("L"), dtype=np.float32)
    kernel = np.array([[1,-2,1],[-2,4,-2],[1,-2,1]], dtype=np.float32)
    laplacian = convolve(gray, kernel)
    sigma = float(np.sqrt(np.pi / 2) * np.abs(laplacian).mean() / 6.0)
    ai_score = float(1.0 / (1.0 + np.exp(0.6 * (sigma - 4.0))))
    return {"sigma": sigma, "ai_score": float(np.clip(ai_score, 0, 1))}

# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL 2 — Error Level Analysis
# ══════════════════════════════════════════════════════════════════════════════
def analyze_ela(img: Image.Image) -> dict:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")
    orig = np.array(img, dtype=np.float32)
    rsav = np.array(resaved, dtype=np.float32)
    ela_map = np.abs(orig - rsav)
    ela_mean = float(ela_map.mean())
    ai_score = float(1.0 / (1.0 + np.exp(-0.5 * (ela_mean - 7.0))))
    return {
        "ela_mean": ela_mean,
        "ela_std": float(ela_map.std()),
        "ela_map": ela_map,
        "ai_score": float(np.clip(ai_score, 0, 1)),
    }

# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL 3 — Source Format
# ══════════════════════════════════════════════════════════════════════════════
def analyze_source_format(mime_type: str) -> dict:
    if mime_type in ("image/jpeg", "image/jpg"):
        return {"fmt_label": "JPEG", "ai_score": 0.25}
    elif mime_type == "image/webp":
        return {"fmt_label": "WebP", "ai_score": 0.62}
    elif mime_type == "image/png":
        return {"fmt_label": "PNG",  "ai_score": 0.67}
    return {"fmt_label": "Unknown", "ai_score": 0.50}

# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL 4 — DCT Block Detection
# ══════════════════════════════════════════════════════════════════════════════
def compute_fft_spectrum(img: Image.Image):
    gray = np.array(img.convert("L"), dtype=np.float32)
    h, w = gray.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    fft_shifted = np.fft.fftshift(np.fft.fft2(gray * window))
    magnitude = np.abs(fft_shifted)
    return magnitude, np.log1p(magnitude)

def detect_dct_blocks(magnitude: np.ndarray, img_size: tuple) -> dict:
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    peak_ratios = []
    for axis in ["vertical", "horizontal"]:
        dim = h if axis == "vertical" else w
        center = cy if axis == "vertical" else cx
        for k in range(1, 4):
            offset = k * dim // 8
            if offset < 4 or offset > dim // 2 - 6:
                continue
            for pos in [center + offset, center - offset]:
                if not (4 <= pos < dim - 4):
                    continue
                if axis == "vertical":
                    peak_val = float(magnitude[pos-2:pos+3, cx-2:cx+3].max())
                    bg_val   = float(magnitude[pos-8:pos+9, cx-2:cx+3].mean())
                else:
                    peak_val = float(magnitude[cy-2:cy+3, pos-2:pos+3].max())
                    bg_val   = float(magnitude[cy-2:cy+3, pos-8:pos+9].mean())
                if bg_val > 0:
                    peak_ratios.append(peak_val / bg_val)
    if not peak_ratios:
        dct_strength = 0.0
    else:
        avg_ratio = float(np.mean(peak_ratios))
        dct_strength = float(np.clip((avg_ratio - 1.0) / 6.0, 0, 1))
    return {
        "dct_strength": dct_strength,
        "avg_ratio": float(np.mean(peak_ratios)) if peak_ratios else 1.0,
        "ai_score": float(np.clip(1.0 - dct_strength, 0, 1)),
    }

# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL 5 — SynthID Carrier Phase Coherence  (reverse-SynthID algorithm)
# ══════════════════════════════════════════════════════════════════════════════
def detect_synthid_watermark(img: Image.Image) -> dict:
    """
    Reverse-SynthID detection based on carrier frequency phase coherence.

    SynthID (Google Gemini's invisible watermark) embeds fixed-phase carriers
    at resolution-dependent frequency bins in the image's noise residual.

    Key property: carriers are image-content-independent — their phase is
    identical across every SynthID image, while real image content produces
    random phases. Cross-image coherence of SynthID carriers reaches 99.5%.

    For single-image detection we measure:
      1. Cross-channel phase coherence at top-magnitude noise-residual bins
         (R, G, B weights: 0.85, 1.0, 0.70 — from reverse-engineering SynthID)
      2. Spatial patch phase stability across quadrants
      3. FFT conjugate symmetry of candidate carrier bins

    Reference: github.com/aloshdenny/reverse-SynthID
    """
    from scipy.ndimage import uniform_filter

    arr = np.array(img, dtype=np.float32)  # H x W x 3 (RGB)
    H, W, _ = arr.shape

    # ── Channel weights recovered from reverse-SynthID project ───────────────
    CH_WEIGHTS = np.array([0.85, 1.0, 0.70], dtype=np.float32)   # R, G, B

    # ── Step 1: Extract noise residual  (image − box_blur) ───────────────────
    blur_size = 7
    noise = np.zeros_like(arr)
    for c in range(3):
        blurred = uniform_filter(arr[:, :, c], size=blur_size)
        noise[:, :, c] = (arr[:, :, c] - blurred) * CH_WEIGHTS[c]

    # ── Step 2: 2-D FFT of noise residual per channel ─────────────────────────
    fft_ch = []
    for c in range(3):
        fft_c = np.fft.fftshift(np.fft.fft2(noise[:, :, c]))
        fft_ch.append(fft_c)

    mag_g = np.abs(fft_ch[1])     # Green is dominant channel

    # ── Step 3: Candidate carrier bins — top-K magnitude outside DC ───────────
    cy, cx = H // 2, W // 2
    dc_h = max(H // 12, 8)
    dc_w = max(W // 12, 8)
    mag_search = mag_g.copy()
    mag_search[cy - dc_h : cy + dc_h, cx - dc_w : cx + dc_w] = 0

    K = min(128, max(16, H * W // 256))
    flat_idx = np.argpartition(mag_search.ravel(), -K)[-K:]
    top_ys, top_xs = np.unravel_index(flat_idx, (H, W))

    # ── Step 4: Cross-channel phase coherence ─────────────────────────────────
    # SynthID embeds the same phase at carrier bins regardless of content.
    # Real images: phases at high-energy bins scatter randomly across channels.
    # SynthID images: R-G and B-G phase differences are highly consistent.
    all_phases = np.zeros((3, K), dtype=np.float32)
    for c in range(3):
        fft = fft_ch[c]
        for j, (y, x) in enumerate(zip(top_ys, top_xs)):
            all_phases[c, j] = np.angle(fft[y, x])

    rg_phasors = np.exp(1j * (all_phases[0] - all_phases[1]))
    bg_phasors = np.exp(1j * (all_phases[2] - all_phases[1]))
    rg_coh  = float(abs(rg_phasors.mean()))
    bg_coh  = float(abs(bg_phasors.mean()))
    # Weight by channel strength
    cross_coherence = float((rg_coh * 0.85 + bg_coh * 0.70) / 1.55)

    # ── Step 5: Spatial patch phase coherence ─────────────────────────────────
    # Divide image into patches; the dominant FFT peak in each patch should
    # maintain consistent phase if a global periodic carrier is present.
    n_div = min(4, H // 64, W // 64)
    if n_div >= 2:
        ph, pw = H // n_div, W // n_div
        patch_top_phases = []
        for pi in range(n_div):
            for pj in range(n_div):
                p = noise[pi*ph:(pi+1)*ph, pj*pw:(pj+1)*pw, 1]
                p_fft = np.fft.fftshift(np.fft.fft2(p))
                p_mag = np.abs(p_fft)
                pcy, pcx = ph // 2, pw // 2
                p_dc_h, p_dc_w = max(ph // 10, 2), max(pw // 10, 2)
                p_mag[pcy-p_dc_h:pcy+p_dc_h, pcx-p_dc_w:pcx+p_dc_w] = 0
                if p_mag.max() > 1e-6:
                    pk = np.argmax(p_mag.ravel())
                    py, px = np.unravel_index(pk, p_fft.shape)
                    patch_top_phases.append(np.angle(p_fft[py, px]))
        if len(patch_top_phases) > 1:
            sp_phasors = np.exp(1j * np.array(patch_top_phases))
            spatial_coherence = float(abs(sp_phasors.mean()))
        else:
            spatial_coherence = 0.5
    else:
        spatial_coherence = 0.5

    # ── Step 6: Conjugate symmetry of candidate carriers ─────────────────────
    # FFT of a real signal satisfies F(y,x) = conj(F(H-y, W-x)).
    # A synthesised carrier introduces perfectly symmetric pairs.
    # We score how well the top bins satisfy this symmetry.
    sym_scores = []
    for y, x in zip(top_ys[:48], top_xs[:48]):
        my = (H - y) % H
        mx = (W - x) % W
        for c in range(3):
            v1 = fft_ch[c][y, x]
            v2 = np.conj(fft_ch[c][my, mx])
            mag_sum = abs(v1) + abs(v2)
            if mag_sum > 1e-6:
                # Normalised agreement between v1 and v2
                sym_scores.append(1.0 - abs(v1 - v2) / (mag_sum + 1e-8))
    sym_score = float(np.mean(sym_scores)) if sym_scores else 0.5

    # ── Step 7: Combine into SynthID probability ──────────────────────────────
    # Threshold guide from reverse-SynthID:
    #   tau = 0.60 separates carrier (>0.6) from content (<0.3)
    combined = (cross_coherence * 0.50
                + spatial_coherence * 0.30
                + (sym_score - 0.5) * 0.40)

    # Sigmoid centred at 0.45 (slightly below tau=0.60 to account for
    # single-image estimation noise)
    ai_score = float(np.clip(1.0 / (1.0 + np.exp(-10.0 * (combined - 0.45))), 0, 1))

    # ── Carrier heat-map for visualisation ───────────────────────────────────
    carrier_map = np.zeros((H, W), dtype=np.float32)
    for y, x in zip(top_ys, top_xs):
        carrier_map[y, x] = 1.0
    # Gaussian spread so individual pixels are visible
    from scipy.ndimage import gaussian_filter
    carrier_map = gaussian_filter(carrier_map, sigma=max(H, W) / 200)
    carrier_map = np.clip(carrier_map / (carrier_map.max() + 1e-8), 0, 1)

    return {
        "rg_coherence":    rg_coh,
        "bg_coherence":    bg_coh,
        "cross_coherence": cross_coherence,
        "spatial_coherence": spatial_coherence,
        "sym_score":       sym_score,
        "combined":        combined,
        "ai_score":        ai_score,
        "mag_spectrum":    mag_g,
        "carrier_map":     carrier_map,
        "K":               K,
        "n_bins_searched": int(H * W),
    }

# ══════════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ══════════════════════════════════════════════════════════════════════════════
def enhance_saturation(img: Image.Image, factor: float = 8.0) -> Image.Image:
    try:
        import cv2
        arr = np.array(img, dtype=np.uint8)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        return Image.fromarray(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))
    except ImportError:
        arr = np.array(img, dtype=np.float32) / 255.0
        mean = arr.mean(axis=2, keepdims=True)
        enhanced = np.clip((mean + (arr - mean) * factor) * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(enhanced)


def ela_to_image(ela_map: np.ndarray) -> Image.Image:
    ch = ela_map.mean(axis=2) if ela_map.ndim == 3 else ela_map
    clipped = np.clip(ch, 0, 30)
    norm = (clipped / 30.0 * 255).astype(np.uint8)
    colored = np.stack([norm // 3, norm, norm // 4], axis=2).astype(np.uint8)
    return Image.fromarray(colored)


def fft_to_image(log_mag: np.ndarray) -> Image.Image:
    norm = ((log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-8) * 255).astype(np.uint8)
    return Image.fromarray(norm).convert("RGB")


def carrier_map_to_image(carrier_map: np.ndarray, mag_spectrum: np.ndarray) -> Image.Image:
    """Render SynthID carrier heat-map: green = suspected carriers, dark = background."""
    # Normalise spectrum as dimmed base
    base = np.log1p(mag_spectrum)
    base = (base / (base.max() + 1e-8) * 80).astype(np.uint8)
    # Overlay carrier heat-map in green
    heat = (carrier_map * 255).astype(np.uint8)
    r = np.clip(base.astype(np.int16) - heat.astype(np.int16) // 2, 0, 255).astype(np.uint8)
    g = np.clip(base.astype(np.int16) + heat.astype(np.int16), 0, 255).astype(np.uint8)
    b = base
    return Image.fromarray(np.stack([r, g, b], axis=2))

# ══════════════════════════════════════════════════════════════════════════════
# Main analysis
# ══════════════════════════════════════════════════════════════════════════════
def compute_ai_probability(noise_res, ela_res, dct_res, fmt_res, sid_res):
    combined = (
        noise_res["ai_score"] * 0.30 +
        ela_res["ai_score"]   * 0.20 +
        fmt_res["ai_score"]   * 0.15 +
        dct_res["ai_score"]   * 0.10 +
        sid_res["ai_score"]   * 0.25
    )
    stretched = float(1.0 / (1.0 + np.exp(-9.0 * (combined - 0.50))))
    probability = float(np.clip(stretched, 0.02, 0.98))
    if probability >= 0.68:
        verdict = "likely_ai"
    elif probability <= 0.38:
        verdict = "likely_real"
    else:
        verdict = "inconclusive"
    return probability, verdict


@st.cache_data(show_spinner=False)
def analyze_image(img_bytes: bytes, mime_type: str) -> dict:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    if max(img.size) > 1024:
        img.thumbnail((1024, 1024), Image.LANCZOS)
    enhanced         = enhance_saturation(img, factor=8.0)
    magnitude, log_m = compute_fft_spectrum(img)
    noise_res        = estimate_noise_level(img)
    ela_res          = analyze_ela(img)
    dct_res          = detect_dct_blocks(magnitude, img.size)
    fmt_res          = analyze_source_format(mime_type)
    sid_res          = detect_synthid_watermark(img)
    probability, verdict = compute_ai_probability(noise_res, ela_res, dct_res, fmt_res, sid_res)
    return {
        "original": img, "enhanced": enhanced,
        "log_mag":  log_m, "ela_map": ela_res["ela_map"],
        "carrier_map":  sid_res["carrier_map"],
        "mag_spectrum": sid_res["mag_spectrum"],
        "noise": noise_res, "ela": ela_res,
        "dct":   dct_res,   "fmt": fmt_res,
        "sid":   sid_res,
        "probability": probability, "verdict": verdict,
    }

# ══════════════════════════════════════════════════════════════════════════════
# UI helpers
# ══════════════════════════════════════════════════════════════════════════════
def inject_css():
    st.html(HACKER_CSS)


def render_hero():
    st.html("""
<div style="text-align:center;padding:32px 0 20px;position:relative;">
  <div style="position:absolute;top:0;left:50%;transform:translateX(-50%);
              width:500px;height:100px;
              background:radial-gradient(ellipse,rgba(0,255,68,.07) 0%,transparent 70%);
              pointer-events:none;"></div>
  <div style="font-family:'Orbitron','Share Tech Mono',monospace;font-size:.7rem;
              letter-spacing:6px;color:rgba(0,255,68,.4);text-transform:uppercase;
              margin-bottom:10px;">// FORENSIC IMAGE ANALYSIS SYSTEM v3.0 //</div>
  <div style="font-family:'Orbitron','Share Tech Mono',monospace;font-size:2.2rem;
              font-weight:900;color:#fff;
              text-shadow:0 0 7px #fff,0 0 15px #fff,0 0 30px #00ff88,0 0 60px rgba(0,255,136,.4);
              letter-spacing:4px;line-height:1.2;margin-bottom:8px;">
    AI WATERMARK<br>DETECTOR
  </div>
  <div style="font-family:'Share Tech Mono','Courier New',monospace;font-size:.82rem;
              color:#336633;letter-spacing:2px;margin-top:8px;">
    [ NOISE &bull; ELA &bull; DCT &bull; FORMAT &bull; <span style="color:#00ff88;
    text-shadow:0 0 8px #00ff88;">SYNTHID CARRIER PHASE</span> ]
  </div>
  <div style="margin:18px auto 0;width:300px;height:1px;
              background:linear-gradient(90deg,transparent,#00ff88,transparent);
              box-shadow:0 0 8px #00ff88;"></div>
</div>
""")


def render_terminal_boot():
    st.html("""
<div style="font-family:'Share Tech Mono','Courier New',monospace;color:#336633;
            font-size:.76rem;letter-spacing:1px;padding:4px 0 14px;line-height:1.9;">
  &gt; INITIALIZING FORENSIC ENGINE......... <span style="color:#00ff88;">OK</span><br>
  &gt; LOADING NOISE ESTIMATOR.............. <span style="color:#00ff88;">OK</span><br>
  &gt; ELA SUBSYSTEM READY.................. <span style="color:#00ff88;">OK</span><br>
  &gt; DCT BLOCK DETECTOR................... <span style="color:#00ff88;">OK</span><br>
  &gt; SYNTHID CARRIER PHASE MODULE......... <span style="color:#00ff88;">OK</span><br>
  &gt; REVERSE-SYNTHID ENGINE v1.0.......... <span style="color:#00ff88;">ARMED</span><br>
  &gt; AWAITING INPUT FILE<span style="color:#00ff88;">_</span>
</div>
""")


def render_section_header(text: str):
    st.html(f"""
<div style="font-family:'Orbitron','Share Tech Mono',monospace;font-size:.78rem;
            letter-spacing:4px;color:#00ff88;
            text-shadow:0 0 8px rgba(0,255,136,.5);text-transform:uppercase;
            padding:18px 0 8px;border-bottom:1px solid rgba(0,255,34,.15);">
  // {text}
</div>
""")


def render_verdict(probability: float, verdict: str, sid_score: float):
    pct = int(probability * 100)
    sid_pct = int(sid_score * 100)
    if verdict == "likely_ai":
        gc = "#ff2244"; label = "WARNING — SYNTHETIC ORIGIN DETECTED"
        sub = "HIGH CONFIDENCE &nbsp;|&nbsp; AI-GENERATED"
        rgb = "255,34,68"
    elif verdict == "likely_real":
        gc = "#00ff88"; label = "AUTHENTIC SIGNAL DETECTED"
        sub = "HIGH CONFIDENCE &nbsp;|&nbsp; REAL PHOTOGRAPH"
        rgb = "0,255,136"
    else:
        gc = "#ffaa00"; label = "SIGNAL AMBIGUOUS"
        sub = "INCONCLUSIVE &nbsp;|&nbsp; FURTHER ANALYSIS REQUIRED"
        rgb = "255,170,0"

    # SynthID badge colour
    if sid_pct >= 65:
        sid_c = "#ff2244"; sid_label = "SYNTHID DETECTED"
    elif sid_pct <= 35:
        sid_c = "#00ff88"; sid_label = "NO SYNTHID"
    else:
        sid_c = "#ffaa00"; sid_label = "SYNTHID UNCERTAIN"

    st.html(f"""
<div style="background:radial-gradient(ellipse at center top,rgba({rgb},.04) 0%,#000 55%);
            border:1px solid {gc}55;border-radius:2px;padding:32px 36px 28px;margin:16px 0;
            text-align:center;position:relative;
            box-shadow:0 0 40px {gc}18,inset 0 0 60px {gc}06;
            font-family:'Share Tech Mono','Courier New',monospace;">
  <div style="position:absolute;top:-1px;left:-1px;width:18px;height:18px;border-top:2px solid {gc};border-left:2px solid {gc};box-shadow:-2px -2px 10px {gc};"></div>
  <div style="position:absolute;top:-1px;right:-1px;width:18px;height:18px;border-top:2px solid {gc};border-right:2px solid {gc};box-shadow:2px -2px 10px {gc};"></div>
  <div style="position:absolute;bottom:-1px;left:-1px;width:18px;height:18px;border-bottom:2px solid {gc};border-left:2px solid {gc};box-shadow:-2px 2px 10px {gc};"></div>
  <div style="position:absolute;bottom:-1px;right:-1px;width:18px;height:18px;border-bottom:2px solid {gc};border-right:2px solid {gc};box-shadow:2px 2px 10px {gc};"></div>
  <div style="font-size:.72rem;letter-spacing:4px;color:{gc};text-shadow:0 0 8px {gc};margin-bottom:14px;">{label}</div>
  <div style="display:flex;align-items:center;justify-content:center;gap:32px;">
    <div>
      <div style="font-family:'Orbitron','Share Tech Mono',monospace;font-size:5rem;font-weight:900;
                  color:#fff;text-shadow:0 0 7px #fff,0 0 20px {gc};line-height:1;margin-bottom:4px;">
        {pct}<span style="font-size:2.2rem;opacity:.6;">%</span>
      </div>
      <div style="font-size:.68rem;letter-spacing:2px;color:{gc}88;">AI PROBABILITY</div>
    </div>
    <div style="width:1px;height:70px;background:linear-gradient(180deg,transparent,{gc}44,transparent);"></div>
    <div>
      <div style="font-family:'Orbitron',monospace;font-size:2.4rem;font-weight:700;
                  color:{sid_c};text-shadow:0 0 10px {sid_c};line-height:1;margin-bottom:4px;">
        {sid_pct}<span style="font-size:1.1rem;opacity:.6;">%</span>
      </div>
      <div style="font-size:.65rem;letter-spacing:2px;color:{sid_c}88;">&#x25C8; {sid_label}</div>
    </div>
  </div>
  <div style="font-size:.68rem;letter-spacing:3px;color:{gc}77;margin:14px 0 18px;">{sub}</div>
  <div style="background:#001008;border:1px solid {gc}28;height:7px;overflow:hidden;border-radius:1px;">
    <div style="background:linear-gradient(90deg,{gc}44,{gc});width:{pct}%;height:100%;box-shadow:0 0 10px {gc};"></div>
  </div>
</div>
""")


def render_signal_card(label: str, score: float, detail: str, symbol: str):
    pct = int(score * 100)
    if pct > 62:
        bc, tc = "#ff2244", "#ff4466"
    elif pct < 38:
        bc, tc = "#00ff88", "#00cc66"
    else:
        bc, tc = "#ffaa00", "#ffcc44"

    st.html(f"""
<div style="border:1px solid {bc}2a;border-left:3px solid {bc};
            background:linear-gradient(90deg,{bc}06 0%,transparent 50%);
            padding:12px 16px;margin:6px 0;border-radius:1px;
            font-family:'Share Tech Mono','Courier New',monospace;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
    <span style="color:#a0e8a0;font-size:.8rem;letter-spacing:1px;">{symbol} {label}</span>
    <span style="color:{tc};font-family:'Orbitron',monospace;font-size:1rem;
                 font-weight:700;text-shadow:0 0 8px {bc};">{pct}%</span>
  </div>
  <div style="background:#001008;height:4px;border-radius:1px;overflow:hidden;margin-bottom:6px;">
    <div style="background:linear-gradient(90deg,{bc}66,{bc});
                width:{pct}%;height:100%;box-shadow:0 0 8px {bc};"></div>
  </div>
  <div style="color:#446644;font-size:.71rem;letter-spacing:.5px;">{detail}</div>
</div>
""")


def render_synthid_detail(sid: dict):
    rg  = int(sid["rg_coherence"]    * 100)
    bg  = int(sid["bg_coherence"]    * 100)
    spa = int(sid["spatial_coherence"]* 100)
    sym = int(sid["sym_score"]        * 100)
    cc  = int(sid["cross_coherence"]  * 100)

    def bar(val, ref=60):
        col = "#ff2244" if val >= ref else "#00ff88" if val < ref-15 else "#ffaa00"
        return (f'<div style="background:#001008;height:3px;border-radius:1px;'
                f'overflow:hidden;margin:2px 0 6px;">'
                f'<div style="background:{col};width:{val}%;height:100%;'
                f'box-shadow:0 0 6px {col};"></div></div>')

    st.html(f"""
<div style="border:1px solid #00ff2233;border-left:3px solid #00ff88;
            background:#000a04;padding:16px 20px;margin:8px 0;border-radius:1px;
            font-family:'Share Tech Mono','Courier New',monospace;font-size:.75rem;">
  <div style="color:#00ff88;letter-spacing:3px;font-size:.78rem;margin-bottom:12px;">
    &#x25C8; SYNTHID CARRIER ANALYSIS — reverse-SynthID
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px 24px;">
    <div>
      <div style="color:#669966;margin-bottom:1px;">R&#x2194;G PHASE COHERENCE</div>
      {bar(rg)}
      <span style="color:#00ff88;">{rg}%</span>
      <span style="color:#334433;"> / tau=60%</span>
    </div>
    <div>
      <div style="color:#669966;margin-bottom:1px;">B&#x2194;G PHASE COHERENCE</div>
      {bar(bg)}
      <span style="color:#00ff88;">{bg}%</span>
      <span style="color:#334433;"> / tau=60%</span>
    </div>
    <div>
      <div style="color:#669966;margin-bottom:1px;">SPATIAL PATCH COHERENCE</div>
      {bar(spa)}
      <span style="color:#00ff88;">{spa}%</span>
    </div>
    <div>
      <div style="color:#669966;margin-bottom:1px;">FFT CONJUGATE SYMMETRY</div>
      {bar(sym, ref=65)}
      <span style="color:#00ff88;">{sym}%</span>
    </div>
  </div>
  <div style="margin-top:10px;padding-top:8px;border-top:1px solid #00ff2211;
              color:#334433;font-size:.69rem;line-height:1.6;">
    CROSS-CHANNEL COHERENCE: <span style="color:#88cc88;">{cc}%</span>
    &nbsp;&bull;&nbsp; K={sid['K']} carrier candidates searched
    &nbsp;&bull;&nbsp; REF: github.com/aloshdenny/reverse-SynthID
  </div>
</div>
""")


def render_log_block(lines: list, accent_color: str):
    rows = ""
    for i, line in enumerate(lines):
        color = accent_color if i == len(lines) - 1 else "#336633"
        rows += f'<div style="color:{color};">{line}</div>'
    st.html(f"""
<div style="background:#000a04;border:1px solid #00ff2218;border-left:3px solid {accent_color};
            padding:14px 18px;font-family:'Share Tech Mono','Courier New',monospace;
            font-size:.76rem;line-height:1.85;margin-top:14px;
            box-shadow:0 0 20px {accent_color}0c;">
  {rows}
</div>
""")

# ══════════════════════════════════════════════════════════════════════════════
# Pages
# ══════════════════════════════════════════════════════════════════════════════
def page_home():
    render_hero()
    render_terminal_boot()

    uploaded_file = st.file_uploader(
        "UPLOAD TARGET IMAGE",
        type=SUPPORTED_FORMATS,
        help="JPG · PNG · WEBP · max 10 MB",
    )

    if uploaded_file is None:
        st.html("""
<div style="border:1px solid rgba(0,255,34,.2);border-radius:2px;
            padding:48px 20px;text-align:center;
            background:radial-gradient(ellipse at center,#001a0a 0%,#000 70%);
            box-shadow:0 0 30px rgba(0,255,26,.04),inset 0 0 40px rgba(0,255,26,.03);
            margin-top:10px;font-family:'Share Tech Mono',monospace;position:relative;">
  <div style="position:absolute;top:-1px;left:-1px;width:12px;height:12px;border-top:2px solid #00ff88;border-left:2px solid #00ff88;box-shadow:-2px -2px 8px #00ff88;"></div>
  <div style="position:absolute;top:-1px;right:-1px;width:12px;height:12px;border-top:2px solid #00ff88;border-right:2px solid #00ff88;box-shadow:2px -2px 8px #00ff88;"></div>
  <div style="position:absolute;bottom:-1px;left:-1px;width:12px;height:12px;border-bottom:2px solid #00ff88;border-left:2px solid #00ff88;box-shadow:-2px 2px 8px #00ff88;"></div>
  <div style="position:absolute;bottom:-1px;right:-1px;width:12px;height:12px;border-bottom:2px solid #00ff88;border-right:2px solid #00ff88;box-shadow:2px 2px 8px #00ff88;"></div>
  <div style="font-size:2.2rem;margin-bottom:12px;filter:drop-shadow(0 0 10px #00ff88);">&#128269;</div>
  <div style="color:rgba(0,255,68,.5);font-size:.9rem;letter-spacing:2px;">AWAITING TARGET IMAGE</div>
  <div style="color:#1a331a;font-size:.73rem;margin-top:8px;letter-spacing:1px;">JPG &nbsp;&#xB7;&nbsp; PNG &nbsp;&#xB7;&nbsp; WEBP &nbsp;&#xB7;&nbsp; MAX 10 MB</div>
</div>
""")
        return

    file_bytes = uploaded_file.read()
    if len(file_bytes) / (1024 * 1024) > MAX_FILE_SIZE_MB:
        st.error("FILE TOO LARGE — Maximum 10 MB")
        return

    mime_type = uploaded_file.type or "image/jpeg"

    with st.spinner("RUNNING FORENSIC ANALYSIS + SYNTHID SCAN..."):
        try:
            results = analyze_image(file_bytes, mime_type)
        except Exception as e:
            st.error(f"ANALYSIS FAILED: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

    sid   = results["sid"]
    noise = results["noise"]
    ela   = results["ela"]
    dct   = results["dct"]
    fmt   = results["fmt"]

    # ── Verdict ───────────────────────────────────────────────────────────────
    render_section_header("VERDICT")
    render_verdict(results["probability"], results["verdict"], sid["ai_score"])

    # ── Visuals ───────────────────────────────────────────────────────────────
    render_section_header("VISUAL FORENSICS")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.image(results["original"],                  caption="[ ORIGINAL ]",          use_container_width=True)
    with c2:
        st.image(results["enhanced"],                  caption="[ SATURATION x8 ]",     use_container_width=True)
    with c3:
        st.image(ela_to_image(results["ela_map"]),     caption="[ ELA MAP ]",            use_container_width=True)
    with c4:
        st.image(fft_to_image(results["log_mag"]),     caption="[ FFT SPECTRUM ]",       use_container_width=True)
    with c5:
        st.image(
            carrier_map_to_image(results["carrier_map"], results["mag_spectrum"]),
            caption="[ SYNTHID CARRIERS ]",
            use_container_width=True,
        )

    # ── SynthID deep-dive ────────────────────────────────────────────────────
    render_section_header("SYNTHID CARRIER PHASE ANALYSIS")
    render_synthid_detail(sid)

    # ── Signal breakdown ──────────────────────────────────────────────────────
    render_section_header("SIGNAL ANALYSIS")
    render_signal_card(
        "SYNTHID CARRIER PHASE", sid["ai_score"],
        (f"cross-channel coherence={int(sid['cross_coherence']*100)}%  "
         f"spatial={int(sid['spatial_coherence']*100)}%  "
         f"sym={int(sid['sym_score']*100)}%  |  "
         + ("CARRIER PHASE COHERENCE ELEVATED — SYNTHID SIGNATURE PROBABLE"
            if sid["ai_score"] > 0.6
            else "NO STRONG CARRIER COHERENCE — SYNTHID NOT DETECTED"
            if sid["ai_score"] < 0.4
            else "MARGINAL COHERENCE — INCONCLUSIVE")),
        "[S]",
    )
    render_signal_card(
        "NOISE LEVEL", noise["ai_score"],
        f"Laplacian sigma={noise['sigma']:.2f}  |  " + (
            "UNNATURALLY CLEAN — AI SIGNATURE" if noise["sigma"] < 3
            else "NATURAL SENSOR NOISE — CAMERA" if noise["sigma"] > 6
            else "BORDERLINE REGION"),
        "[!]",
    )
    render_signal_card(
        "ERROR LEVEL ANALYSIS", ela["ai_score"],
        f"Mean ELA={ela['ela_mean']:.2f}  |  " + (
            "HIGH — FIRST JPEG COMPRESSION — PNG/AI SOURCE" if ela["ela_mean"] > 7
            else "LOW — PREVIOUSLY JPEG COMPRESSED — CAMERA"),
        "[~]",
    )
    render_signal_card(
        "SOURCE FORMAT", fmt["ai_score"],
        f"Detected: {fmt['fmt_label']}  |  " + (
            "PNG IS DEFAULT OUTPUT FOR MOST AI GENERATORS" if fmt["fmt_label"] == "PNG"
            else "WEBP USED BY GEMINI / AI PLATFORMS" if fmt["fmt_label"] == "WebP"
            else "JPEG IS NATIVE CAMERA FORMAT"),
        "[F]",
    )
    render_signal_card(
        "DCT BLOCK STRUCTURE", dct["ai_score"],
        f"Block strength={dct['dct_strength']:.3f}  |  " + (
            "STRONG 8x8 JPEG GRID — CAMERA JPEG" if dct["dct_strength"] > 0.5
            else "NO JPEG BLOCK PATTERN — LOSSLESS/AI SOURCE"),
        "[D]",
    )

    # ── Conclusion log ───────────────────────────────────────────────────────
    prob    = results["probability"]
    verdict = results["verdict"]
    if verdict == "likely_ai":
        color = "#ff2244"; status = "SYNTHETIC"
        conclusion = "SYNTHETIC ORIGIN — AI WATERMARK SIGNATURE DETECTED"
    elif verdict == "likely_real":
        color = "#00ff88"; status = "AUTHENTIC"
        conclusion = "SIGNALS CONSISTENT WITH REAL CAMERA PHOTOGRAPH"
    else:
        color = "#ffaa00"; status = "INCONCLUSIVE"
        conclusion = "SIGNALS CONFLICT — POSSIBLE POST-PROCESSING OR RE-ENCODING"

    render_log_block([
        f"[RESULT]   AI_PROBABILITY={int(prob*100)}%  STATUS={status}",
        f"[SYNTHID]  coherence={int(sid['cross_coherence']*100)}%  spatial={int(sid['spatial_coherence']*100)}%  score={int(sid['ai_score']*100)}%",
        f"[NOISE]    sigma={noise['sigma']:.2f}  threshold=4.0",
        f"[ELA]      mean={ela['ela_mean']:.2f}  threshold=7.0",
        f"[FORMAT]   type={fmt['fmt_label']}",
        f"[DCT]      strength={dct['dct_strength']:.3f}",
        f"[CONCLUSION] {conclusion}",
    ], color)

    # ── Raw data ──────────────────────────────────────────────────────────────
    with st.expander("[ RAW FORENSIC DATA ]"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.json({"noise_sigma": round(noise["sigma"], 4), "noise_ai": round(noise["ai_score"], 4)})
            st.json({"dct_strength": round(dct["dct_strength"], 4), "dct_ai": round(dct["ai_score"], 4)})
        with c2:
            st.json({"ela_mean": round(ela["ela_mean"], 4), "ela_std": round(ela["ela_std"], 4), "ela_ai": round(ela["ai_score"], 4)})
            st.json({"format": fmt["fmt_label"], "fmt_ai": round(fmt["ai_score"], 4)})
        with c3:
            st.json({
                "synthid_ai_score":    round(sid["ai_score"], 4),
                "cross_coherence":     round(sid["cross_coherence"], 4),
                "rg_coherence":        round(sid["rg_coherence"], 4),
                "bg_coherence":        round(sid["bg_coherence"], 4),
                "spatial_coherence":   round(sid["spatial_coherence"], 4),
                "sym_score":           round(sid["sym_score"], 4),
                "K_bins":              sid["K"],
            })

    st.caption(
        "FORENSIC HEURISTICS — SYNTHID REVERSE-ENGINEERING BASED ON "
        "GITHUB.COM/ALOSHDENNY/REVERSE-SYNTHID — NOT A TRAINED ML MODEL — "
        "PROBABILISTIC ESTIMATES — POST-PROCESSING MAY AFFECT RESULTS"
    )


def page_about():
    st.html("""
<div style="text-align:center;padding:28px 0 18px;">
  <div style="font-family:'Orbitron',monospace;font-size:.68rem;
              letter-spacing:5px;color:rgba(0,255,68,.35);margin-bottom:10px;">
    // TECHNICAL DOCUMENTATION //</div>
  <div style="font-family:'Orbitron',monospace;font-size:1.7rem;font-weight:900;
              color:#fff;letter-spacing:4px;
              text-shadow:0 0 10px #fff,0 0 25px #00ff88,0 0 60px rgba(0,255,68,.25);">
    DETECTION ALGORITHMS
  </div>
  <div style="margin:14px auto 0;width:180px;height:1px;
              background:linear-gradient(90deg,transparent,#00ff88,transparent);
              box-shadow:0 0 8px #00ff88;"></div>
</div>
""")

    render_section_header("SIGNAL 01 — SYNTHID CARRIER PHASE COHERENCE  [WEIGHT: 25%]")
    st.markdown("""
**Source:** [aloshdenny/reverse-SynthID](https://github.com/aloshdenny/reverse-SynthID) — 90 % detector accuracy

Google Gemini (SynthID) embeds an invisible watermark **during the diffusion process itself** — not as a post-processing overlay.
The watermark lives in the **noise residual** at specific resolution-dependent carrier frequencies in the 2-D Fourier domain.

**Critical property:** carrier phases are **image-content-independent**.
They are identical across every SynthID image regardless of scene content.
Real images have random phases at those bins because their energy comes from actual content.

Reverse-engineered cross-image phase coherence: **99.5%** (SynthID) vs **< 0.3** (content bins).

**Implementation (4 steps):**
```python
# 1. Extract noise residual  (remove structural content)
noise = image − box_blur(image, radius=7)

# 2. 2-D FFT of noise residual, weighted by channel strength
#    Channel weights recovered from reverse-SynthID:
CH_WEIGHTS = [R=0.85, G=1.0, B=0.70]   # Green is dominant

# 3. Find top-K magnitude bins (carrier candidates) outside DC region
K = 128 bins; exclude centre (H/12 × W/12) DC mask

# 4. Measure cross-channel phase coherence at candidate bins
rg_coherence = |mean(exp(i × (phase_R − phase_G)))|   # ≥ 0.6 → carrier
bg_coherence = |mean(exp(i × (phase_B − phase_G)))|
tau = 0.60   # threshold from reverse-SynthID codebook
```

Additional sub-checks:
- **Spatial patch coherence**: split into N×N quadrants; check phase stability of dominant bin across patches
- **FFT conjugate symmetry**: real signals satisfy `F(y,x) = conj(F(H-y, W-x))` — synthetic carriers are perfectly symmetric
""")

    render_section_header("SIGNAL 02 — NOISE LEVEL ESTIMATION  [WEIGHT: 30%]")
    st.markdown("""
Real camera sensors produce **shot noise** and **thermal noise**. AI generators synthesise values mathematically — images are unnaturally clean.

```python
kernel = [[1,-2,1],[-2,4,-2],[1,-2,1]]   # Laplacian
sigma  = sqrt(pi/2) * mean(|convolve(gray, kernel)|) / 6.0
```
- `sigma < 2` → AI &nbsp;&nbsp; `sigma 2–5` → borderline &nbsp;&nbsp; `sigma > 6` → real
""")

    render_section_header("SIGNAL 03 — ERROR LEVEL ANALYSIS (ELA)  [WEIGHT: 20%]")
    st.markdown("""
AI images output as **lossless PNG** → first JPEG compression causes large ELA.
Camera photos are already JPEG in-device → re-saving causes low ELA.
```
re_saved = save(img, JPEG, quality=92) → reload
ELA_mean = mean(|original − re_saved|)   # > 10 → AI  |  < 5 → real
```
""")

    render_section_header("SIGNAL 04 — SOURCE FORMAT  [WEIGHT: 15%]")
    st.markdown("""
| Format | Typical Source | AI Score |
|--------|---------------|----------|
| JPEG | Camera / smartphone — nearly always real | 0.25 |
| PNG | ChatGPT, DALL·E, Midjourney, Stable Diffusion | 0.67 |
| WebP | Google Gemini (SynthID), web-delivered AI | 0.62 |
""")

    render_section_header("SIGNAL 05 — DCT BLOCK STRUCTURE  [WEIGHT: 10%]")
    st.markdown("""
JPEG divides images into **8×8 DCT blocks**. In the 2-D FFT these appear as energy spikes at multiples of `N/8` from the DC centre.
AI PNG images have no such structure.
""")

    render_section_header("KNOWN LIMITATIONS")
    st.warning("""
FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY — NOT A TRAINED ML MODEL.

- SynthID detection uses a single-image approximation of multi-image phase coherence
- A SynthID-watermarked image re-saved as JPEG may be misclassified
- V4 bypass pipeline (VAE + elastic deform + FFT subtraction) can defeat this detector
- Do NOT use as sole evidence in any legal, journalistic, or forensic context
""")

    render_section_header("REFERENCES")
    st.markdown("""
- [aloshdenny/reverse-SynthID](https://github.com/aloshdenny/reverse-SynthID) — carrier frequency reverse engineering, 90% detector accuracy
- [SynthID — Google DeepMind](https://deepmind.google/technologies/synthid/)
- Immerkær (1996): *Fast Noise Variance Estimation*
- [C2PA Content Provenance Standard](https://c2pa.org/)
- [Error Level Analysis — 29a.ch](https://29a.ch/photo-forensics/#error-level-analysis)
""")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    inject_css()
    tabs = st.tabs(["  DETECTOR  ", "  ALGORITHMS  "])
    with tabs[0]:
        page_home()
    with tabs[1]:
        page_about()


if __name__ == "__main__":
    main()
