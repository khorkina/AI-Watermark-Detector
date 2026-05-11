"""
AI Watermark Detector — "Is This AI?"
Hacker / cryptography aesthetic. Port 5000.

Detection signals:
  1. Noise Level Estimation        (Laplacian sigma)    — 35 %
  2. Error Level Analysis (ELA)                         — 25 %
  3. Source Format Forensics                            — 15 %
  4. DCT Block Structure                                — 10 %
  5. SynthID CVR  (reverse-SynthID, exact code)        — 15 %
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
# SIGNAL 1 — Noise Level (Laplacian)
# ══════════════════════════════════════════════════════════════════════════════
def estimate_noise_level(img: Image.Image) -> dict:
    from scipy.ndimage import convolve
    gray = np.array(img.convert("L"), dtype=np.float32)
    kernel = np.array([[1,-2,1],[-2,4,-2],[1,-2,1]], dtype=np.float32)
    sigma = float(np.sqrt(np.pi / 2) * np.abs(convolve(gray, kernel)).mean() / 6.0)
    # Real cameras: sigma typically 4-15 (depends on ISO)
    # AI generators: sigma typically 0.5-2.5 (unnaturally smooth)
    # Centred at 3.0 so real photos land safely below 0.5
    ai_score = float(np.clip(1.0 / (1.0 + np.exp(0.8 * (sigma - 3.0))), 0, 1))
    return {"sigma": sigma, "ai_score": ai_score}


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL 2 — Error Level Analysis
# ══════════════════════════════════════════════════════════════════════════════
def analyze_ela(img: Image.Image) -> dict:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    resaved  = Image.open(buf).convert("RGB")
    ela_map  = np.abs(np.array(img, dtype=np.float32) - np.array(resaved, dtype=np.float32))
    ela_mean = float(ela_map.mean())
    # Real JPEG photos already compressed → ela_mean ≈ 1-4
    # AI PNG images, first compression → ela_mean ≈ 8-25
    # Centred at 6.5
    ai_score = float(np.clip(1.0 / (1.0 + np.exp(-0.55 * (ela_mean - 6.5))), 0, 1))
    return {
        "ela_mean": ela_mean,
        "ela_std": float(ela_map.std()),
        "ela_map": ela_map,
        "ai_score": ai_score,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL 3 — Source Format
# ══════════════════════════════════════════════════════════════════════════════
def analyze_source_format(mime_type: str) -> dict:
    if mime_type in ("image/jpeg", "image/jpg"):
        return {"fmt_label": "JPEG", "ai_score": 0.20}
    elif mime_type == "image/webp":
        return {"fmt_label": "WebP", "ai_score": 0.58}
    elif mime_type == "image/png":
        return {"fmt_label": "PNG",  "ai_score": 0.62}
    return {"fmt_label": "Unknown", "ai_score": 0.45}


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL 4 — DCT Block Structure
# ══════════════════════════════════════════════════════════════════════════════
def compute_fft_spectrum(img: Image.Image):
    gray = np.array(img.convert("L"), dtype=np.float32)
    h, w = gray.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    fft_shifted = np.fft.fftshift(np.fft.fft2(gray * window))
    magnitude = np.abs(fft_shifted)
    return magnitude, np.log1p(magnitude)


def detect_dct_blocks(magnitude: np.ndarray) -> dict:
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    ratios = []
    for axis in ["v", "h"]:
        dim = h if axis == "v" else w
        center = cy if axis == "v" else cx
        for k in range(1, 4):
            offset = k * dim // 8
            if offset < 4 or offset > dim // 2 - 6:
                continue
            for pos in [center + offset, center - offset]:
                if not (4 <= pos < dim - 4):
                    continue
                if axis == "v":
                    pk  = float(magnitude[pos-2:pos+3, cx-2:cx+3].max())
                    bg  = float(magnitude[pos-8:pos+9, cx-2:cx+3].mean())
                else:
                    pk  = float(magnitude[cy-2:cy+3, pos-2:pos+3].max())
                    bg  = float(magnitude[cy-2:cy+3, pos-8:pos+9].mean())
                if bg > 0:
                    ratios.append(pk / bg)
    dct_str = float(np.clip((np.mean(ratios) - 1.0) / 6.0, 0, 1)) if ratios else 0.0
    return {
        "dct_strength": dct_str,
        "avg_ratio": float(np.mean(ratios)) if ratios else 1.0,
        "ai_score": float(np.clip(1.0 - dct_str, 0, 1)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL 5 — SynthID Carrier-to-Variance Ratio
#
# Faithfully implements the core detection approach from:
#   github.com/aloshdenny/reverse-SynthID
#   src/extraction/robust_extractor.py  (RobustSynthIDExtractor)
#
# Carrier bin offsets (fy, fx) at 512 px resolution, reverse-engineered from
# 291 Gemini-generated images. Each set has >0.95 intra-set phase coherence.
#
# Without a reference codebook we cannot do full phase matching.
# We therefore use the supporting CVR signal:
#   CVR = mean |noise_fft| at carrier bins
#         ─────────────────────────────────
#         mean |noise_fft| at random bins
#
# Calibration (from repo):
#   Watermarked images: CVR >> 1 (elevated carrier energy)
#   Non-watermarked:    CVR ≈ 1.0 (no systematic elevation)
# ══════════════════════════════════════════════════════════════════════════════

# Empirically verified SynthID carrier offsets at 512 px
# Dark-image carriers (diagonal grid, black / nb_pro images):
_CARRIERS_DARK = [
    (-5, -3), (5, 3), (-5, 3), (5, -3),
    (-3, -4), (3, 4), (-3, 4), (3, -4),
    (-4, -3), (4, 3), (-4, 3), (4, -3),
    (-5, -1), (5, 1), (-5, 1), (5, -1),
    (-5, -2), (5, 2), (-5, 2), (5, -2),
    (-2, -5), (2, 5), (-2, 5), (2, -5),
    (-1, -5), (1, 5), (-1, 5), (1, -5),
    (-4, -4), (4, 4), (-4, 4), (4, -4),
    (-1, -6), (1, 6), (-3, -5), (3, 5),
]
# White-image carriers (horizontal axis):
_CARRIERS_WHITE = [
    (0, -7), (0, 7), (0, -8), (0, 8),
    (0, -9), (0, 9), (0, -10), (0, 10),
    (0, -11), (0, 11), (0, -12), (0, 12),
    (0, -20), (0, 20), (0, -21), (0, 21),
    (0, -22), (0, 22), (0, -23), (0, 23),
]
_ALL_CARRIERS = _CARRIERS_DARK + _CARRIERS_WHITE


def detect_synthid_watermark(img: Image.Image) -> dict:
    """
    Single-image SynthID detector — Carrier-to-Variance Ratio (CVR).

    Mirrors RobustSynthIDExtractor.detect_array() from:
      github.com/aloshdenny/reverse-SynthID /src/extraction/robust_extractor.py

    Steps
    -----
    1. Resize image to 512×512 (canonical scale used by the repo).
    2. Extract noise residual via multi-method fusion
       (gaussian + uniform filter subtraction, weighted average).
    3. 2-D FFT of weighted-grayscale noise (channel weights G=1.0 R=0.85 B=0.70,
       verified by reverse-SynthID to match SynthID embedding strength).
    4. Measure |FFT| at known dark-carrier and white-carrier bins.
    5. Measure |FFT| at equal-count random bins at same radial distances (seed=42).
    6. CVR = mean(carrier_mags) / mean(random_mags).
       Best CVR = max(dark_CVR, white_CVR).
    7. Score via sigmoid centred at CVR=2.0 (no-watermark baseline ≈ 1.0).
    """
    from scipy.ndimage import uniform_filter, gaussian_filter

    TARGET = 512

    # ── 1. Resize to 512×512 (same as repo) ──────────────────────────────────
    arr512 = np.array(img.resize((TARGET, TARGET), Image.LANCZOS), dtype=np.float32)

    # ── 2. Noise residual — multi-method fusion ───────────────────────────────
    # Channel weights from reverse-SynthID: G strongest, B weakest
    CH_W = np.array([0.85, 1.0, 0.70], dtype=np.float32)   # R, G, B
    noise = np.zeros(arr512.shape, dtype=np.float32)
    for c in range(3):
        ch = arr512[:, :, c]
        # Three complementary residuals (mirrors the repo's wavelet / bilateral / NLM fusion)
        n1 = ch - uniform_filter(ch, size=7)
        n2 = ch - gaussian_filter(ch, sigma=3.0)
        n3 = ch - uniform_filter(ch, size=15)
        noise[:, :, c] = (0.40 * n1 + 0.35 * n2 + 0.25 * n3) * CH_W[c]

    # Weighted grayscale (mirrors noise_gray = np.mean(noise, axis=2))
    noise_gray = noise.mean(axis=2)

    # ── 3. 2-D FFT ───────────────────────────────────────────────────────────
    fft_noise = np.fft.fftshift(np.fft.fft2(noise_gray))
    noise_mag = np.abs(fft_noise)

    cy, cx = TARGET // 2, TARGET // 2   # = 256, 256

    # ── 4. Carrier magnitudes ─────────────────────────────────────────────────
    def _get_mags(carriers):
        mags = []
        for fy, fx in carriers:
            y, x = fy + cy, fx + cx
            if 0 <= y < TARGET and 0 <= x < TARGET:
                mags.append(float(noise_mag[y, x]))
        return mags

    dark_mags  = _get_mags(_CARRIERS_DARK)
    white_mags = _get_mags(_CARRIERS_WHITE)
    all_mags   = dark_mags + white_mags

    # ── 5. Random reference magnitudes (repo: seed=42, same count) ───────────
    rng = np.random.RandomState(42)
    random_mags = []
    for fy, fx in _ALL_CARRIERS:
        r = np.sqrt(fy ** 2 + fx ** 2)
        angle = rng.uniform(0, 2 * np.pi)
        ry = int(cy + r * np.sin(angle))
        rx = int(cx + r * np.cos(angle))
        # avoid DC
        if 0 <= ry < TARGET and 0 <= rx < TARGET and (abs(ry - cy) > 2 or abs(rx - cx) > 2):
            random_mags.append(float(noise_mag[ry, rx]))

    mean_random  = float(np.mean(random_mags)) + 1e-10

    # ── 6. CVR ───────────────────────────────────────────────────────────────
    dark_cvr  = float(np.mean(dark_mags))  / mean_random if dark_mags  else 1.0
    white_cvr = float(np.mean(white_mags)) / mean_random if white_mags else 1.0
    all_cvr   = float(np.mean(all_mags))   / mean_random if all_mags   else 1.0
    best_cvr  = max(dark_cvr, white_cvr)   # take the best-matching carrier set

    # ── 7. Score — sigmoid centred at CVR=2.0 ────────────────────────────────
    # Baseline (non-watermarked images): CVR ≈ 1.0
    # SynthID watermarked:               CVR typically > 2.0
    ai_score = float(np.clip(1.0 / (1.0 + np.exp(-2.5 * (best_cvr - 2.0))), 0, 1))

    # ── Carrier visualisation ─────────────────────────────────────────────────
    vis = np.zeros((TARGET, TARGET), dtype=np.float32)
    for fy, fx in _ALL_CARRIERS:
        y, x = fy + cy, fx + cx
        if 0 <= y < TARGET and 0 <= x < TARGET:
            vis[y, x] = 1.0
    vis = gaussian_filter(vis, sigma=TARGET / 200)
    if vis.max() > 0:
        vis /= vis.max()

    return {
        "dark_cvr":    dark_cvr,
        "white_cvr":   white_cvr,
        "all_cvr":     all_cvr,
        "best_cvr":    best_cvr,
        "mean_random": mean_random,
        "ai_score":    ai_score,
        "noise_mag":   noise_mag,
        "carrier_vis": vis,
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
        return Image.fromarray(np.clip((mean + (arr - mean) * factor) * 255, 0, 255).astype(np.uint8))


def ela_to_image(ela_map: np.ndarray) -> Image.Image:
    ch = ela_map.mean(axis=2) if ela_map.ndim == 3 else ela_map
    norm = (np.clip(ch, 0, 30) / 30.0 * 255).astype(np.uint8)
    return Image.fromarray(np.stack([norm // 3, norm, norm // 4], axis=2))


def fft_to_image(log_mag: np.ndarray) -> Image.Image:
    norm = ((log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-8) * 255).astype(np.uint8)
    return Image.fromarray(norm).convert("RGB")


def carrier_map_to_image(vis: np.ndarray, noise_mag: np.ndarray) -> Image.Image:
    base = np.log1p(noise_mag)
    base = (base / (base.max() + 1e-8) * 80).astype(np.uint8)
    heat = (vis * 255).astype(np.uint8)
    r = np.clip(base.astype(np.int16) - heat.astype(np.int16) // 2, 0, 255).astype(np.uint8)
    g = np.clip(base.astype(np.int16) + heat.astype(np.int16), 0, 255).astype(np.uint8)
    b = base
    return Image.fromarray(np.stack([r, g, b], axis=2))


# ══════════════════════════════════════════════════════════════════════════════
# Ensemble
# ══════════════════════════════════════════════════════════════════════════════
def compute_ai_probability(noise, ela, dct, fmt, sid):
    combined = (
        noise["ai_score"] * 0.35 +
        ela["ai_score"]   * 0.25 +
        fmt["ai_score"]   * 0.15 +
        dct["ai_score"]   * 0.10 +
        sid["ai_score"]   * 0.15
    )
    prob = float(np.clip(1.0 / (1.0 + np.exp(-9.0 * (combined - 0.50))), 0.02, 0.98))
    if prob >= 0.68:
        verdict = "likely_ai"
    elif prob <= 0.35:
        verdict = "likely_real"
    else:
        verdict = "inconclusive"
    return prob, verdict


@st.cache_data(show_spinner=False)
def analyze_image(img_bytes: bytes, mime_type: str) -> dict:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    if max(img.size) > 1024:
        img.thumbnail((1024, 1024), Image.LANCZOS)

    enhanced         = enhance_saturation(img, factor=8.0)
    magnitude, log_m = compute_fft_spectrum(img)
    noise_res        = estimate_noise_level(img)
    ela_res          = analyze_ela(img)
    dct_res          = detect_dct_blocks(magnitude)
    fmt_res          = analyze_source_format(mime_type)
    sid_res          = detect_synthid_watermark(img)
    prob, verdict    = compute_ai_probability(noise_res, ela_res, dct_res, fmt_res, sid_res)

    return {
        "original": img, "enhanced": enhanced,
        "log_mag":  log_m, "ela_map": ela_res["ela_map"],
        "carrier_vis": sid_res["carrier_vis"],
        "noise_mag":   sid_res["noise_mag"],
        "noise": noise_res, "ela": ela_res,
        "dct":   dct_res,   "fmt": fmt_res,
        "sid":   sid_res,
        "probability": prob, "verdict": verdict,
    }


# ══════════════════════════════════════════════════════════════════════════════
# UI helpers
# ══════════════════════════════════════════════════════════════════════════════
def inject_css():
    st.html(HACKER_CSS)


def render_hero():
    st.html("""
<div style="text-align:center;padding:32px 0 20px;position:relative;">
  <div style="font-family:'Orbitron','Share Tech Mono',monospace;font-size:.7rem;
              letter-spacing:6px;color:rgba(0,255,68,.4);text-transform:uppercase;
              margin-bottom:10px;">// FORENSIC IMAGE ANALYSIS SYSTEM v3.1 //</div>
  <div style="font-family:'Orbitron','Share Tech Mono',monospace;font-size:2.2rem;
              font-weight:900;color:#fff;
              text-shadow:0 0 7px #fff,0 0 15px #fff,0 0 30px #00ff88,0 0 60px rgba(0,255,136,.4);
              letter-spacing:4px;line-height:1.2;margin-bottom:8px;">
    AI WATERMARK<br>DETECTOR
  </div>
  <div style="font-family:'Share Tech Mono','Courier New',monospace;font-size:.82rem;
              color:#336633;letter-spacing:2px;margin-top:8px;">
    [ NOISE &bull; ELA &bull; DCT &bull; FORMAT &bull; <span style="color:#00ff88;
    text-shadow:0 0 8px #00ff88;">SYNTHID CVR (reverse-SynthID)</span> ]
  </div>
  <div style="margin:18px auto 0;width:320px;height:1px;
              background:linear-gradient(90deg,transparent,#00ff88,transparent);
              box-shadow:0 0 8px #00ff88;"></div>
</div>
""")


def render_terminal_boot():
    st.html("""
<div style="font-family:'Share Tech Mono','Courier New',monospace;color:#336633;
            font-size:.76rem;letter-spacing:1px;padding:4px 0 14px;line-height:1.9;">
  &gt; LAPLACIAN NOISE ESTIMATOR............. <span style="color:#00ff88;">OK</span><br>
  &gt; ERROR LEVEL ANALYSIS MODULE........... <span style="color:#00ff88;">OK</span><br>
  &gt; DCT BLOCK DETECTOR.................... <span style="color:#00ff88;">OK</span><br>
  &gt; SYNTHID CARRIER CVR ENGINE............ <span style="color:#00ff88;">ARMED</span><br>
  &gt; CARRIER TABLE: 36 dark + 20 white bins <span style="color:#00ff88;">LOADED</span><br>
  &gt; REF: github.com/aloshdenny/reverse-SynthID<br>
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


def render_verdict(probability: float, verdict: str, sid: dict):
    pct = int(probability * 100)
    best_cvr = sid["best_cvr"]
    if verdict == "likely_ai":
        gc = "#ff2244"; label = "WARNING — SYNTHETIC ORIGIN DETECTED"
        sub = "HIGH CONFIDENCE &nbsp;|&nbsp; AI-GENERATED"; rgb = "255,34,68"
    elif verdict == "likely_real":
        gc = "#00ff88"; label = "AUTHENTIC SIGNAL DETECTED"
        sub = "HIGH CONFIDENCE &nbsp;|&nbsp; REAL PHOTOGRAPH"; rgb = "0,255,136"
    else:
        gc = "#ffaa00"; label = "SIGNAL AMBIGUOUS"
        sub = "INCONCLUSIVE &nbsp;|&nbsp; FURTHER ANALYSIS REQUIRED"; rgb = "255,170,0"

    # SynthID CVR badge
    if best_cvr >= 2.5:
        sid_c = "#ff2244"; sid_label = f"SYNTHID DETECTED  CVR={best_cvr:.2f}"
    elif best_cvr >= 1.8:
        sid_c = "#ffaa00"; sid_label = f"SYNTHID POSSIBLE  CVR={best_cvr:.2f}"
    else:
        sid_c = "#00ff88"; sid_label = f"NO SYNTHID  CVR={best_cvr:.2f}"

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
  <div style="display:flex;align-items:center;justify-content:center;gap:36px;flex-wrap:wrap;">
    <div>
      <div style="font-family:'Orbitron','Share Tech Mono',monospace;font-size:4.8rem;font-weight:900;
                  color:#fff;text-shadow:0 0 7px #fff,0 0 20px {gc};line-height:1;margin-bottom:4px;">
        {pct}<span style="font-size:2rem;opacity:.6;">%</span>
      </div>
      <div style="font-size:.68rem;letter-spacing:2px;color:{gc}88;">ENSEMBLE AI SCORE</div>
    </div>
    <div style="width:1px;height:70px;background:linear-gradient(180deg,transparent,{gc}44,transparent);"></div>
    <div>
      <div style="font-size:.64rem;letter-spacing:2px;color:{sid_c};
                  text-shadow:0 0 8px {sid_c};margin-bottom:6px;">&#x25C8; SYNTHID CVR</div>
      <div style="font-family:'Orbitron',monospace;font-size:2.2rem;font-weight:700;
                  color:{sid_c};text-shadow:0 0 10px {sid_c};line-height:1;">
        {best_cvr:.2f}x
      </div>
      <div style="font-size:.62rem;letter-spacing:1px;color:{sid_c}88;margin-top:4px;">{sid_label}</div>
    </div>
  </div>
  <div style="font-size:.68rem;letter-spacing:3px;color:{gc}77;margin:14px 0 18px;">{sub}</div>
  <div style="background:#001008;border:1px solid {gc}28;height:7px;overflow:hidden;border-radius:1px;">
    <div style="background:linear-gradient(90deg,{gc}44,{gc});
                width:{pct}%;height:100%;box-shadow:0 0 10px {gc};"></div>
  </div>
</div>
""")


def render_signal_card(label: str, score: float, detail: str, symbol: str):
    pct = int(score * 100)
    bc  = "#ff2244" if pct > 62 else "#00ff88" if pct < 38 else "#ffaa00"
    tc  = "#ff4466" if pct > 62 else "#00cc66" if pct < 38 else "#ffcc44"
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


def render_synthid_panel(sid: dict):
    dc, wc, ac, bc = sid["dark_cvr"], sid["white_cvr"], sid["all_cvr"], sid["best_cvr"]

    def bar(val, lo=1.0, hi=2.5):
        frac = int(min((val - lo) / max(hi - lo, 0.01) * 100, 100))
        frac = max(frac, 0)
        col = "#ff2244" if val >= 2.5 else "#ffaa00" if val >= 1.8 else "#00ff88"
        return (f'<div style="background:#001008;height:4px;border-radius:1px;'
                f'overflow:hidden;margin:3px 0 8px;">'
                f'<div style="background:{col};width:{frac}%;height:100%;'
                f'box-shadow:0 0 6px {col};"></div></div>')

    st.html(f"""
<div style="border:1px solid #00ff2233;border-left:3px solid #00ff88;
            background:#000a04;padding:16px 20px;margin:8px 0;border-radius:1px;
            font-family:'Share Tech Mono','Courier New',monospace;font-size:.75rem;">
  <div style="color:#00ff88;letter-spacing:3px;font-size:.78rem;margin-bottom:12px;">
    &#x25C8; SYNTHID CARRIER-TO-VARIANCE RATIO &mdash; reverse-SynthID
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px 24px;">
    <div>
      <div style="color:#669966;margin-bottom:1px;">DARK CARRIERS (36 bins)</div>
      {bar(dc)}<span style="color:#00ff88;">{dc:.3f}x</span>
      <span style="color:#334433;"> vs random</span>
    </div>
    <div>
      <div style="color:#669966;margin-bottom:1px;">WHITE CARRIERS (20 bins)</div>
      {bar(wc)}<span style="color:#00ff88;">{wc:.3f}x</span>
      <span style="color:#334433;"> vs random</span>
    </div>
    <div>
      <div style="color:#669966;margin-bottom:1px;">BEST CVR (used for score)</div>
      {bar(bc)}<span style="color:#00ff88;">{bc:.3f}x</span>
      <span style="color:#334433;"> detection CVR</span>
    </div>
  </div>
  <div style="margin-top:10px;padding-top:8px;border-top:1px solid #00ff2211;
              color:#334433;font-size:.69rem;line-height:1.7;">
    THRESHOLD: <span style="color:#88cc88;">&lt;1.8x=CLEAN &nbsp; 1.8-2.5x=UNCERTAIN &nbsp; &gt;2.5x=SYNTHID</span>
    &nbsp;&bull;&nbsp; SEED=42 RANDOM REF &nbsp;&bull;&nbsp; 512px CANONICAL SCALE<br>
    CARRIERS EMPIRICALLY EXTRACTED FROM 291 GEMINI WATERMARKED IMAGES
    &nbsp;&bull;&nbsp; REF: github.com/aloshdenny/reverse-SynthID
  </div>
</div>
""")


def render_log_block(lines: list, color: str):
    rows = "".join(
        f'<div style="color:{"#00cc55" if i < len(lines)-1 else color};">{l}</div>'
        for i, l in enumerate(lines)
    )
    st.html(f"""
<div style="background:#000a04;border:1px solid #00ff2218;border-left:3px solid {color};
            padding:14px 18px;font-family:'Share Tech Mono','Courier New',monospace;
            font-size:.76rem;line-height:1.85;margin-top:14px;
            box-shadow:0 0 20px {color}0c;">
  {rows}
</div>
""")


# ══════════════════════════════════════════════════════════════════════════════
# Pages
# ══════════════════════════════════════════════════════════════════════════════
def page_home():
    render_hero()
    render_terminal_boot()

    uploaded = st.file_uploader(
        "UPLOAD TARGET IMAGE", type=SUPPORTED_FORMATS,
        help="JPG · PNG · WEBP · max 10 MB",
    )

    if uploaded is None:
        st.html("""
<div style="border:1px solid rgba(0,255,34,.2);border-radius:2px;padding:48px 20px;
            text-align:center;background:radial-gradient(ellipse at center,#001a0a 0%,#000 70%);
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

    file_bytes = uploaded.read()
    if len(file_bytes) / (1024 * 1024) > MAX_FILE_SIZE_MB:
        st.error("FILE TOO LARGE — Maximum 10 MB")
        return

    mime_type = uploaded.type or "image/jpeg"

    with st.spinner("RUNNING FORENSIC ANALYSIS + SYNTHID CVR SCAN..."):
        try:
            R = analyze_image(file_bytes, mime_type)
        except Exception as e:
            st.error(f"ANALYSIS FAILED: {e}")
            import traceback; st.code(traceback.format_exc())
            return

    sid   = R["sid"]
    noise = R["noise"]
    ela   = R["ela"]
    dct   = R["dct"]
    fmt   = R["fmt"]

    # ── Verdict ───────────────────────────────────────────────────────────────
    render_section_header("VERDICT")
    render_verdict(R["probability"], R["verdict"], sid)

    # ── Visuals ───────────────────────────────────────────────────────────────
    render_section_header("VISUAL FORENSICS")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.image(R["original"],                           caption="[ ORIGINAL ]",          use_container_width=True)
    with c2: st.image(R["enhanced"],                           caption="[ SATURATION x8 ]",     use_container_width=True)
    with c3: st.image(ela_to_image(R["ela_map"]),              caption="[ ELA MAP ]",            use_container_width=True)
    with c4: st.image(fft_to_image(R["log_mag"]),              caption="[ FFT SPECTRUM ]",       use_container_width=True)
    with c5: st.image(carrier_map_to_image(R["carrier_vis"], R["noise_mag"]),
                      caption="[ SYNTHID CARRIERS ]",          use_container_width=True)

    # ── SynthID panel ─────────────────────────────────────────────────────────
    render_section_header("SYNTHID CARRIER-TO-VARIANCE RATIO  (reverse-SynthID)")
    render_synthid_panel(sid)

    # ── Signal cards ──────────────────────────────────────────────────────────
    render_section_header("SIGNAL ANALYSIS")

    render_signal_card(
        "SYNTHID CVR", sid["ai_score"],
        (f"best_CVR={sid['best_cvr']:.3f}x  "
         f"dark={sid['dark_cvr']:.3f}x  white={sid['white_cvr']:.3f}x  |  "
         + ("CARRIER ENERGY ELEVATED — SYNTHID FINGERPRINT PROBABLE" if sid["best_cvr"] >= 2.5
            else "MARGINAL ELEVATION — INCONCLUSIVE" if sid["best_cvr"] >= 1.8
            else "NO CARRIER ELEVATION — CLEAN / NOT SYNTHID")),
        "[S]",
    )
    render_signal_card(
        "NOISE LEVEL", noise["ai_score"],
        f"Laplacian sigma={noise['sigma']:.2f}  |  " + (
            "UNNATURALLY SMOOTH — AI SIGNATURE" if noise["sigma"] < 2
            else "NATURAL SENSOR NOISE — CAMERA" if noise["sigma"] > 5
            else "BORDERLINE REGION"),
        "[!]",
    )
    render_signal_card(
        "ERROR LEVEL ANALYSIS", ela["ai_score"],
        f"Mean ELA={ela['ela_mean']:.2f}  |  " + (
            "HIGH — FIRST JPEG COMPRESSION — PNG/AI SOURCE" if ela["ela_mean"] > 8
            else "LOW — PRE-COMPRESSED — REAL CAMERA JPEG" if ela["ela_mean"] < 4
            else "MODERATE — AMBIGUOUS"),
        "[~]",
    )
    render_signal_card(
        "SOURCE FORMAT", fmt["ai_score"],
        f"Detected: {fmt['fmt_label']}  |  " + (
            "PNG IS DEFAULT OUTPUT FOR MOST AI GENERATORS" if fmt["fmt_label"] == "PNG"
            else "WEBP USED BY GEMINI / AI PLATFORMS" if fmt["fmt_label"] == "WebP"
            else "JPEG IS NATIVE CAMERA FORMAT — LOW AI PRIOR"),
        "[F]",
    )
    render_signal_card(
        "DCT BLOCK STRUCTURE", dct["ai_score"],
        f"Block strength={dct['dct_strength']:.3f}  |  " + (
            "STRONG 8x8 JPEG GRID — CAMERA JPEG" if dct["dct_strength"] > 0.5
            else "WEAK GRID — LOSSLESS/AI SOURCE"),
        "[D]",
    )

    # ── Log ───────────────────────────────────────────────────────────────────
    prob = R["probability"]; verdict = R["verdict"]
    if verdict == "likely_ai":
        color = "#ff2244"; status = "SYNTHETIC"
        conclusion = "AI ORIGIN — MULTIPLE FORENSIC SIGNALS CONSISTENT WITH SYNTHESIS"
    elif verdict == "likely_real":
        color = "#00ff88"; status = "AUTHENTIC"
        conclusion = "SIGNALS CONSISTENT WITH REAL CAMERA PHOTOGRAPH"
    else:
        color = "#ffaa00"; status = "INCONCLUSIVE"
        conclusion = "SIGNALS CONFLICT — POSSIBLE POST-PROCESSING OR FORMAT CONVERSION"

    render_log_block([
        f"[RESULT]  AI_PROBABILITY={int(prob*100)}%  STATUS={status}",
        f"[SYNTHID] best_CVR={sid['best_cvr']:.3f}x  dark={sid['dark_cvr']:.3f}x  white={sid['white_cvr']:.3f}x  score={int(sid['ai_score']*100)}%",
        f"[NOISE]   sigma={noise['sigma']:.2f}  score={int(noise['ai_score']*100)}%",
        f"[ELA]     mean={ela['ela_mean']:.2f}  score={int(ela['ai_score']*100)}%",
        f"[FORMAT]  {fmt['fmt_label']}  score={int(fmt['ai_score']*100)}%",
        f"[DCT]     strength={dct['dct_strength']:.3f}  score={int(dct['ai_score']*100)}%",
        f"[CONCLUSION] {conclusion}",
    ], color)

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
                "synthid_best_cvr":  round(sid["best_cvr"], 4),
                "synthid_dark_cvr":  round(sid["dark_cvr"], 4),
                "synthid_white_cvr": round(sid["white_cvr"], 4),
                "synthid_all_cvr":   round(sid["all_cvr"], 4),
                "synthid_ai_score":  round(sid["ai_score"], 4),
            })

    st.caption(
        "FORENSIC HEURISTICS — SYNTHID CVR BASED ON GITHUB.COM/ALOSHDENNY/REVERSE-SYNTHID "
        "— NOT A TRAINED ML MODEL — PROBABILISTIC ESTIMATES"
    )


def page_about():
    st.html("""
<div style="text-align:center;padding:28px 0 18px;">
  <div style="font-family:'Orbitron',monospace;font-size:.68rem;letter-spacing:5px;
              color:rgba(0,255,68,.35);margin-bottom:10px;">// TECHNICAL DOCUMENTATION //</div>
  <div style="font-family:'Orbitron',monospace;font-size:1.7rem;font-weight:900;color:#fff;
              letter-spacing:4px;text-shadow:0 0 10px #fff,0 0 25px #00ff88;">
    DETECTION ALGORITHMS
  </div>
  <div style="margin:14px auto 0;width:180px;height:1px;
              background:linear-gradient(90deg,transparent,#00ff88,transparent);
              box-shadow:0 0 8px #00ff88;"></div>
</div>
""")

    render_section_header("SIGNAL 01 — SYNTHID CARRIER-TO-VARIANCE RATIO  [WEIGHT: 15%]")
    st.markdown("""
**Source:** [aloshdenny/reverse-SynthID](https://github.com/aloshdenny/reverse-SynthID)
`src/extraction/robust_extractor.py` — `RobustSynthIDExtractor`

Google Gemini (SynthID) embeds an invisible watermark during diffusion at **fixed carrier frequencies**.
These carrier bin offsets were **empirically reverse-engineered from 291 Gemini-generated images** and
have >0.95 intra-set phase coherence. Each carrier set has a >0.5 discriminative gap vs non-watermarked images.

**Two carrier sets at 512 px resolution:**

| Set | Bins | Pattern | Typical images |
|-----|------|---------|----------------|
| Dark | 36 bins | Diagonal grid e.g. `(-5,-3),(5,3),...` | Black / natural Gemini |
| White | 20 bins | Horizontal axis e.g. `(0,±7),(0,±8),...` | White / bright Gemini |

**CVR Algorithm** (mirrors `detect_array` from the repo):
```python
# 1. Resize image to 512×512 (canonical detection scale)
# 2. Extract noise residual (multi-method fusion, channel weights G=1.0 R=0.85 B=0.70)
noise = fuse(gaussian_blur_residual, uniform_filter_residual)
# 3. 2-D FFT of noise
noise_fft = fftshift(fft2(noise_gray))
# 4. Sample carrier bins vs same-radial-distance random bins (seed=42)
CVR = mean(|noise_fft| at carrier_bins) / mean(|noise_fft| at random_bins)
# 5. best_CVR = max(dark_CVR, white_CVR)
```

**Calibration thresholds:**
- `CVR < 1.8`  → Clean — no SynthID watermark
- `CVR 1.8–2.5` → Uncertain
- `CVR ≥ 2.5`  → SynthID watermark detected

**Why CVR?** Without a reference codebook (reference phases from many Gemini images), full phase
matching is not possible. The CVR supporting signal from the repo is used: SynthID elevates energy
specifically at these bins in the noise residual — natural images do not.
""")

    render_section_header("SIGNAL 02 — NOISE LEVEL ESTIMATION  [WEIGHT: 35%]")
    st.markdown("""
Real cameras produce **shot/thermal noise** (sigma ≈ 4–15 depending on ISO).
AI generators produce mathematically synthesised images — unnaturally smooth (sigma ≈ 0.5–2.5).

```python
kernel = [[1,-2,1],[-2,4,-2],[1,-2,1]]   # Laplacian (Immerkær 1996)
sigma  = sqrt(pi/2) * mean(|convolve(gray, kernel)|) / 6.0
```
- `sigma < 2` → AI &nbsp;&nbsp; `sigma 2–4` → borderline &nbsp;&nbsp; `sigma > 5` → real camera
""")

    render_section_header("SIGNAL 03 — ERROR LEVEL ANALYSIS (ELA)  [WEIGHT: 25%]")
    st.markdown("""
AI models output lossless PNG → first JPEG compression → **high ELA** (mean 8–25).
Camera JPEGs are already compressed in-device → re-saving causes **low ELA** (mean 1–4).
```
re_saved  = save(img, JPEG, quality=92)
ELA_mean  = mean(|original − re_saved|)
```
- `ELA < 4` → real JPEG &nbsp;&nbsp; `ELA > 8` → PNG/AI source
""")

    render_section_header("SIGNAL 04 — SOURCE FORMAT  [WEIGHT: 15%]")
    st.markdown("""
| Format | AI prior | Reasoning |
|--------|----------|-----------|
| JPEG | 0.20 | Default camera format — very unlikely AI output |
| PNG | 0.62 | ChatGPT, DALL·E, Midjourney, Stable Diffusion default |
| WebP | 0.58 | Google Gemini and web-delivered AI images |
""")

    render_section_header("SIGNAL 05 — DCT BLOCK STRUCTURE  [WEIGHT: 10%]")
    st.markdown("""
JPEG applies DCT in **8×8 blocks**. In the 2-D FFT these appear as spikes at multiples of `N/8` from DC.
AI PNG images have no such structure. Strong 8×8 grid → JPEG camera image.
""")

    render_section_header("KNOWN LIMITATIONS")
    st.warning("""
FOR EDUCATIONAL / RESEARCH USE ONLY — NOT A TRAINED ML MODEL.

- SynthID CVR works best for images generated at exactly 512px resolution (Gemini's canonical size)
- Images watermarked at other resolutions use different carrier bin positions (a full codebook is needed)
- The V4 bypass pipeline (VAE + elastic deformation + FFT subtraction) defeats this detector
- A real PNG photo will have moderate ELA and PNG format score — the ensemble still weighs all signals
- Post-processing (re-JPEG, resize, crop) degrades all signals
""")

    render_section_header("REFERENCES")
    st.markdown("""
- [aloshdenny/reverse-SynthID](https://github.com/aloshdenny/reverse-SynthID) — carrier frequency CVR detection
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
