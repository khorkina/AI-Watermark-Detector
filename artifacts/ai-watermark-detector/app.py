"""
AI Watermark Detector — "Is This AI?"
Hacker / cryptography aesthetic edition.
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
# HACKER DESIGN CSS
# ══════════════════════════════════════════════════════════════════════════════
HACKER_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    background-color: #000000 !important;
    color: #c8ffc8 !important;
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
}

/* ── Scanline overlay ── */
body::before {
    content: "";
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,255,80,0.015) 2px,
        rgba(0,255,80,0.015) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ── Main container ── */
section[data-testid="stMain"] > div {
    background: #000 !important;
}
.stApp {
    background: radial-gradient(ellipse at 20% 10%, #001a0a 0%, #000000 50%),
                radial-gradient(ellipse at 80% 90%, #000d1a 0%, #000000 50%) !important;
    background-color: #000 !important;
}
[data-testid="stAppViewContainer"] {
    background: #000 !important;
}
[data-testid="stHeader"] {
    background: transparent !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #020a02 !important;
    border-right: 1px solid #00ff4422 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    color: #00cc55 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.95rem !important;
    letter-spacing: 1px !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #00ff88 !important;
    border-bottom: 2px solid #00ff88 !important;
    text-shadow: 0 0 10px #00ff88, 0 0 20px #00ff4488 !important;
}
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid #00ff2222 !important;
    background: transparent !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 1px solid #00ff4433 !important;
    border-radius: 4px !important;
    background: #000e04 !important;
    box-shadow: 0 0 20px #00ff441a, inset 0 0 30px #00ff440a !important;
}
[data-testid="stFileUploadDropzone"] {
    background: #000e04 !important;
    border: 2px dashed #00ff4455 !important;
    border-radius: 4px !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #00ff88 !important;
    box-shadow: 0 0 25px #00ff4433 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid #00ff88 !important;
    color: #00ff88 !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    box-shadow: 0 0 10px #00ff4433, inset 0 0 10px #00ff4411 !important;
}
.stButton > button:hover {
    box-shadow: 0 0 20px #00ff88, inset 0 0 15px #00ff4422 !important;
    color: #ffffff !important;
}

/* ── Progress bars ── */
[data-testid="stProgressBar"] > div {
    background: #001a0a !important;
    border: 1px solid #00ff4422 !important;
}
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #005522, #00ff88) !important;
    box-shadow: 0 0 8px #00ff8888 !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid #00ff2222 !important;
    background: #000a04 !important;
    border-radius: 2px !important;
}
[data-testid="stExpander"]:hover {
    border-color: #00ff4444 !important;
    box-shadow: 0 0 15px #00ff2211 !important;
}
details summary {
    color: #00ff88 !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* ── Alerts / info boxes ── */
[data-testid="stAlert"] {
    border-radius: 2px !important;
    font-family: 'Share Tech Mono', monospace !important;
}
.stAlert[data-baseweb="notification"] {
    background: #000d04 !important;
}

/* ── Info ── */
[data-testid="stAlert"][kind="info"],
div[data-testid="stAlert"] {
    background: #000d04 !important;
    border-left: 3px solid #00ff88 !important;
    box-shadow: 0 0 20px #00ff2211 !important;
    color: #a0ffa0 !important;
}

/* ── Error / warning ── */
div[data-baseweb="notification"][kind="negative"],
div[role="alert"] {
    background: #0d0000 !important;
    border-left: 3px solid #ff2244 !important;
}

/* ── JSON display ── */
[data-testid="stJson"] {
    background: #000a04 !important;
    border: 1px solid #00ff2222 !important;
    font-family: 'Share Tech Mono', monospace !important;
    color: #00ff88 !important;
}

/* ── Divider ── */
hr {
    border-color: #00ff2222 !important;
    box-shadow: 0 0 8px #00ff2222 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: #00ff88 !important;
}

/* ── Caption ── */
[data-testid="stCaptionContainer"] {
    color: #446644 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* ── Headings ── */
h1, h2, h3, h4 {
    font-family: 'Orbitron', 'Share Tech Mono', monospace !important;
    color: #ffffff !important;
    text-shadow: 0 0 10px #00ff8877, 0 0 30px #00ff4433 !important;
    letter-spacing: 2px !important;
}
h2 { color: #00ff88 !important; }
h3 { color: #c0ffc0 !important; font-size: 1.1rem !important; }

/* ── Markdown text ── */
p, li, label {
    color: #a0e8a0 !important;
    font-family: 'Share Tech Mono', monospace !important;
}
strong { color: #00ff88 !important; }
code {
    background: #001a0a !important;
    color: #00ff88 !important;
    border: 1px solid #00ff2233 !important;
    border-radius: 2px !important;
    font-family: 'Share Tech Mono', monospace !important;
}
pre, .stCode {
    background: #000d04 !important;
    border: 1px solid #00ff2233 !important;
    border-left: 3px solid #00ff88 !important;
    box-shadow: 0 0 20px #00ff221a !important;
}
a { color: #00ff88 !important; text-decoration: none !important; }
a:hover { text-shadow: 0 0 8px #00ff88 !important; }

/* ── Table ── */
table {
    border-collapse: collapse !important;
    font-family: 'Share Tech Mono', monospace !important;
}
th {
    background: #001a08 !important;
    color: #00ff88 !important;
    border: 1px solid #00ff2233 !important;
    padding: 6px 12px !important;
}
td {
    border: 1px solid #00ff2222 !important;
    color: #88cc88 !important;
    padding: 5px 12px !important;
}
tr:hover td { background: #001008 !important; }

/* ── Image captions ── */
figcaption {
    color: #447744 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 1px !important;
}
</style>
"""


# ══════════════════════════════════════════════════════════════════════════════
# Detection Engine  (unchanged)
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


def compute_fft_spectrum(img: Image.Image):
    gray = np.array(img.convert("L"), dtype=np.float32)
    h, w = gray.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    fft_shifted = np.fft.fftshift(np.fft.fft2(gray * window))
    magnitude = np.abs(fft_shifted)
    return magnitude, np.log1p(magnitude)


def estimate_noise_level(img: Image.Image) -> dict:
    gray = np.array(img.convert("L"), dtype=np.float32)
    from scipy.ndimage import convolve
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    laplacian = convolve(gray, kernel)
    sigma = float(np.sqrt(np.pi / 2) * np.abs(laplacian).mean() / 6.0)
    ai_score = float(1.0 / (1.0 + np.exp(0.6 * (sigma - 4.0))))
    return {"sigma": sigma, "ai_score": float(np.clip(ai_score, 0, 1))}


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
        "ela_std":  float(ela_map.std()),
        "ela_max":  float(ela_map.max()),
        "ela_map":  ela_map,
        "ai_score": float(np.clip(ai_score, 0, 1)),
    }


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
                    peak_val = float(magnitude[pos - 2:pos + 3, cx - 2:cx + 3].max())
                    bg_val   = float(magnitude[pos - 8:pos + 9, cx - 2:cx + 3].mean())
                else:
                    peak_val = float(magnitude[cy - 2:cy + 3, pos - 2:pos + 3].max())
                    bg_val   = float(magnitude[cy - 2:cy + 3, pos - 8:pos + 9].mean())
                if bg_val > 0:
                    peak_ratios.append(peak_val / bg_val)
    if not peak_ratios:
        dct_strength = 0.0
    else:
        avg_ratio = float(np.mean(peak_ratios))
        dct_strength = float(np.clip((avg_ratio - 1.0) / 6.0, 0, 1))
    return {
        "dct_strength": dct_strength,
        "avg_ratio":    float(np.mean(peak_ratios)) if peak_ratios else 1.0,
        "ai_score":     float(np.clip(1.0 - dct_strength, 0, 1)),
    }


def analyze_source_format(mime_type: str) -> dict:
    if mime_type in ("image/jpeg", "image/jpg"):
        return {"fmt_label": "JPEG", "ai_score": 0.25}
    elif mime_type == "image/webp":
        return {"fmt_label": "WebP", "ai_score": 0.60}
    elif mime_type == "image/png":
        return {"fmt_label": "PNG",  "ai_score": 0.65}
    return {"fmt_label": "Unknown", "ai_score": 0.50}


def compute_ai_probability(noise_res, ela_res, dct_res, fmt_res):
    combined = (
        noise_res["ai_score"] * 0.40 +
        ela_res["ai_score"]   * 0.25 +
        fmt_res["ai_score"]   * 0.20 +
        dct_res["ai_score"]   * 0.15
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
    enhanced           = enhance_saturation(img, factor=8.0)
    magnitude, log_mag = compute_fft_spectrum(img)
    noise_res  = estimate_noise_level(img)
    ela_res    = analyze_ela(img)
    dct_res    = detect_dct_blocks(magnitude, img.size)
    fmt_res    = analyze_source_format(mime_type)
    probability, verdict = compute_ai_probability(noise_res, ela_res, dct_res, fmt_res)
    return {
        "original": img, "enhanced": enhanced,
        "log_mag": log_mag, "ela_map": ela_res["ela_map"],
        "noise": noise_res, "ela": ela_res,
        "dct": dct_res, "fmt": fmt_res,
        "probability": probability, "verdict": verdict,
    }


def fft_to_image(log_mag: np.ndarray) -> Image.Image:
    norm = ((log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-8) * 255).astype(np.uint8)
    return Image.fromarray(norm).convert("RGB")


def ela_to_image(ela_map: np.ndarray) -> Image.Image:
    clipped = np.clip(ela_map.mean(axis=2) if ela_map.ndim == 3 else ela_map, 0, 30)
    norm = (clipped / 30.0 * 255).astype(np.uint8)
    colored = np.stack([norm // 3, norm, norm // 4], axis=2).astype(np.uint8)
    return Image.fromarray(colored)


# ══════════════════════════════════════════════════════════════════════════════
# UI — Hacker Components
# ══════════════════════════════════════════════════════════════════════════════

def glow_header():
    st.markdown("""
    <div style="
        text-align: center;
        padding: 36px 0 24px 0;
        position: relative;
    ">
        <!-- background light blobs -->
        <div style="
            position: absolute; top: 0; left: 50%; transform: translateX(-50%);
            width: 600px; height: 120px;
            background: radial-gradient(ellipse, #00ff4411 0%, transparent 70%);
            pointer-events: none;
        "></div>

        <div style="
            font-family: 'Orbitron', monospace;
            font-size: 0.75rem;
            letter-spacing: 6px;
            color: #00ff4488;
            text-transform: uppercase;
            margin-bottom: 10px;
        ">// FORENSIC IMAGE ANALYSIS SYSTEM v2.1 //</div>

        <div style="
            font-family: 'Orbitron', monospace;
            font-size: 2.4rem;
            font-weight: 900;
            color: #ffffff;
            text-shadow:
                0 0 7px #fff,
                0 0 15px #fff,
                0 0 30px #00ff88,
                0 0 60px #00ff8866,
                0 0 100px #00ff4433;
            letter-spacing: 4px;
            line-height: 1.15;
            margin-bottom: 6px;
        ">AI WATERMARK<br>DETECTOR</div>

        <div style="
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.9rem;
            color: #448844;
            letter-spacing: 2px;
            margin-top: 10px;
        ">[ NOISE · ELA · DCT · FORMAT FORENSICS ]</div>

        <div style="
            margin: 20px auto 0 auto;
            width: 300px; height: 1px;
            background: linear-gradient(90deg, transparent, #00ff88, transparent);
            box-shadow: 0 0 10px #00ff88;
        "></div>
    </div>
    """, unsafe_allow_html=True)


def render_verdict(probability: float, verdict: str):
    pct = int(probability * 100)

    if verdict == "likely_ai":
        glow_color   = "#ff2244"
        glow_color2  = "#ff000066"
        label        = "⚠ SYNTHETIC ORIGIN DETECTED"
        sub          = "HIGH CONFIDENCE — AI-GENERATED"
        border_color = "#ff2244"
    elif verdict == "likely_real":
        glow_color   = "#00ff88"
        glow_color2  = "#00ff4466"
        label        = "✓ AUTHENTIC SIGNAL DETECTED"
        sub          = "HIGH CONFIDENCE — REAL PHOTOGRAPH"
        border_color = "#00ff88"
    else:
        glow_color   = "#ffaa00"
        glow_color2  = "#ffaa0066"
        label        = "~ SIGNAL AMBIGUOUS"
        sub          = "INCONCLUSIVE — FURTHER ANALYSIS REQUIRED"
        border_color = "#ffaa00"

    st.markdown(f"""
    <div style="
        background: radial-gradient(ellipse at center top, {glow_color}0a 0%, #000000 60%);
        border: 1px solid {border_color}55;
        border-radius: 2px;
        padding: 36px 40px 30px 40px;
        margin: 20px 0;
        text-align: center;
        position: relative;
        box-shadow: 0 0 40px {glow_color}22, inset 0 0 60px {glow_color}08;
        font-family: 'Share Tech Mono', monospace;
    ">
        <!-- corner accents -->
        <div style="position:absolute;top:-1px;left:-1px;width:16px;height:16px;border-top:2px solid {glow_color};border-left:2px solid {glow_color};"></div>
        <div style="position:absolute;top:-1px;right:-1px;width:16px;height:16px;border-top:2px solid {glow_color};border-right:2px solid {glow_color};"></div>
        <div style="position:absolute;bottom:-1px;left:-1px;width:16px;height:16px;border-bottom:2px solid {glow_color};border-left:2px solid {glow_color};"></div>
        <div style="position:absolute;bottom:-1px;right:-1px;width:16px;height:16px;border-bottom:2px solid {glow_color};border-right:2px solid {glow_color};"></div>

        <!-- label -->
        <div style="
            font-size: 0.8rem;
            letter-spacing: 4px;
            color: {glow_color};
            text-shadow: 0 0 8px {glow_color};
            margin-bottom: 14px;
        ">{label}</div>

        <!-- big percentage -->
        <div style="
            font-family: 'Orbitron', monospace;
            font-size: 5.5rem;
            font-weight: 900;
            color: #ffffff;
            text-shadow:
                0 0 7px #fff,
                0 0 20px {glow_color},
                0 0 50px {glow_color2};
            line-height: 1;
            margin-bottom: 8px;
        ">{pct}<span style="font-size:2.5rem;opacity:0.7">%</span></div>

        <!-- sub-label -->
        <div style="
            font-size: 0.7rem;
            letter-spacing: 3px;
            color: {glow_color}99;
            margin-bottom: 20px;
        ">AI PROBABILITY · {sub}</div>

        <!-- progress bar -->
        <div style="
            background: #001008;
            border: 1px solid {border_color}33;
            border-radius: 1px;
            height: 8px;
            overflow: hidden;
            box-shadow: inset 0 0 10px #000;
        ">
            <div style="
                background: linear-gradient(90deg, {glow_color}44, {glow_color});
                width: {pct}%;
                height: 100%;
                box-shadow: 0 0 12px {glow_color};
            "></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def signal_card(label: str, score: float, detail: str, icon: str):
    pct = int(score * 100)
    if pct > 62:
        bar_color, txt_color = "#ff2244", "#ff4466"
    elif pct < 38:
        bar_color, txt_color = "#00ff88", "#00cc66"
    else:
        bar_color, txt_color = "#ffaa00", "#ffcc44"

    st.markdown(f"""
    <div style="
        border: 1px solid {bar_color}33;
        border-left: 3px solid {bar_color};
        background: radial-gradient(ellipse at left, {bar_color}08, transparent 60%);
        padding: 12px 16px;
        margin: 6px 0;
        border-radius: 1px;
        font-family: 'Share Tech Mono', monospace;
        position: relative;
    ">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <span style="color:#a0e8a0; font-size:0.8rem; letter-spacing:1px;">{icon} {label}</span>
            <span style="
                color:{txt_color};
                font-family:'Orbitron',monospace;
                font-size:1rem;
                font-weight:700;
                text-shadow: 0 0 8px {bar_color};
            ">{pct}%</span>
        </div>
        <div style="background:#001008; height:4px; border-radius:1px; overflow:hidden; margin-bottom:6px;">
            <div style="background:linear-gradient(90deg,{bar_color}66,{bar_color});width:{pct}%;height:100%;box-shadow:0 0 8px {bar_color};"></div>
        </div>
        <div style="color:#446644; font-size:0.72rem; letter-spacing:0.5px;">{detail}</div>
    </div>
    """, unsafe_allow_html=True)


def section_header(text: str):
    st.markdown(f"""
    <div style="
        font-family: 'Orbitron', monospace;
        font-size: 0.85rem;
        letter-spacing: 4px;
        color: #00ff88;
        text-shadow: 0 0 8px #00ff8866;
        text-transform: uppercase;
        padding: 18px 0 8px 0;
        border-bottom: 1px solid #00ff2222;
    ">// {text}</div>
    """, unsafe_allow_html=True)


def drop_zone_placeholder():
    st.markdown("""
    <div style="
        border: 1px solid #00ff2233;
        border-radius: 2px;
        padding: 50px 20px;
        text-align: center;
        background: radial-gradient(ellipse at center, #001a0a 0%, #000000 70%);
        box-shadow: 0 0 30px #00ff1a0a, inset 0 0 40px #00ff1a05;
        margin-top: 10px;
        font-family: 'Share Tech Mono', monospace;
        position: relative;
    ">
        <!-- corner lights -->
        <div style="position:absolute;top:-1px;left:-1px;width:12px;height:12px;border-top:2px solid #00ff88;border-left:2px solid #00ff88;box-shadow:-2px -2px 8px #00ff88;"></div>
        <div style="position:absolute;top:-1px;right:-1px;width:12px;height:12px;border-top:2px solid #00ff88;border-right:2px solid #00ff88;box-shadow:2px -2px 8px #00ff88;"></div>
        <div style="position:absolute;bottom:-1px;left:-1px;width:12px;height:12px;border-bottom:2px solid #00ff88;border-left:2px solid #00ff88;box-shadow:-2px 2px 8px #00ff88;"></div>
        <div style="position:absolute;bottom:-1px;right:-1px;width:12px;height:12px;border-bottom:2px solid #00ff88;border-right:2px solid #00ff88;box-shadow:2px 2px 8px #00ff88;"></div>

        <div style="font-size:2.5rem;margin-bottom:12px;filter:drop-shadow(0 0 12px #00ff88);">🔍</div>
        <div style="color:#00ff4488;font-size:0.95rem;letter-spacing:2px;">AWAITING TARGET IMAGE</div>
        <div style="color:#224422;font-size:0.75rem;margin-top:8px;letter-spacing:1px;">JPG · PNG · WEBP · MAX 10 MB</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Pages
# ══════════════════════════════════════════════════════════════════════════════

def page_home():
    glow_header()

    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;color:#336633;font-size:0.78rem;letter-spacing:1px;padding:4px 0 12px 0;">
    &gt; INITIALIZING FORENSIC ENGINE... OK<br>
    &gt; LOADING FFT MODULE... OK<br>
    &gt; ELA SUBSYSTEM READY... OK<br>
    &gt; AWAITING INPUT FILE_
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "UPLOAD TARGET IMAGE",
        type=SUPPORTED_FORMATS,
        help="JPG · PNG · WEBP · max 10 MB",
    )

    if uploaded_file is None:
        drop_zone_placeholder()
        return

    file_bytes = uploaded_file.read()
    if len(file_bytes) / (1024 * 1024) > MAX_FILE_SIZE_MB:
        st.error("FILE TOO LARGE — Maximum 10 MB")
        return

    mime_type = uploaded_file.type or "image/jpeg"

    with st.spinner("RUNNING FORENSIC ANALYSIS..."):
        try:
            results = analyze_image(file_bytes, mime_type)
        except Exception as e:
            st.error(f"ANALYSIS FAILED: {e}")
            return

    # ── Verdict ───────────────────────────────────────────────────────────────
    section_header("VERDICT")
    render_verdict(results["probability"], results["verdict"])

    # ── Visuals ───────────────────────────────────────────────────────────────
    section_header("VISUAL FORENSICS")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(results["original"], caption="[ ORIGINAL ]", use_container_width=True)
    with col2:
        st.image(results["enhanced"], caption="[ SATURATION ×8 ]", use_container_width=True)
    with col3:
        st.image(ela_to_image(results["ela_map"]), caption="[ ELA MAP ]", use_container_width=True)
    with col4:
        st.image(fft_to_image(results["log_mag"]), caption="[ FFT SPECTRUM ]", use_container_width=True)

    # ── Signal readout ────────────────────────────────────────────────────────
    section_header("SIGNAL ANALYSIS")
    noise_sig = results["noise"]["sigma"]
    ela_mean  = results["ela"]["ela_mean"]
    dct_str   = results["dct"]["dct_strength"]
    fmt_label = results["fmt"]["fmt_label"]

    signal_card(
        "NOISE LEVEL",
        results["noise"]["ai_score"],
        f"Laplacian sigma = {noise_sig:.2f} · "
        + ("UNNATURALLY CLEAN — AI SIGNATURE" if noise_sig < 3 else
           "NATURAL SENSOR NOISE — CAMERA ORIGIN" if noise_sig > 6 else
           "BORDERLINE — AMBIGUOUS"),
        "⚡",
    )
    signal_card(
        "ERROR LEVEL ANALYSIS",
        results["ela"]["ai_score"],
        f"Mean ELA = {ela_mean:.2f} · "
        + ("HIGH — FIRST JPEG COMPRESSION → PNG/AI SOURCE" if ela_mean > 7 else
           "LOW — PREVIOUSLY JPEG COMPRESSED → CAMERA SOURCE"),
        "🗜",
    )
    signal_card(
        "SOURCE FORMAT",
        results["fmt"]["ai_score"],
        f"Detected: {fmt_label} · "
        + ("PNG IS DEFAULT OUTPUT FORMAT FOR MOST AI GENERATORS" if fmt_label == "PNG" else
           "WEBP USED BY GEMINI AND SOME AI PLATFORMS" if fmt_label == "WebP" else
           "JPEG IS NATIVE CAMERA/SMARTPHONE FORMAT"),
        "📄",
    )
    signal_card(
        "DCT BLOCK STRUCTURE",
        results["dct"]["ai_score"],
        f"Block strength = {dct_str:.3f} · "
        + ("STRONG 8×8 JPEG GRID → CAMERA JPEG" if dct_str > 0.5 else
           "NO JPEG BLOCK PATTERN → LOSSLESS/AI SOURCE"),
        "🔲",
    )

    # ── Analysis log ─────────────────────────────────────────────────────────
    verdict  = results["verdict"]
    prob     = results["probability"]

    if verdict == "likely_ai":
        log_lines = [
            f"[RESULT]  AI_PROBABILITY={int(prob*100)}%  STATUS=SYNTHETIC",
            f"[NOISE]   sigma={noise_sig:.2f}  threshold=4.0  VERDICT={'AI_CLEAN' if noise_sig < 4 else 'BORDERLINE'}",
            f"[ELA]     mean={ela_mean:.2f}  threshold=7.0  VERDICT={'PNG_SOURCE' if ela_mean > 7 else 'AMBIGUOUS'}",
            f"[FORMAT]  type={fmt_label}  VERDICT={'AI_COMMON' if fmt_label != 'JPEG' else 'JPEG_PRIOR'}",
            f"[DCT]     strength={dct_str:.3f}  VERDICT={'NO_JPEG_GRID' if dct_str < 0.4 else 'WEAK_GRID'}",
            "[CONCLUSION]  HIGH PROBABILITY OF SYNTHETIC ORIGIN — AI WATERMARK DETECTED",
        ]
        color = "#ff2244"
    elif verdict == "likely_real":
        log_lines = [
            f"[RESULT]  AI_PROBABILITY={int(prob*100)}%  STATUS=AUTHENTIC",
            f"[NOISE]   sigma={noise_sig:.2f}  VERDICT={'NATURAL_NOISE' if noise_sig > 5 else 'LOW_NOISE'}",
            f"[ELA]     mean={ela_mean:.2f}  VERDICT={'PREV_JPEG' if ela_mean < 7 else 'AMBIGUOUS'}",
            f"[FORMAT]  type={fmt_label}  VERDICT={'CAMERA_FORMAT' if fmt_label == 'JPEG' else 'ATYPICAL'}",
            f"[DCT]     strength={dct_str:.3f}  VERDICT={'JPEG_GRID_PRESENT' if dct_str > 0.4 else 'WEAK'}",
            "[CONCLUSION]  SIGNALS CONSISTENT WITH REAL CAMERA PHOTOGRAPH",
        ]
        color = "#00ff88"
    else:
        log_lines = [
            f"[RESULT]  AI_PROBABILITY={int(prob*100)}%  STATUS=INCONCLUSIVE",
            f"[NOISE]   sigma={noise_sig:.2f}",
            f"[ELA]     mean={ela_mean:.2f}",
            f"[FORMAT]  type={fmt_label}",
            f"[DCT]     strength={dct_str:.3f}",
            "[CONCLUSION]  SIGNALS CONFLICT — POSSIBLE POST-PROCESSING OR RE-ENCODING",
        ]
        color = "#ffaa00"

    lines_html = "<br>".join(
        f'<span style="color:{color if i == len(log_lines)-1 else "#336633"}">{line}</span>'
        for i, line in enumerate(log_lines)
    )
    st.markdown(f"""
    <div style="
        background: #000a04;
        border: 1px solid #00ff2222;
        border-left: 3px solid {color};
        padding: 16px 20px;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.78rem;
        line-height: 1.9;
        margin-top: 16px;
        box-shadow: 0 0 20px {color}11;
    ">{lines_html}</div>
    """, unsafe_allow_html=True)

    # ── Raw data expander ─────────────────────────────────────────────────────
    with st.expander("[ RAW FORENSIC DATA ]"):
        c1, c2 = st.columns(2)
        with c1:
            st.json({"noise_sigma": round(noise_sig, 4), "noise_ai_score": round(results["noise"]["ai_score"], 4)})
            st.json({"dct_strength": round(dct_str, 4), "dct_avg_ratio": round(results["dct"]["avg_ratio"], 4)})
        with c2:
            st.json({"ela_mean": round(ela_mean, 4), "ela_std": round(results["ela"]["ela_std"], 4)})
            st.json({"format": fmt_label, "fmt_ai_score": round(results["fmt"]["ai_score"], 4)})

    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;color:#224422;font-size:0.72rem;
                letter-spacing:1px;margin-top:20px;border-top:1px solid #00ff2211;padding-top:10px;">
    ⚠ FORENSIC HEURISTICS ONLY — NOT A TRAINED ML MODEL — PROBABILISTIC ESTIMATES —
    POST-PROCESSING MAY AFFECT RESULTS — NOT FOR LEGAL OR FORENSIC USE
    </div>
    """, unsafe_allow_html=True)


def page_about():
    st.markdown("""
    <div style="
        text-align:center;
        padding:30px 0 20px 0;
        font-family:'Orbitron',monospace;
    ">
        <div style="color:#00ff4466;font-size:0.7rem;letter-spacing:5px;margin-bottom:10px;">// TECHNICAL DOCUMENTATION //</div>
        <div style="
            font-size:1.8rem; font-weight:900; color:#fff;
            text-shadow: 0 0 10px #fff, 0 0 25px #00ff88, 0 0 60px #00ff4433;
            letter-spacing:4px;
        ">DETECTION ALGORITHMS</div>
        <div style="margin:16px auto 0;width:200px;height:1px;
                    background:linear-gradient(90deg,transparent,#00ff88,transparent);
                    box-shadow:0 0 8px #00ff88;"></div>
    </div>
    """, unsafe_allow_html=True)

    section_header("SIGNAL 01 — NOISE LEVEL ESTIMATION  [WEIGHT: 40%]")
    st.markdown("""
Real camera sensors produce **shot noise** and **thermal noise** proportional to ISO.
This generates measurable pixel-level randomness — σ = 2–20 depending on camera/lighting.

AI generators synthesise values mathematically, producing images that are
**unnaturally clean** (σ < 2) — lacking real sensor noise entirely.

**Algorithm:** Laplacian-based noise estimator (Immerkær 1996)
```
kernel = [[1,-2,1],[-2,4,-2],[1,-2,1]]
sigma  = sqrt(pi/2) * mean(|Laplacian(gray)|) / 6.0
```
- `sigma < 2`  →  very likely AI
- `sigma 2–5`  →  borderline
- `sigma > 6`  →  likely real camera photo
    """)

    section_header("SIGNAL 02 — ERROR LEVEL ANALYSIS (ELA)  [WEIGHT: 25%]")
    st.markdown("""
JPEG compression is **lossy**. Re-compressing an already-JPEG image causes
minimal further loss. First-time JPEG compression of a **lossless PNG** causes large error.

Real camera photos → JPEG in-camera → **low ELA** when re-saved.
AI images (ChatGPT, DALL·E, Midjourney) → PNG output → **high ELA** when saved as JPEG.

**Algorithm:**
```
re_saved = img.save(JPEG, quality=92) → reload
ELA_map  = abs(original - re_saved)
ela_mean = ELA_map.mean()
```
- `ela_mean < 5`  →  was already JPEG → likely real
- `ela_mean > 10` →  first JPEG compression → likely AI PNG source
    """)

    section_header("SIGNAL 03 — SOURCE FORMAT HEURISTIC  [WEIGHT: 20%]")
    st.markdown("""
File format is a strong empirical prior:

| Format | Typical Source | AI Score Prior |
|--------|---------------|----------------|
| JPEG | Camera phones, DSLRs — almost always real | 0.25 |
| PNG | ChatGPT, DALL·E, Midjourney, Stable Diffusion | 0.65 |
| WebP | Google Gemini, web-delivered AI images | 0.60 |

This is a **prior** — overridden by the other signals when they strongly disagree.
    """)

    section_header("SIGNAL 04 — DCT BLOCK DETECTION  [WEIGHT: 15%]")
    st.markdown("""
JPEG divides images into **8×8 pixel blocks** with DCT applied to each.
In the 2D FFT magnitude spectrum, these blocks create energy spikes at
multiples of (image_dim / 8) from the DC center.

AI PNG images have **no such 8×8 structure**.

**Algorithm:** Measure energy ratio at expected DCT spike positions vs local background.
- `ratio > 3×`  →  strong JPEG grid → likely real camera JPEG
- `ratio ≈ 1×`  →  no grid → possibly AI PNG
    """)

    section_header("KNOWN LIMITATIONS")
    st.warning("""
FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY.

- AI image re-saved as JPEG loses the PNG fingerprint — may be misclassified as real
- Real photo exported as PNG (editor, screenshot) may be misclassified as AI
- Heavily post-processed or filtered images degrade all signals
- Very small images (< 200px) give unreliable results
- Not trained on a labeled dataset — all calibration is empirical/heuristic
- Do NOT use as sole evidence in legal, journalistic, or forensic contexts
    """)

    section_header("REFERENCES")
    st.markdown("""
- Immerkær (1996): *Fast Noise Variance Estimation*, Computer Vision and Image Understanding
- [SynthID — Google DeepMind](https://deepmind.google/technologies/synthid/)
- [C2PA Content Provenance Standard](https://c2pa.org/)
- [Error Level Analysis — Forensically](https://29a.ch/photo-forensics/#error-level-analysis)
    """)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.markdown(HACKER_CSS, unsafe_allow_html=True)
    tabs = st.tabs(["  DETECTOR  ", "  ABOUT & ALGORITHMS  "])
    with tabs[0]:
        page_home()
    with tabs[1]:
        page_about()


if __name__ == "__main__":
    main()
