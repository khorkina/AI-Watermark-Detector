"""
AI Watermark Detector — "Is This AI?"
Hacker / cryptography aesthetic. Port 5000.
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
# HACKER CSS  — injected once via st.html()
# ══════════════════════════════════════════════════════════════════════════════
HACKER_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');

html, body, [class*="css"] {
    background-color: #000000 !important;
    color: #c8ffc8 !important;
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
}
body::before {
    content: "";
    position: fixed; top: 0; left: 0;
    width: 100%; height: 100%;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 2px,
        rgba(0,255,80,0.012) 2px, rgba(0,255,80,0.012) 4px
    );
    pointer-events: none; z-index: 9999;
}
.stApp, [data-testid="stAppViewContainer"] {
    background: #000000 !important;
}
[data-testid="stHeader"] { background: transparent !important; }
section[data-testid="stMain"] > div { background: #000 !important; }
[data-testid="stTabs"] button {
    color: #00cc55 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.9rem !important;
    letter-spacing: 2px !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
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
[data-testid="stFileUploader"] {
    border: 1px solid #00ff4433 !important;
    background: #000e04 !important;
    box-shadow: 0 0 20px #00ff441a, inset 0 0 30px #00ff440a !important;
}
[data-testid="stFileUploadDropzone"] {
    background: #000e04 !important;
    border: 2px dashed #00ff4455 !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #00ff88 !important;
    box-shadow: 0 0 25px #00ff4433 !important;
}
[data-testid="stProgressBar"] > div {
    background: #001a0a !important;
    border: 1px solid #00ff4422 !important;
}
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #005522, #00ff88) !important;
    box-shadow: 0 0 8px #00ff8888 !important;
}
[data-testid="stExpander"] {
    border: 1px solid #00ff2222 !important;
    background: #000a04 !important;
}
details summary { color: #00ff88 !important; font-family: 'Share Tech Mono', monospace !important; }
hr { border-color: #00ff2222 !important; box-shadow: 0 0 6px #00ff2211 !important; }
h1, h2, h3 {
    font-family: 'Orbitron', 'Share Tech Mono', monospace !important;
    text-shadow: 0 0 10px #00ff8877, 0 0 30px #00ff4433 !important;
    letter-spacing: 2px !important;
}
h1 { color: #ffffff !important; }
h2 { color: #00ff88 !important; }
h3 { color: #c0ffc0 !important; font-size: 1.05rem !important; }
p, li { color: #a0e8a0 !important; font-family: 'Share Tech Mono', monospace !important; }
strong { color: #00ff88 !important; }
code {
    background: #001a0a !important; color: #00ff88 !important;
    border: 1px solid #00ff2233 !important; border-radius: 2px !important;
    font-family: 'Share Tech Mono', monospace !important;
}
pre, .stCode {
    background: #000d04 !important;
    border: 1px solid #00ff2233 !important;
    border-left: 3px solid #00ff88 !important;
    box-shadow: 0 0 20px #00ff221a !important;
}
a { color: #00ff88 !important; }
a:hover { text-shadow: 0 0 8px #00ff88 !important; }
table { border-collapse: collapse !important; font-family: 'Share Tech Mono', monospace !important; }
th { background: #001a08 !important; color: #00ff88 !important; border: 1px solid #00ff2233 !important; padding: 6px 12px !important; }
td { border: 1px solid #00ff2222 !important; color: #88cc88 !important; padding: 5px 12px !important; }
tr:hover td { background: #001008 !important; }
figcaption { color: #447744 !important; font-size: 0.75rem !important; letter-spacing: 1px !important; }
[data-testid="stAlert"] { font-family: 'Share Tech Mono', monospace !important; }
[data-testid="stCaptionContainer"] { color: #336633 !important; font-family: 'Share Tech Mono', monospace !important; font-size: 0.73rem !important; }
</style>
"""

# ══════════════════════════════════════════════════════════════════════════════
# Detection Engine
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
        "ela_std": float(ela_map.std()),
        "ela_max": float(ela_map.max()),
        "ela_map": ela_map,
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
        "avg_ratio": float(np.mean(peak_ratios)) if peak_ratios else 1.0,
        "ai_score": float(np.clip(1.0 - dct_strength, 0, 1)),
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
    enhanced            = enhance_saturation(img, factor=8.0)
    magnitude, log_mag  = compute_fft_spectrum(img)
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
    ch = ela_map.mean(axis=2) if ela_map.ndim == 3 else ela_map
    clipped = np.clip(ch, 0, 30)
    norm = (clipped / 30.0 * 255).astype(np.uint8)
    colored = np.stack([norm // 3, norm, norm // 4], axis=2).astype(np.uint8)
    return Image.fromarray(colored)


# ══════════════════════════════════════════════════════════════════════════════
# UI Components — using st.html() for isolated HTML blocks
# ══════════════════════════════════════════════════════════════════════════════

def inject_css():
    st.html(HACKER_CSS)


def render_hero():
    st.html("""
<div style="text-align:center;padding:32px 0 20px 0;position:relative;">
  <div style="position:absolute;top:0;left:50%;transform:translateX(-50%);
              width:500px;height:100px;
              background:radial-gradient(ellipse,rgba(0,255,68,0.07) 0%,transparent 70%);
              pointer-events:none;"></div>
  <div style="font-family:'Orbitron','Share Tech Mono',monospace;
              font-size:0.7rem;letter-spacing:6px;color:rgba(0,255,68,0.4);
              text-transform:uppercase;margin-bottom:10px;">
    // FORENSIC IMAGE ANALYSIS SYSTEM v2.1 //
  </div>
  <div style="font-family:'Orbitron','Share Tech Mono',monospace;
              font-size:2.2rem;font-weight:900;color:#ffffff;
              text-shadow:0 0 7px #fff,0 0 15px #fff,0 0 30px #00ff88,0 0 60px rgba(0,255,136,0.4);
              letter-spacing:4px;line-height:1.2;margin-bottom:8px;">
    AI WATERMARK<br>DETECTOR
  </div>
  <div style="font-family:'Share Tech Mono','Courier New',monospace;
              font-size:0.82rem;color:#336633;letter-spacing:2px;margin-top:8px;">
    [ NOISE &nbsp;&bull;&nbsp; ELA &nbsp;&bull;&nbsp; DCT &nbsp;&bull;&nbsp; FORMAT FORENSICS ]
  </div>
  <div style="margin:18px auto 0 auto;width:260px;height:1px;
              background:linear-gradient(90deg,transparent,#00ff88,transparent);
              box-shadow:0 0 8px #00ff88;"></div>
</div>
""")


def render_terminal_boot():
    st.html("""
<div style="font-family:'Share Tech Mono','Courier New',monospace;
            color:#336633;font-size:0.76rem;letter-spacing:1px;
            padding:4px 0 14px 0;line-height:1.9;">
  &gt; INITIALIZING FORENSIC ENGINE... <span style="color:#00ff88;">OK</span><br>
  &gt; LOADING FFT MODULE............. <span style="color:#00ff88;">OK</span><br>
  &gt; ELA SUBSYSTEM READY............ <span style="color:#00ff88;">OK</span><br>
  &gt; DCT BLOCK DETECTOR............. <span style="color:#00ff88;">OK</span><br>
  &gt; AWAITING INPUT FILE<span style="color:#00ff88;">_</span>
</div>
""")


def render_section_header(text: str):
    st.html(f"""
<div style="font-family:'Orbitron','Share Tech Mono',monospace;
            font-size:0.78rem;letter-spacing:4px;color:#00ff88;
            text-shadow:0 0 8px rgba(0,255,136,0.5);
            text-transform:uppercase;padding:18px 0 8px 0;
            border-bottom:1px solid rgba(0,255,34,0.15);">
  // {text}
</div>
""")


def render_drop_zone():
    st.html("""
<div style="border:1px solid rgba(0,255,34,0.2);border-radius:2px;
            padding:48px 20px;text-align:center;
            background:radial-gradient(ellipse at center,#001a0a 0%,#000000 70%);
            box-shadow:0 0 30px rgba(0,255,26,0.04),inset 0 0 40px rgba(0,255,26,0.03);
            margin-top:10px;font-family:'Share Tech Mono',monospace;position:relative;">
  <div style="position:absolute;top:-1px;left:-1px;width:12px;height:12px;
              border-top:2px solid #00ff88;border-left:2px solid #00ff88;
              box-shadow:-2px -2px 8px #00ff88;"></div>
  <div style="position:absolute;top:-1px;right:-1px;width:12px;height:12px;
              border-top:2px solid #00ff88;border-right:2px solid #00ff88;
              box-shadow:2px -2px 8px #00ff88;"></div>
  <div style="position:absolute;bottom:-1px;left:-1px;width:12px;height:12px;
              border-bottom:2px solid #00ff88;border-left:2px solid #00ff88;
              box-shadow:-2px 2px 8px #00ff88;"></div>
  <div style="position:absolute;bottom:-1px;right:-1px;width:12px;height:12px;
              border-bottom:2px solid #00ff88;border-right:2px solid #00ff88;
              box-shadow:2px 2px 8px #00ff88;"></div>
  <div style="font-size:2.2rem;margin-bottom:12px;filter:drop-shadow(0 0 10px #00ff88);">&#128269;</div>
  <div style="color:rgba(0,255,68,0.5);font-size:0.9rem;letter-spacing:2px;">AWAITING TARGET IMAGE</div>
  <div style="color:#1a331a;font-size:0.73rem;margin-top:8px;letter-spacing:1px;">JPG &nbsp;&#xB7;&nbsp; PNG &nbsp;&#xB7;&nbsp; WEBP &nbsp;&#xB7;&nbsp; MAX 10 MB</div>
</div>
""")


def render_verdict(probability: float, verdict: str):
    pct = int(probability * 100)
    if verdict == "likely_ai":
        gc, gc2 = "#ff2244", "rgba(255,0,40,0.35)"
        label = "WARNING — SYNTHETIC ORIGIN DETECTED"
        sub   = "HIGH CONFIDENCE &nbsp;|&nbsp; AI-GENERATED"
    elif verdict == "likely_real":
        gc, gc2 = "#00ff88", "rgba(0,255,80,0.35)"
        label = "AUTHENTIC SIGNAL DETECTED"
        sub   = "HIGH CONFIDENCE &nbsp;|&nbsp; REAL PHOTOGRAPH"
    else:
        gc, gc2 = "#ffaa00", "rgba(255,160,0,0.35)"
        label = "SIGNAL AMBIGUOUS"
        sub   = "INCONCLUSIVE &nbsp;|&nbsp; FURTHER ANALYSIS REQUIRED"

    st.html(f"""
<div style="background:radial-gradient(ellipse at center top,rgba({
    '255,34,68' if verdict=='likely_ai' else '0,255,136' if verdict=='likely_real' else '255,170,0'
},0.04) 0%,#000000 55%);
            border:1px solid {gc}55;border-radius:2px;
            padding:32px 36px 28px 36px;margin:16px 0;
            text-align:center;position:relative;
            box-shadow:0 0 40px {gc}18,inset 0 0 60px {gc}06;
            font-family:'Share Tech Mono','Courier New',monospace;">
  <div style="position:absolute;top:-1px;left:-1px;width:18px;height:18px;border-top:2px solid {gc};border-left:2px solid {gc};box-shadow:-2px -2px 10px {gc};"></div>
  <div style="position:absolute;top:-1px;right:-1px;width:18px;height:18px;border-top:2px solid {gc};border-right:2px solid {gc};box-shadow:2px -2px 10px {gc};"></div>
  <div style="position:absolute;bottom:-1px;left:-1px;width:18px;height:18px;border-bottom:2px solid {gc};border-left:2px solid {gc};box-shadow:-2px 2px 10px {gc};"></div>
  <div style="position:absolute;bottom:-1px;right:-1px;width:18px;height:18px;border-bottom:2px solid {gc};border-right:2px solid {gc};box-shadow:2px 2px 10px {gc};"></div>
  <div style="font-size:0.72rem;letter-spacing:4px;color:{gc};
              text-shadow:0 0 8px {gc};margin-bottom:14px;">{label}</div>
  <div style="font-family:'Orbitron','Share Tech Mono',monospace;
              font-size:5rem;font-weight:900;color:#ffffff;
              text-shadow:0 0 7px #fff,0 0 20px {gc},{gc2 if 'ai' not in verdict else '0 0 50px ' + gc2};
              line-height:1;margin-bottom:8px;">
    {pct}<span style="font-size:2.2rem;opacity:0.6;">%</span>
  </div>
  <div style="font-size:0.68rem;letter-spacing:3px;color:{gc}88;margin-bottom:20px;">
    AI PROBABILITY &nbsp;&#xB7;&nbsp; {sub}
  </div>
  <div style="background:#001008;border:1px solid {gc}28;height:7px;overflow:hidden;border-radius:1px;">
    <div style="background:linear-gradient(90deg,{gc}44,{gc});
                width:{pct}%;height:100%;box-shadow:0 0 10px {gc};"></div>
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
    <span style="color:#a0e8a0;font-size:0.8rem;letter-spacing:1px;">{symbol} {label}</span>
    <span style="color:{tc};font-family:'Orbitron',monospace;font-size:1rem;
                 font-weight:700;text-shadow:0 0 8px {bc};">{pct}%</span>
  </div>
  <div style="background:#001008;height:4px;border-radius:1px;overflow:hidden;margin-bottom:6px;">
    <div style="background:linear-gradient(90deg,{bc}66,{bc});
                width:{pct}%;height:100%;box-shadow:0 0 8px {bc};"></div>
  </div>
  <div style="color:#446644;font-size:0.71rem;letter-spacing:0.5px;">{detail}</div>
</div>
""")


def render_log_block(lines: list, accent_color: str):
    rows = ""
    for i, line in enumerate(lines):
        color = accent_color if i == len(lines) - 1 else "#336633"
        rows += f'<div style="color:{color};">{line}</div>'
    st.html(f"""
<div style="background:#000a04;border:1px solid #00ff2218;
            border-left:3px solid {accent_color};
            padding:14px 18px;font-family:'Share Tech Mono','Courier New',monospace;
            font-size:0.76rem;line-height:1.85;margin-top:14px;
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
        render_drop_zone()
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
    render_section_header("VERDICT")
    render_verdict(results["probability"], results["verdict"])

    # ── Visuals ───────────────────────────────────────────────────────────────
    render_section_header("VISUAL FORENSICS")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(results["original"],            caption="[ ORIGINAL ]",      use_container_width=True)
    with col2:
        st.image(results["enhanced"],            caption="[ SATURATION x8 ]", use_container_width=True)
    with col3:
        st.image(ela_to_image(results["ela_map"]), caption="[ ELA MAP ]",       use_container_width=True)
    with col4:
        st.image(fft_to_image(results["log_mag"]), caption="[ FFT SPECTRUM ]",   use_container_width=True)

    # ── Signals ───────────────────────────────────────────────────────────────
    render_section_header("SIGNAL ANALYSIS")
    noise_sig = results["noise"]["sigma"]
    ela_mean  = results["ela"]["ela_mean"]
    dct_str   = results["dct"]["dct_strength"]
    fmt_label = results["fmt"]["fmt_label"]

    render_signal_card(
        "NOISE LEVEL", results["noise"]["ai_score"],
        f"Laplacian sigma = {noise_sig:.2f}  |  " +
        ("UNNATURALLY CLEAN — AI SIGNATURE" if noise_sig < 3
         else "NATURAL SENSOR NOISE — CAMERA" if noise_sig > 6
         else "BORDERLINE — AMBIGUOUS"),
        "[!]",
    )
    render_signal_card(
        "ERROR LEVEL ANALYSIS", results["ela"]["ai_score"],
        f"Mean ELA = {ela_mean:.2f}  |  " +
        ("HIGH — FIRST JPEG COMPRESSION — PNG/AI SOURCE" if ela_mean > 7
         else "LOW — PREVIOUSLY JPEG COMPRESSED — CAMERA"),
        "[~]",
    )
    render_signal_card(
        "SOURCE FORMAT", results["fmt"]["ai_score"],
        f"Detected: {fmt_label}  |  " +
        ("PNG IS DEFAULT OUTPUT FOR MOST AI GENERATORS" if fmt_label == "PNG"
         else "WEBP USED BY GEMINI AND SOME AI PLATFORMS" if fmt_label == "WebP"
         else "JPEG IS NATIVE CAMERA/SMARTPHONE FORMAT"),
        "[F]",
    )
    render_signal_card(
        "DCT BLOCK STRUCTURE", results["dct"]["ai_score"],
        f"Block strength = {dct_str:.3f}  |  " +
        ("STRONG 8x8 JPEG GRID — CAMERA JPEG" if dct_str > 0.5
         else "NO JPEG BLOCK PATTERN — LOSSLESS/AI SOURCE"),
        "[D]",
    )

    # ── Log ───────────────────────────────────────────────────────────────────
    verdict  = results["verdict"]
    prob     = results["probability"]
    if verdict == "likely_ai":
        color = "#ff2244"
        status = "SYNTHETIC"
        conclusion = "HIGH PROBABILITY OF SYNTHETIC ORIGIN — AI WATERMARK DETECTED"
    elif verdict == "likely_real":
        color = "#00ff88"
        status = "AUTHENTIC"
        conclusion = "SIGNALS CONSISTENT WITH REAL CAMERA PHOTOGRAPH"
    else:
        color = "#ffaa00"
        status = "INCONCLUSIVE"
        conclusion = "SIGNALS CONFLICT — POSSIBLE POST-PROCESSING OR RE-ENCODING"

    render_log_block([
        f"[RESULT]  AI_PROBABILITY={int(prob*100)}%  STATUS={status}",
        f"[NOISE]   sigma={noise_sig:.2f}  threshold=4.0",
        f"[ELA]     mean={ela_mean:.2f}  threshold=7.0",
        f"[FORMAT]  type={fmt_label}",
        f"[DCT]     strength={dct_str:.3f}",
        f"[CONCLUSION]  {conclusion}",
    ], color)

    # ── Raw data ──────────────────────────────────────────────────────────────
    with st.expander("[ RAW FORENSIC DATA ]"):
        c1, c2 = st.columns(2)
        with c1:
            st.json({"noise_sigma": round(noise_sig, 4), "noise_ai_score": round(results["noise"]["ai_score"], 4)})
            st.json({"dct_strength": round(dct_str, 4), "avg_ratio": round(results["dct"]["avg_ratio"], 4)})
        with c2:
            st.json({"ela_mean": round(ela_mean, 4), "ela_std": round(results["ela"]["ela_std"], 4)})
            st.json({"format": fmt_label, "fmt_ai_score": round(results["fmt"]["ai_score"], 4)})

    st.caption(
        "FORENSIC HEURISTICS ONLY — NOT A TRAINED ML MODEL — "
        "PROBABILISTIC ESTIMATES — POST-PROCESSING MAY AFFECT RESULTS"
    )


def page_about():
    st.html("""
<div style="text-align:center;padding:28px 0 18px 0;">
  <div style="font-family:'Orbitron',monospace;font-size:0.68rem;
              letter-spacing:5px;color:rgba(0,255,68,0.35);margin-bottom:10px;">
    // TECHNICAL DOCUMENTATION //
  </div>
  <div style="font-family:'Orbitron',monospace;font-size:1.7rem;font-weight:900;
              color:#ffffff;letter-spacing:4px;
              text-shadow:0 0 10px #fff,0 0 25px #00ff88,0 0 60px rgba(0,255,68,0.25);">
    DETECTION ALGORITHMS
  </div>
  <div style="margin:14px auto 0;width:180px;height:1px;
              background:linear-gradient(90deg,transparent,#00ff88,transparent);
              box-shadow:0 0 8px #00ff88;"></div>
</div>
""")

    render_section_header("SIGNAL 01 — NOISE LEVEL ESTIMATION  [WEIGHT: 40%]")
    st.markdown("""
Real camera sensors produce **shot noise** and **thermal noise** proportional to ISO.
This creates measurable pixel-level randomness — sigma = 2–20 depending on camera/lighting.

AI generators synthesise values mathematically — images are **unnaturally clean** (sigma < 2).

**Algorithm** (Laplacian noise estimator, Immerkær 1996):
```
kernel = [[1,-2,1],[-2,4,-2],[1,-2,1]]
sigma  = sqrt(pi/2) * mean(|Laplacian(gray)|) / 6.0
```
- `sigma < 2`  — very likely AI
- `sigma 2–5`  — borderline
- `sigma > 6`  — likely real camera photo
""")

    render_section_header("SIGNAL 02 — ERROR LEVEL ANALYSIS (ELA)  [WEIGHT: 25%]")
    st.markdown("""
JPEG compression is **lossy**. Re-compressing an already-JPEG image causes minimal further loss.
First-time JPEG compression of a **lossless PNG** causes large error.

Real camera photos → JPEG in-camera → **low ELA**. AI images → PNG output → **high ELA**.

```
re_saved = save(img, JPEG, quality=92) → reload
ELA_map  = abs(original - re_saved)
ela_mean = ELA_map.mean()
```
- `ela_mean < 5`  — was already JPEG → likely real
- `ela_mean > 10` — first JPEG compression → likely AI PNG
""")

    render_section_header("SIGNAL 03 — SOURCE FORMAT  [WEIGHT: 20%]")
    st.markdown("""
| Format | Typical Source | AI Score Prior |
|--------|---------------|----------------|
| JPEG | Camera phones, DSLRs — almost always real | 0.25 |
| PNG | ChatGPT, DALL·E, Midjourney, Stable Diffusion | 0.65 |
| WebP | Google Gemini, web-delivered AI images | 0.60 |
""")

    render_section_header("SIGNAL 04 — DCT BLOCK DETECTION  [WEIGHT: 15%]")
    st.markdown("""
JPEG divides images into **8×8 pixel blocks** with DCT applied to each.
In the 2D FFT magnitude spectrum these appear as energy spikes at multiples of `(N/8)` from center.

AI PNG images have **no such 8×8 structure**.

- `peak ratio > 3x`  — strong JPEG grid → likely camera JPEG
- `peak ratio ~ 1x`  — no grid → possibly AI PNG
""")

    render_section_header("KNOWN LIMITATIONS")
    st.warning("""
FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY.

- AI image re-saved as JPEG loses the PNG fingerprint and may be misclassified as real
- Real photo exported as PNG may be misclassified as AI
- Heavily post-processed or filtered images degrade all signals
- Not a trained ML model — calibration is empirical/heuristic
- Do NOT use as sole evidence in any legal, journalistic, or forensic context
""")

    render_section_header("REFERENCES")
    st.markdown("""
- Immerkær (1996): *Fast Noise Variance Estimation* — Computer Vision and Image Understanding
- [SynthID — Google DeepMind](https://deepmind.google/technologies/synthid/)
- [C2PA Content Provenance Standard](https://c2pa.org/)
- [Error Level Analysis — Forensically](https://29a.ch/photo-forensics/#error-level-analysis)
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
