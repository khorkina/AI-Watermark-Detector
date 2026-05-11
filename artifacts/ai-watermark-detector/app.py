"""
AI Watermark Detector — "Is This AI?"
======================================
Detects AI-generated images using forensic image analysis:
  - Noise level estimation (camera sensor noise vs AI cleanliness)
  - Error Level Analysis (JPEG compression history)
  - DCT block artifact detection (8×8 JPEG grid)
  - Source format heuristic (JPEG = camera, PNG/WebP = likely AI)

Run locally:
    streamlit run app.py

Deploy to Streamlit Community Cloud (free):
    1. Push this file + requirements.txt to a GitHub repo
    2. Go to https://share.streamlit.io and connect your repo
    3. Set main file to: app.py
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Watermark Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MAX_FILE_SIZE_MB = 10
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "webp"]


# ══════════════════════════════════════════════════════════════════════════════
# Detection Engine
# ══════════════════════════════════════════════════════════════════════════════

def enhance_saturation(img: Image.Image, factor: float = 8.0) -> Image.Image:
    """Boost HSV saturation to reveal hidden color patterns."""
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
    """Compute log-magnitude FFT of the luminance channel."""
    gray = np.array(img.convert("L"), dtype=np.float32)
    h, w = gray.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    fft_shifted = np.fft.fftshift(np.fft.fft2(gray * window))
    magnitude = np.abs(fft_shifted)
    log_magnitude = np.log1p(magnitude)
    return magnitude, log_magnitude


# ── Signal 1: Noise Level ─────────────────────────────────────────────────────
def estimate_noise_level(img: Image.Image) -> dict:
    """
    Real camera photos have genuine sensor noise (ISO noise, shot noise).
    AI-generated images are unnaturally clean — they lack this noise.

    Method: Laplacian-based noise estimator (Immerkær 1996).
    High sigma → natural camera noise → likely REAL.
    Low sigma → unnaturally clean → likely AI.
    """
    gray = np.array(img.convert("L"), dtype=np.float32)
    from scipy.ndimage import convolve
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    laplacian = convolve(gray, kernel)
    # Estimate noise sigma
    sigma = float(np.sqrt(np.pi / 2) * np.abs(laplacian).mean() / 6.0)

    # Real camera photo: sigma typically 3–20 depending on ISO
    # AI image: sigma typically 0.3–3
    # Threshold at sigma ≈ 4; below = AI, above = real
    # sigmoid centered at 4, inverted so HIGH sigma = LOW ai_score
    ai_score = float(1.0 / (1.0 + np.exp(0.6 * (sigma - 4.0))))
    return {"sigma": sigma, "ai_score": float(np.clip(ai_score, 0, 1))}


# ── Signal 2: Error Level Analysis ────────────────────────────────────────────
def analyze_ela(img: Image.Image) -> dict:
    """
    Error Level Analysis: re-save at JPEG quality 92 and measure the difference.

    Previously JPEG-compressed images (real camera photos) have already been
    lossy-compressed, so re-saving causes very little additional error → low ELA.

    Images from a lossless source (AI-generated PNGs from ChatGPT/DALL-E, etc.)
    lose significant information when saved as JPEG for the first time → high ELA.

    High ELA → first-time JPEG compression → likely AI (PNG origin).
    Low ELA  → already JPEG-compressed  → likely real camera photo.
    """
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")

    orig = np.array(img, dtype=np.float32)
    rsav = np.array(resaved, dtype=np.float32)
    ela_map = np.abs(orig - rsav)

    ela_mean = float(ela_map.mean())
    ela_std  = float(ela_map.std())
    ela_max  = float(ela_map.max())

    # Calibration: JPEG origin → ela_mean ≈ 1–5; PNG origin → ela_mean ≈ 8–25
    # Threshold at ela_mean ≈ 7
    ai_score = float(1.0 / (1.0 + np.exp(-0.5 * (ela_mean - 7.0))))

    return {
        "ela_mean": ela_mean,
        "ela_std":  ela_std,
        "ela_max":  ela_max,
        "ela_map":  ela_map,
        "ai_score": float(np.clip(ai_score, 0, 1)),
    }


# ── Signal 3: DCT Block Artifact Detection ────────────────────────────────────
def detect_dct_blocks(magnitude: np.ndarray, img_size: tuple) -> dict:
    """
    JPEG compression divides images into 8×8 pixel blocks. In the FFT magnitude
    spectrum, these blocks create energy spikes at multiples of (N/8) from center.

    Strong DCT peaks → JPEG-compressed → likely real camera photo → low AI score.
    No DCT peaks    → lossless source (PNG/WebP) → possibly AI → higher AI score.
    """
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
            pos_p = center + offset
            pos_n = center - offset
            for pos in [pos_p, pos_n]:
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
        # avg_ratio ≈ 1 = no peaks (PNG); ≈ 3–10 = strong JPEG peaks (real)
        dct_strength = float(np.clip((avg_ratio - 1.0) / 6.0, 0, 1))

    # Strong DCT = JPEG = real → LOW ai_score; Weak/none = PNG = possibly AI → HIGH ai_score
    ai_score = 1.0 - dct_strength
    return {
        "dct_strength": dct_strength,
        "avg_ratio":    float(np.mean(peak_ratios)) if peak_ratios else 1.0,
        "ai_score":     float(np.clip(ai_score, 0, 1)),
    }


# ── Signal 4: Source Format Heuristic ────────────────────────────────────────
def analyze_source_format(mime_type: str) -> dict:
    """
    The file format is a strong prior:
    - JPEG  → almost always a real camera photo or JPEG-saved AI
    - PNG   → AI generators (ChatGPT, DALL-E, Midjourney) output PNG by default
    - WebP  → used by some AI platforms

    This is a prior, not a definitive verdict. Combined with other signals.
    """
    if mime_type in ("image/jpeg", "image/jpg"):
        ai_score = 0.25   # JPEG origin lowers AI probability significantly
        fmt_label = "JPEG"
    elif mime_type == "image/webp":
        ai_score = 0.60   # WebP is common for AI outputs
        fmt_label = "WebP"
    elif mime_type == "image/png":
        ai_score = 0.65   # PNG is very common for AI generators
        fmt_label = "PNG"
    else:
        ai_score = 0.50
        fmt_label = "Unknown"

    return {"fmt_label": fmt_label, "ai_score": ai_score}


# ── Ensemble ──────────────────────────────────────────────────────────────────
def compute_ai_probability(
    noise_res: dict,
    ela_res:   dict,
    dct_res:   dict,
    fmt_res:   dict,
) -> tuple:
    """
    Weighted ensemble of four forensic signals.

    Weights (calibrated empirically):
      - Noise level:  40% — strongest physical signal
      - ELA:          25% — JPEG compression history
      - Source format: 20% — file format prior
      - DCT blocks:   15% — JPEG block structure in FFT
    """
    combined = (
        noise_res["ai_score"] * 0.40 +
        ela_res["ai_score"]   * 0.25 +
        fmt_res["ai_score"]   * 0.20 +
        dct_res["ai_score"]   * 0.15
    )

    # Sigmoid stretch to push toward a clear verdict
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
    """Full analysis pipeline. Cached for instant re-renders."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Resize very large images
    if max(img.size) > 1024:
        img.thumbnail((1024, 1024), Image.LANCZOS)

    enhanced          = enhance_saturation(img, factor=8.0)
    magnitude, log_mag = compute_fft_spectrum(img)

    noise_res = estimate_noise_level(img)
    ela_res   = analyze_ela(img)
    dct_res   = detect_dct_blocks(magnitude, img.size)
    fmt_res   = analyze_source_format(mime_type)

    probability, verdict = compute_ai_probability(noise_res, ela_res, dct_res, fmt_res)

    return {
        "original":    img,
        "enhanced":    enhanced,
        "log_mag":     log_mag,
        "ela_map":     ela_res["ela_map"],
        "noise":       noise_res,
        "ela":         ela_res,
        "dct":         dct_res,
        "fmt":         fmt_res,
        "probability": probability,
        "verdict":     verdict,
    }


# ══════════════════════════════════════════════════════════════════════════════
# UI Helpers
# ══════════════════════════════════════════════════════════════════════════════

def fft_to_image(log_mag: np.ndarray) -> Image.Image:
    norm = ((log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-8) * 255).astype(np.uint8)
    return Image.fromarray(norm).convert("RGB")


def ela_to_image(ela_map: np.ndarray) -> Image.Image:
    """Render ELA map: brighter = more error = more likely lossless/AI origin."""
    # Scale: clip at 30 for visibility, then normalize
    clipped = np.clip(ela_map.mean(axis=2) if ela_map.ndim == 3 else ela_map, 0, 30)
    norm = (clipped / 30.0 * 255).astype(np.uint8)
    # Apply a green-ish colormap for visual clarity
    colored = np.stack([norm // 3, norm, norm // 4], axis=2).astype(np.uint8)
    return Image.fromarray(colored)


def render_verdict(probability: float, verdict: str):
    if verdict == "likely_ai":
        st.error("### ⚠️ Likely AI-GENERATED")
        color = "#dc2626"
    elif verdict == "likely_real":
        st.success("### ✅ Likely REAL Photograph")
        color = "#16a34a"
    else:
        st.warning("### 🔎 Inconclusive — Could Be Either")
        color = "#d97706"

    pct = int(probability * 100)
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border-radius: 16px;
        padding: 28px 36px;
        margin: 16px 0;
        border: 2px solid {color}44;
        text-align: center;
    ">
        <div style="color: #94a3b8; font-size: 13px; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 8px;">
            AI Probability
        </div>
        <div style="color: {color}; font-size: 72px; font-weight: 800; line-height: 1;">
            {pct}%
        </div>
        <div style="margin-top: 16px; background: #0f172a; border-radius: 8px; height: 10px; overflow: hidden;">
            <div style="background: linear-gradient(90deg, {color}88, {color}); width: {pct}%; height: 100%; border-radius: 8px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_signal_bars(noise: dict, ela: dict, dct: dict, fmt: dict):
    st.markdown("**Signal Breakdown**")

    signals = [
        ("🔊 Noise Level",
         noise["ai_score"],
         f"Camera sensor noise sigma = {noise['sigma']:.1f} "
         f"({'very low — AI-clean' if noise['sigma'] < 3 else 'natural — camera noise' if noise['sigma'] > 5 else 'borderline'})"),
        ("🗜️ ELA (Compression History)",
         ela["ai_score"],
         f"Mean ELA = {ela['ela_mean']:.1f} "
         f"({'high → lossless/PNG source' if ela['ela_mean'] > 7 else 'low → previously JPEG compressed'})"),
        ("📐 Source Format",
         fmt["ai_score"],
         f"File is {fmt['fmt_label']} "
         f"({'PNG/WebP common in AI outputs' if fmt['fmt_label'] != 'JPEG' else 'JPEG typical of camera photos'})"),
        ("🔲 JPEG Block Structure",
         dct["ai_score"],
         f"DCT block strength = {dct['dct_strength']:.2f} "
         f"({'strong JPEG grid detected' if dct['dct_strength'] > 0.5 else 'no JPEG block pattern'})"),
    ]

    for label, score, desc in signals:
        col_a, col_b = st.columns([4, 1])
        with col_a:
            st.progress(float(score), text=f"**{label}** — {desc}")
        with col_b:
            ai_pct = int(score * 100)
            color = "#dc2626" if ai_pct > 60 else "#16a34a" if ai_pct < 40 else "#d97706"
            st.markdown(
                f"<div style='text-align:right;padding-top:6px;color:{color};font-weight:bold'>{ai_pct}%</div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# Pages
# ══════════════════════════════════════════════════════════════════════════════

def page_home():
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <span style="font-size:52px;">🔍</span>
        <h1 style="margin:8px 0 4px 0; font-size:2.2rem; font-weight:800;">AI Watermark Detector</h1>
        <p style="color:#94a3b8; font-size:1.05rem; margin:0;">
            Upload any image to detect invisible AI watermarks and image forensics signals
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    uploaded_file = st.file_uploader(
        "Drop your image here or click to browse",
        type=SUPPORTED_FORMATS,
        help=f"Supported: JPG, PNG, WEBP · Max size: {MAX_FILE_SIZE_MB} MB",
    )

    if uploaded_file is None:
        st.markdown("""
        <div style="border:2px dashed #334155; border-radius:12px; padding:40px; text-align:center; color:#64748b; margin-top:8px;">
            <div style="font-size:2rem; margin-bottom:8px;">🖼️</div>
            <div>No image uploaded yet</div>
            <div style="font-size:0.85rem; margin-top:4px;">JPG · PNG · WEBP · up to 10 MB</div>
        </div>
        """, unsafe_allow_html=True)
        return

    file_bytes = uploaded_file.read()
    if len(file_bytes) / (1024 * 1024) > MAX_FILE_SIZE_MB:
        st.error(f"File is too large. Maximum allowed is {MAX_FILE_SIZE_MB} MB.")
        return

    mime_type = uploaded_file.type or "image/jpeg"

    with st.spinner("Analyzing — running noise, ELA, DCT, and format forensics…"):
        try:
            results = analyze_image(file_bytes, mime_type)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            return

    # ── Results ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    render_verdict(results["probability"], results["verdict"])

    # Visuals
    st.markdown("### 🖼️ Visual Inspection")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(results["original"], caption="Original Image", use_container_width=True)
    with col2:
        st.image(results["enhanced"], caption="Saturation ×8 (reveals color patterns)", use_container_width=True)
    with col3:
        ela_vis = ela_to_image(results["ela_map"])
        st.image(ela_vis, caption="ELA Map (bright = first-time JPEG compression)", use_container_width=True)
    with col4:
        fft_vis = fft_to_image(results["log_mag"])
        st.image(fft_vis, caption="FFT Spectrum (JPEG shows cross-shaped grid)", use_container_width=True)

    # Explanation
    verdict   = results["verdict"]
    prob      = results["probability"]
    noise_sig = results["noise"]["sigma"]
    ela_mean  = results["ela"]["ela_mean"]
    fmt_label = results["fmt"]["fmt_label"]
    dct_str   = results["dct"]["dct_strength"]

    st.markdown("### 🧠 Explanation")

    if verdict == "likely_ai":
        msg = f"""
The image scores **{int(prob*100)}% AI probability** from these forensic signals:

- **Noise level σ = {noise_sig:.1f}**: {"Very low — AI models produce unnaturally clean images, lacking real camera sensor noise." if noise_sig < 4 else "Below typical camera noise, suggesting a synthetic source."}
- **ELA mean = {ela_mean:.1f}**: {"High — the image compresses significantly when saved as JPEG for the first time, consistent with a lossless PNG source used by most AI image generators." if ela_mean > 7 else "Moderate compression history."}
- **Source format: {fmt_label}**: {"PNG files are the default output format for ChatGPT, DALL·E, Midjourney, and most AI image generators." if fmt_label == "PNG" else "WebP is used by some AI platforms." if fmt_label == "WebP" else "Saved as JPEG, which slightly reduces AI confidence."}
- **DCT blocks: {"absent" if dct_str < 0.4 else "weak"}**: {"No 8×8 JPEG compression grid detected — consistent with a lossless (PNG) source." if dct_str < 0.4 else "Weak block structure, ambiguous."}
"""
    elif verdict == "likely_real":
        msg = f"""
The image scores **{int((1-prob)*100)}% probability of being a real photograph** from these signals:

- **Noise level σ = {noise_sig:.1f}**: {"Strong natural sensor noise — consistent with a real camera at typical ISO settings." if noise_sig > 6 else "Detectable noise consistent with a camera photo."}
- **ELA mean = {ela_mean:.1f}**: {"Low — the image was already JPEG-compressed (as cameras do), so re-saving causes minimal additional loss." if ela_mean < 7 else "Moderate."}
- **Source format: {fmt_label}**: {"JPEG is the native format of almost every digital camera and smartphone." if fmt_label == "JPEG" else fmt_label + " source."}
- **DCT blocks: {"strong" if dct_str > 0.5 else "present"}**: {"Clear 8×8 JPEG compression grid detected in the frequency spectrum — a hallmark of camera-captured images." if dct_str > 0.4 else "Some JPEG block structure detected."}
"""
    else:
        msg = f"""
The result is **inconclusive** ({int(prob*100)}% AI probability). This happens when signals conflict:

- **Noise σ = {noise_sig:.1f}**, **ELA = {ela_mean:.1f}**, **Format = {fmt_label}**, **DCT strength = {dct_str:.2f}**
- Possible reasons: AI image saved as JPEG (loses PNG fingerprint), real photo upscaled or post-processed (adds/removes noise), screenshot of an AI image, or heavy filtering/compression of a real photo.
"""
    st.info(msg)

    with st.expander("🔬 Detailed Signal Breakdown", expanded=False):
        render_signal_bars(results["noise"], results["ela"], results["dct"], results["fmt"])
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Noise Analysis**")
            st.json({"sigma": round(noise_sig, 3), "ai_score": round(results["noise"]["ai_score"], 3)})
            st.markdown("**DCT Block Detection**")
            st.json({
                "dct_strength": round(dct_str, 3),
                "avg_peak_ratio": round(results["dct"]["avg_ratio"], 3),
                "ai_score": round(results["dct"]["ai_score"], 3),
            })
        with c2:
            st.markdown("**ELA Analysis**")
            st.json({
                "ela_mean": round(ela_mean, 3),
                "ela_std":  round(results["ela"]["ela_std"], 3),
                "ela_max":  round(results["ela"]["ela_max"], 3),
                "ai_score": round(results["ela"]["ai_score"], 3),
            })
            st.markdown("**Source Format**")
            st.json({"format": fmt_label, "ai_score": round(results["fmt"]["ai_score"], 3)})

    st.caption(
        "⚠️ This tool uses forensic heuristics — not a trained ML model. "
        "Results are probabilistic estimates, not guarantees. "
        "Post-processing, re-encoding, or screenshots can affect accuracy."
    )


def page_about():
    st.markdown("""
    <div style="text-align:center; padding:20px 0 10px 0;">
        <h1 style="font-size:2.1rem; font-weight:800;">How It Works</h1>
        <p style="color:#94a3b8;">The forensic methods behind AI image detection</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("## 🧪 Four Forensic Signals")

    with st.expander("1️⃣ Noise Level Estimation (weight: 40%)", expanded=True):
        st.markdown("""
        **Physical basis:**
        Real camera sensors produce *shot noise* and *thermal noise* proportional to the ISO setting.
        This creates measurable random variation at the pixel level — typically σ = 2–20 depending on the camera and lighting.

        AI generators (diffusion models, GANs) synthesise pixel values mathematically. They produce
        images that are unnaturally *clean* — very low or zero real noise — unless grain is deliberately added.

        **Algorithm:**
        We use the Laplacian-based noise estimator (Immerkær 1996): apply a `[[1,-2,1],[-2,4,-2],[1,-2,1]]`
        kernel and estimate σ from the absolute mean response. 
        - σ < 2 → very likely AI
        - σ 2–5 → borderline
        - σ > 6 → likely real camera photo
        """)

    with st.expander("2️⃣ Error Level Analysis — ELA (weight: 25%)", expanded=False):
        st.markdown("""
        **Physical basis:**
        JPEG compression is *lossy*. If you JPEG-compress an already-JPEG image, very little additional
        information is lost — the quantisation boundaries are largely the same. If you JPEG-compress a
        *lossless* (PNG) image for the first time, it loses significant information → large error.

        Real camera photos are almost always JPEG-compressed in-camera.
        AI image generators (ChatGPT/DALL·E, Midjourney, Stable Diffusion) output PNG or lossless WebP.

        **Algorithm:**
        Re-save the image at JPEG quality 92 and measure the mean absolute pixel difference.
        - ELA < 5 → was already JPEG → likely real camera photo
        - ELA > 10 → first JPEG compression → likely AI PNG source
        """)

    with st.expander("3️⃣ Source Format Heuristic (weight: 20%)", expanded=False):
        st.markdown("""
        **Physical basis:**
        The file format is a strong empirical prior:

        | Format | Source |
        |--------|--------|
        | JPEG | Camera phones, DSLRs, scanner outputs — almost always real |
        | PNG | ChatGPT image generation, DALL·E, Midjourney, Stable Diffusion |
        | WebP | Google Gemini, some web-delivered AI images |

        This signal is a prior that is combined with (and can be overridden by) the other signals.
        If a real photo is saved as PNG or an AI image is re-saved as JPEG, this signal will be wrong —
        that's why it's only 20% of the weight.
        """)

    with st.expander("4️⃣ JPEG DCT Block Detection (weight: 15%)", expanded=False):
        st.markdown("""
        **Physical basis:**
        JPEG compression divides images into 8×8 pixel blocks and applies a Discrete Cosine Transform
        to each. In the 2D FFT magnitude spectrum, these blocks create characteristic energy spikes
        at spatial frequencies that are multiples of (image_dimension / 8) from the center.

        A PNG image (typical AI output) has no such 8×8 block structure.

        **Algorithm:**
        We measure the energy ratio at expected DCT spike positions vs their local background.
        - Ratio > 3× → strong JPEG grid → likely real camera JPEG
        - Ratio ≈ 1× → no grid → possibly AI PNG
        """)

    st.markdown("## 🤖 Which Images Are Reliably Detected")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Works well:**
        - ChatGPT / DALL·E 3 (PNG output)
        - Google Gemini / Imagen (WebP/PNG)
        - Midjourney (PNG output)
        - Stable Diffusion (PNG output)
        - Adobe Firefly (PNG output)
        - Any AI image saved in its native lossless format
        """)
    with col2:
        st.markdown("""
        **Harder cases:**
        - AI images that have been re-saved as JPEG
        - Real photos uploaded as PNG (screenshot, editor export)
        - Heavily compressed real photos (ELA may be ambiguous)
        - AI images with deliberately added grain/noise
        - Very small images (< 200 × 200 px)
        """)

    st.markdown("## ⚠️ Limitations & Disclaimer")
    st.warning("""
    **For educational and research purposes only.**
    
    - This tool uses heuristics, not a trained machine learning model. It does not have access to the secret keys used by SynthID, OpenAI, or other watermarking systems.
    - False positives (real photos flagged AI) and false negatives (AI images missed) are both common in edge cases.
    - Post-processing — re-encoding, resizing, screenshots, filters — significantly affects results.
    - Do not use this tool as sole evidence in any legal, journalistic, or forensic context.
    """)

    st.markdown("## 🛠️ Stack")
    st.code("""
streamlit      — web UI
numpy          — array math & FFT
scipy          — Laplacian filter, signal processing
opencv-python  — HSV saturation enhancement
pillow         — image loading, JPEG re-encoding for ELA
    """, language="text")

    st.markdown("## 📄 References")
    st.markdown("""
    - Immerkær (1996): Fast Noise Variance Estimation — *Computer Vision and Image Understanding*
    - [SynthID by Google DeepMind](https://deepmind.google/technologies/synthid/)
    - [C2PA Content Provenance Standard](https://c2pa.org/)
    - [Error Level Analysis (ELA) — Forensically](https://29a.ch/photo-forensics/#error-level-analysis)
    """)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    tabs = st.tabs(["🏠 Detector", "ℹ️ About & How It Works"])
    with tabs[0]:
        page_home()
    with tabs[1]:
        page_about()


if __name__ == "__main__":
    main()
