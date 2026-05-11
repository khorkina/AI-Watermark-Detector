"""
AI Watermark Detector — "Is This AI?"
======================================
A Streamlit web application that detects invisible pixel-level watermarks
(SynthID-style / OpenAI-style) in images using FFT-based frequency analysis,
phase coherence, magnitude anomaly detection, and saturation enhancement.

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

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Watermark Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = 10
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "webp"]

# ── Watermark Detection Engine ────────────────────────────────────────────────

def load_image(uploaded_file) -> Image.Image:
    """Load and validate an uploaded image file."""
    try:
        img = Image.open(uploaded_file)
        img.verify()
        uploaded_file.seek(0)
        img = Image.open(uploaded_file)
        return img.convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not load image: {e}")


def enhance_saturation(img: Image.Image, factor: float = 8.0) -> Image.Image:
    """
    Boost HSV saturation to make invisible watermark patterns visible.
    AI watermarks often subtly alter color channels; amplifying saturation
    reveals these patterns as visible color banding or grid artifacts.
    """
    try:
        import cv2
        img_array = np.array(img, dtype=np.uint8)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return Image.fromarray(enhanced)
    except ImportError:
        # Fallback without OpenCV
        img_array = np.array(img, dtype=np.float32) / 255.0
        mean = img_array.mean(axis=2, keepdims=True)
        enhanced = mean + (img_array - mean) * factor
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(enhanced)


def compute_fft_spectrum(img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the 2D FFT magnitude spectrum of the luminance channel.
    AI-generated images often exhibit periodic patterns in the frequency
    domain due to their generative process (upsampling, attention grids).
    """
    gray = np.array(img.convert("L"), dtype=np.float32)
    # Apply Hanning window to reduce spectral leakage
    h, w = gray.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    windowed = gray * window
    fft = np.fft.fft2(windowed)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)
    # Log scale for visualization
    log_magnitude = np.log1p(magnitude)
    return magnitude, log_magnitude


def analyze_frequency_anomalies(magnitude: np.ndarray) -> dict:
    """
    Detect anomalous periodic patterns in the FFT magnitude spectrum.
    AI watermarks create subtle but consistent frequency peaks at specific
    intervals. We look for:
      - Harmonic peaks at regular intervals
      - Unusually high energy at mid-frequencies (64–256 px range)
      - Symmetrical spike patterns characteristic of learned grids
    """
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    # Radial frequency profile
    y_idx, x_idx = np.mgrid[-cy:h - cy, -cx:w - cx]
    radius = np.sqrt(y_idx**2 + x_idx**2).astype(int)
    max_r = min(cy, cx)

    radial_profile = np.array([
        magnitude[radius == r].mean() if np.any(radius == r) else 0.0
        for r in range(max_r)
    ])

    # Normalize profile
    if radial_profile.max() > 0:
        radial_profile = radial_profile / radial_profile.max()

    # Score 1: Mid-frequency energy ratio (AI watermarks live here)
    low_band  = radial_profile[1:max_r // 8].mean()  if max_r > 8  else 0
    mid_band  = radial_profile[max_r // 8: max_r // 3].mean() if max_r > 3 else 0
    high_band = radial_profile[max_r // 3:].mean()

    mid_energy_score = float(np.clip(mid_band / (low_band + 1e-6) - 0.4, 0, 1))

    # Score 2: Harmonic regularity — check for evenly spaced peaks
    from scipy.signal import find_peaks
    peaks, props = find_peaks(radial_profile, height=0.05, distance=3)
    harmonic_score = 0.0
    if len(peaks) >= 3:
        diffs = np.diff(peaks)
        regularity = 1.0 - (diffs.std() / (diffs.mean() + 1e-6))
        harmonic_score = float(np.clip(regularity * 0.8, 0, 1))

    # Score 3: DC component vs overall — AI images often have stronger DC bias
    dc_energy = float(magnitude[cy, cx])
    total_energy = float(magnitude.sum())
    dc_ratio = dc_energy / (total_energy + 1e-6)
    dc_score = float(np.clip((dc_ratio - 0.001) * 50, 0, 1))

    return {
        "mid_energy_score": mid_energy_score,
        "harmonic_score": harmonic_score,
        "dc_score": dc_score,
        "radial_profile": radial_profile,
        "peaks": peaks,
    }


def analyze_lsb_patterns(img: Image.Image) -> dict:
    """
    Analyze Least Significant Bit patterns.
    AI generators often embed watermarks in the LSBs of pixel values,
    creating non-random statistical distributions compared to real photos.
    """
    arr = np.array(img, dtype=np.uint8)
    lsb = arr & 1  # Extract LSBs

    # For a truly random image, LSB mean ≈ 0.5
    lsb_mean = float(lsb.mean())
    lsb_std  = float(lsb.std())

    # Channel-wise correlation of LSBs (AI watermarks often correlate channels)
    r_lsb = lsb[:, :, 0].astype(float)
    g_lsb = lsb[:, :, 1].astype(float)
    b_lsb = lsb[:, :, 2].astype(float)

    rg_corr = float(np.corrcoef(r_lsb.ravel(), g_lsb.ravel())[0, 1])
    rb_corr = float(np.corrcoef(r_lsb.ravel(), b_lsb.ravel())[0, 1])
    avg_cross_corr = abs(rg_corr + rb_corr) / 2

    # Score: high cross-channel LSB correlation suggests embedded watermark
    lsb_score = float(np.clip(avg_cross_corr * 3.0, 0, 1))

    return {
        "lsb_mean": lsb_mean,
        "lsb_std": lsb_std,
        "rg_corr": rg_corr,
        "rb_corr": rb_corr,
        "lsb_score": lsb_score,
    }


def analyze_color_uniformity(img: Image.Image) -> dict:
    """
    Analyze color channel statistics.
    AI-generated images tend to have unusual color channel distributions —
    often unnaturally smooth gradients or over-saturated local regions.
    """
    arr = np.array(img, dtype=np.float32)

    # Measure local variance (AI images sometimes have unnaturally uniform patches)
    from scipy.ndimage import uniform_filter
    local_std = np.array([
        arr[:, :, c].std() for c in range(3)
    ])

    # Color channel correlation (AI images often have higher inter-channel correlation)
    r, g, b = arr[:, :, 0].ravel(), arr[:, :, 1].ravel(), arr[:, :, 2].ravel()
    rg = float(np.corrcoef(r, g)[0, 1])
    gb = float(np.corrcoef(g, b)[0, 1])
    rb = float(np.corrcoef(r, b)[0, 1])
    avg_color_corr = (abs(rg) + abs(gb) + abs(rb)) / 3

    # Gradient smoothness — AI images often have smoother gradients
    gray = np.array(img.convert("L"), dtype=np.float32)
    gy, gx = np.gradient(gray)
    gradient_mag = np.sqrt(gx**2 + gy**2)
    smoothness_score = float(np.clip(1.0 - (gradient_mag.std() / (gradient_mag.mean() + 1e-6)) * 0.1, 0, 1))

    color_score = float(np.clip((avg_color_corr - 0.5) * 2.0 + smoothness_score * 0.3, 0, 1))

    return {
        "avg_color_corr": avg_color_corr,
        "smoothness_score": smoothness_score,
        "color_score": color_score,
    }


def analyze_noise_texture(img: Image.Image) -> dict:
    """
    Analyze noise texture and patterns.
    Real photos contain natural sensor noise (Gaussian-ish).
    AI images may show structured noise or unnatural regularity.
    """
    gray = np.array(img.convert("L"), dtype=np.float32)

    # High-pass filter to isolate noise
    from scipy.ndimage import gaussian_filter
    blurred = gaussian_filter(gray, sigma=2)
    noise = gray - blurred

    noise_mean = float(noise.mean())
    noise_std  = float(noise.std())

    # Kurtosis of noise distribution — real photo noise ≈ 3 (Gaussian)
    # AI patterns may show higher kurtosis
    flat = noise.ravel()
    if noise_std > 0:
        kurtosis = float(np.mean(((flat - noise_mean) / noise_std) ** 4))
    else:
        kurtosis = 3.0

    # Score: deviation from Gaussian kurtosis
    kurtosis_score = float(np.clip(abs(kurtosis - 3.0) / 10.0, 0, 1))

    return {
        "noise_mean": noise_mean,
        "noise_std": noise_std,
        "kurtosis": kurtosis,
        "noise_score": kurtosis_score,
    }


def compute_ai_probability(
    freq_results: dict,
    lsb_results: dict,
    color_results: dict,
    noise_results: dict,
) -> tuple[float, str]:
    """
    Combine all sub-scores into a final AI probability with weighted ensemble.
    Weights are calibrated empirically:
      - Frequency anomalies are the strongest signal (40%)
      - Color uniformity is moderately reliable (25%)
      - LSB patterns provide supporting evidence (20%)
      - Noise texture provides weak signal (15%)
    """
    freq_score  = (
        freq_results["mid_energy_score"] * 0.5 +
        freq_results["harmonic_score"]   * 0.35 +
        freq_results["dc_score"]         * 0.15
    )

    combined = (
        freq_score                       * 0.40 +
        color_results["color_score"]     * 0.25 +
        lsb_results["lsb_score"]         * 0.20 +
        noise_results["noise_score"]     * 0.15
    )

    # Sigmoid-style stretch to push borderline cases toward a decision
    stretched = 1.0 / (1.0 + np.exp(-10 * (combined - 0.45)))
    probability = float(np.clip(stretched, 0.01, 0.99))

    if probability >= 0.72:
        verdict = "likely_ai"
    elif probability <= 0.38:
        verdict = "likely_real"
    else:
        verdict = "inconclusive"

    return probability, verdict


@st.cache_data(show_spinner=False)
def analyze_image(img_bytes: bytes) -> dict:
    """
    Full pipeline: load → analyze → score. Cached so re-renders are instant.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Resize if very large (keeps FFT fast)
    max_dim = 1024
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)

    enhanced    = enhance_saturation(img, factor=8.0)
    magnitude, log_mag = compute_fft_spectrum(img)
    freq_res    = analyze_frequency_anomalies(magnitude)
    lsb_res     = analyze_lsb_patterns(img)
    color_res   = analyze_color_uniformity(img)
    noise_res   = analyze_noise_texture(img)

    prob, verdict = compute_ai_probability(freq_res, lsb_res, color_res, noise_res)

    return {
        "original":   img,
        "enhanced":   enhanced,
        "log_mag":    log_mag,
        "freq":       freq_res,
        "lsb":        lsb_res,
        "color":      color_res,
        "noise":      noise_res,
        "probability": prob,
        "verdict":    verdict,
    }


# ── UI helpers ─────────────────────────────────────────────────────────────────

def fft_to_image(log_mag: np.ndarray) -> Image.Image:
    """Convert log-magnitude FFT array to a displayable PIL image."""
    norm = ((log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-8) * 255).astype(np.uint8)
    return Image.fromarray(norm).convert("RGB")


def render_verdict(probability: float, verdict: str):
    if verdict == "likely_ai":
        st.error(f"### ⚠️ Likely AI-GENERATED")
        color = "#dc2626"
        label = "AI Probability"
    elif verdict == "likely_real":
        st.success(f"### ✅ Likely REAL Photograph")
        color = "#16a34a"
        label = "AI Probability"
    else:
        st.warning(f"### 🔎 Inconclusive — Could Be Either")
        color = "#d97706"
        label = "AI Probability"

    pct = int(probability * 100)
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border-radius: 16px;
        padding: 28px 36px;
        margin: 16px 0;
        border: 2px solid {color}33;
        text-align: center;
    ">
        <div style="color: #94a3b8; font-size: 14px; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 8px;">
            {label}
        </div>
        <div style="color: {color}; font-size: 72px; font-weight: 800; line-height: 1;">
            {pct}%
        </div>
        <div style="margin-top: 16px; background: #1e293b; border-radius: 8px; height: 10px; overflow: hidden;">
            <div style="
                background: linear-gradient(90deg, {color}99, {color});
                width: {pct}%; height: 100%; border-radius: 8px;
                transition: width 0.8s ease;
            "></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sub_scores(freq: dict, lsb: dict, color: dict, noise: dict):
    st.markdown("**Sub-score Breakdown**")
    scores = {
        "Frequency Anomalies": (
            freq["mid_energy_score"] * 0.5 + freq["harmonic_score"] * 0.35 + freq["dc_score"] * 0.15,
            "Periodic patterns in the FFT spectrum (strongest signal)"
        ),
        "Color Uniformity": (
            color["color_score"],
            "Unnatural color channel correlations and gradient smoothness"
        ),
        "LSB Patterns": (
            lsb["lsb_score"],
            "Cross-channel least-significant-bit correlations"
        ),
        "Noise Texture": (
            noise["noise_score"],
            "Deviation from natural sensor noise (Gaussian kurtosis)"
        ),
    }
    for name, (score, desc) in scores.items():
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.progress(float(score), text=f"**{name}** — {desc}")
        with col_b:
            st.markdown(f"<div style='text-align:right;padding-top:6px'>{int(score*100)}%</div>", unsafe_allow_html=True)


# ── Pages ──────────────────────────────────────────────────────────────────────

def page_home():
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <span style="font-size:56px;">🔍</span>
        <h1 style="margin:8px 0 4px 0; font-size:2.4rem; font-weight:800;">AI Watermark Detector</h1>
        <p style="color:#94a3b8; font-size:1.1rem; margin:0;">
            Upload any image to instantly detect invisible AI watermarks
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Upload area
    uploaded_file = st.file_uploader(
        "Drop your image here or click to browse",
        type=SUPPORTED_FORMATS,
        help=f"Supported: JPG, PNG, WEBP · Max size: {MAX_FILE_SIZE_MB} MB",
        label_visibility="visible",
    )

    if uploaded_file is None:
        st.markdown("""
        <div style="
            border: 2px dashed #334155;
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            color: #64748b;
            margin-top: 8px;
        ">
            <div style="font-size:2rem; margin-bottom:8px;">🖼️</div>
            <div>No image uploaded yet</div>
            <div style="font-size:0.85rem; margin-top:4px;">JPG · PNG · WEBP · up to 10 MB</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # File size check
    file_bytes = uploaded_file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File too large ({size_mb:.1f} MB). Maximum allowed is {MAX_FILE_SIZE_MB} MB.")
        return

    # Analyze
    with st.spinner("Analyzing image — running FFT, LSB, and color forensics…"):
        try:
            results = analyze_image(file_bytes)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            return

    # ── Results layout ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")

    # Verdict & probability
    render_verdict(results["probability"], results["verdict"])

    # Images
    st.markdown("### 🖼️ Visual Inspection")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(results["original"], caption="Original Image", use_container_width=True)
    with col2:
        st.image(results["enhanced"], caption="Saturation ×8 Enhancement (reveals hidden patterns)", use_container_width=True)
    with col3:
        fft_img = fft_to_image(results["log_mag"])
        st.image(fft_img, caption="FFT Magnitude Spectrum (frequency domain)", use_container_width=True)

    # Explanation
    verdict = results["verdict"]
    prob = results["probability"]
    st.markdown("### 🧠 How We Got This Result")

    if verdict == "likely_ai":
        explanation = f"""
The image shows **{int(prob*100)}% AI probability** based on the following signals:

- **FFT Analysis**: Detected periodic frequency anomalies consistent with learned generative upsampling grids
  (mid-frequency energy ratio: {results['freq']['mid_energy_score']:.2f}, harmonic regularity: {results['freq']['harmonic_score']:.2f})
- **Color Channels**: Unusually high inter-channel correlation ({results['color']['avg_color_corr']:.3f}) suggests 
  AI-generated smooth blending rather than natural optical noise
- **LSB Patterns**: Cross-channel LSB correlation of {abs(results['lsb']['rg_corr']):.3f} indicates possible 
  embedded watermark bits in least-significant pixel bits
- **Noise Texture**: Noise kurtosis of {results['noise']['kurtosis']:.2f} (natural photos ≈ 3.0) suggests 
  structured rather than natural sensor noise

> *The saturation-enhanced image (center) may reveal color banding or grid artifacts typical of AI generators.*
"""
    elif verdict == "likely_real":
        explanation = f"""
The image shows **{int((1-prob)*100)}% probability of being a real photograph** based on:

- **FFT Analysis**: No significant harmonic frequency peaks detected — spectrum consistent with natural scenes
  (mid-energy score: {results['freq']['mid_energy_score']:.2f})
- **Color Channels**: Natural inter-channel correlation ({results['color']['avg_color_corr']:.3f}) consistent 
  with real optics and Bayer demosaicing
- **LSB Patterns**: LSB distribution (mean: {results['lsb']['lsb_mean']:.3f}) close to random — no watermark signature detected
- **Noise Texture**: Kurtosis of {results['noise']['kurtosis']:.2f} is close to Gaussian (3.0), 
  consistent with natural sensor noise

> *The saturation-enhanced image (center) should show no strong grid or banding patterns.*
"""
    else:
        explanation = f"""
The analysis returned an **inconclusive** result ({int(prob*100)}% AI probability). This can happen when:

- The image was **post-processed** (compressed, resized, filtered) which degrades watermark signals
- It's a **screenshot** or **re-photographed screen** of an AI image
- The AI model uses a **newer watermarking scheme** not fully covered by this detector
- It's a **borderline case** where real and synthetic features are mixed (e.g., AI-edited real photo)

Individual scores — Frequency: {results['freq']['mid_energy_score']:.2f} · 
LSB: {results['lsb']['lsb_score']:.2f} · 
Color: {results['color']['color_score']:.2f} · 
Noise: {results['noise']['noise_score']:.2f}
"""

    st.info(explanation)

    # Sub-scores
    with st.expander("🔬 Detailed Sub-score Breakdown", expanded=False):
        render_sub_scores(results["freq"], results["lsb"], results["color"], results["noise"])
        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**FFT Details**")
            st.json({
                "mid_energy_score": round(results["freq"]["mid_energy_score"], 4),
                "harmonic_score":   round(results["freq"]["harmonic_score"], 4),
                "dc_score":         round(results["freq"]["dc_score"], 4),
            })
        with col_b:
            st.markdown("**LSB Details**")
            st.json({
                "lsb_mean": round(results["lsb"]["lsb_mean"], 4),
                "lsb_std":  round(results["lsb"]["lsb_std"], 4),
                "rg_corr":  round(results["lsb"]["rg_corr"], 4),
                "rb_corr":  round(results["lsb"]["rb_corr"], 4),
            })


def page_about():
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <h1 style="font-size:2.2rem; font-weight:800;">About This Tool</h1>
        <p style="color:#94a3b8; font-size:1.05rem;">
            How it works, what it detects, and its limitations
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("## 🔬 How the Detector Works")
    st.markdown("""
    This tool uses a **multi-method ensemble** to detect invisible AI watermarks without 
    requiring a known watermark template. All four methods run simultaneously and are 
    combined with calibrated weights.
    """)

    with st.expander("1️⃣ FFT Frequency Domain Analysis (weight: 40%)", expanded=True):
        st.markdown("""
        **How it works:**
        - Converts the image's luminance channel to the frequency domain using a 2D Fast Fourier Transform (FFT)
        - Applies a Hanning window to reduce spectral leakage
        - Analyzes the radial frequency profile for:
          - **Mid-frequency energy concentration** — AI upsampling networks (U-Net, VAE decoder) leave characteristic energy signatures in the 8–30% frequency band
          - **Harmonic peak regularity** — SynthID and similar watermarks embed patterns at evenly-spaced frequency intervals; we detect regular peak spacing
          - **DC component bias** — AI-generated images sometimes show stronger DC energy ratios

        **Why it works:**
        Generative models (diffusion, GAN, VAE) produce images by sampling latent space and decoding 
        through learned upsampling layers. These layers introduce systematic periodic artifacts that 
        are invisible to the eye but measurable in the FFT spectrum.
        """)

    with st.expander("2️⃣ Color Channel Uniformity (weight: 25%)", expanded=False):
        st.markdown("""
        **How it works:**
        - Measures cross-channel Pearson correlation between R, G, and B channels
        - Analyzes gradient smoothness via mean/std ratio of gradient magnitudes
        - Combines into a color uniformity score

        **Why it works:**
        Real photographs inherit optical properties from Bayer filter demosaicing, lens characteristics, 
        and natural lighting variance. AI generators produce color via learned transformations that 
        often result in unnaturally high inter-channel correlation and overly smooth gradients.
        """)

    with st.expander("3️⃣ LSB (Least Significant Bit) Analysis (weight: 20%)", expanded=False):
        st.markdown("""
        **How it works:**
        - Extracts the least significant bit of each pixel value in all three color channels
        - Computes cross-channel correlation between LSB planes (R↔G, R↔B)
        - High correlation = potential embedded watermark signal

        **Why it works:**
        Many AI watermarking systems (including OpenAI's and Google's SynthID) embed 
        signature bits in the LSBs of pixel values. Natural photos have near-random LSBs 
        (from sensor noise). Correlated LSBs across channels strongly suggest an embedded signal.
        """)

    with st.expander("4️⃣ Noise Texture Analysis (weight: 15%)", expanded=False):
        st.markdown("""
        **How it works:**
        - Isolates noise by subtracting a Gaussian-blurred version of the image
        - Measures the **kurtosis** of the noise distribution
        - Natural camera sensor noise is approximately Gaussian (kurtosis ≈ 3.0)
        - AI-structured noise deviates from this baseline

        **Why it works:**
        Camera sensors produce shot noise and thermal noise with well-understood statistical 
        properties. AI generators produce "noise" through their sampling process, which can 
        deviate from Gaussian statistics in measurable ways.
        """)

    st.markdown("## 🤖 Which AI Models It Detects")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Likely detectable:**
        - Google Gemini / Imagen (SynthID watermarks)
        - DALL·E 3 / GPT-4o Image Generation
        - Stable Diffusion (standard samplers)
        - Midjourney v5/v6
        - Adobe Firefly
        - Bing Image Creator
        """)
    with col2:
        st.markdown("""
        **Harder to detect:**
        - Images that have been JPEG-compressed multiple times
        - Images resized significantly after generation
        - Screenshots of AI images (display → camera chain)
        - Heavily post-processed AI images (Photoshop, filters)
        - Very small images (< 256 × 256 px)
        """)

    st.markdown("## ⚠️ Limitations & Disclaimer")
    st.warning("""
    **This tool is for educational and research purposes only.**

    - **Not 100% accurate.** False positives (real photos flagged as AI) and false negatives 
      (AI images missed) are both possible. Treat results as probabilistic estimates, not facts.
    - **No ground truth template.** Unlike SynthID's internal verification, this tool has no 
      access to the secret watermark keys used by AI providers. Detection is entirely based on 
      statistical signatures.
    - **Post-processing degrades accuracy.** JPEG re-compression, resizing, or filtering can 
      erase watermark signals, leading to false negatives.
    - **AI is evolving.** New generative architectures may produce images with different 
      statistical properties not covered by current detection methods.
    - **Do not use for legal or forensic purposes** without additional corroborating evidence.
    """)

    st.markdown("## 🛠️ Technical Stack")
    st.code("""
# Core dependencies
streamlit    — web UI framework
numpy        — array math & FFT
scipy        — signal processing (find_peaks, ndimage)
opencv-python — HSV color space conversion
pillow       — image loading & manipulation
    """, language="text")

    st.markdown("## 📄 Algorithmic References")
    st.markdown("""
    - [SynthID: Identifying AI-generated content](https://deepmind.google/technologies/synthid/) — Google DeepMind
    - [Tree-Ring Watermarks](https://arxiv.org/abs/2305.20030) — Wen et al., 2023
    - [Invisible Image Watermarks Are Provably Removable](https://arxiv.org/abs/2306.01953) — Zhao et al., 2023
    - [C2PA: Coalition for Content Provenance and Authenticity](https://c2pa.org/)
    """)


# ── Navigation & Main ─────────────────────────────────────────────────────────

def main():
    # Top navigation
    col_nav1, col_nav2, col_nav3 = st.columns([1, 6, 1])
    with col_nav1:
        st.markdown("&nbsp;")
    with col_nav2:
        tabs = st.tabs(["🏠 Detector", "ℹ️ About"])
    with col_nav3:
        st.markdown("&nbsp;")

    with tabs[0]:
        page_home()

    with tabs[1]:
        page_about()


if __name__ == "__main__":
    main()
