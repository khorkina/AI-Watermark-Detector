# AI Watermark Detector

A Streamlit web application that detects invisible AI watermarks in images using FFT-based frequency analysis, saturation enhancement, LSB pattern detection, and noise texture analysis.

## Run & Operate

- `cd artifacts/ai-watermark-detector && streamlit run app.py` — run the Streamlit app (port 5000)
- Workflow name: **AI Watermark Detector**

## Stack

- Python 3.11
- Streamlit 1.57 — web UI framework
- NumPy — FFT and array math
- SciPy — signal processing (find_peaks, ndimage filters)
- OpenCV (opencv-python) — HSV color space conversion
- Pillow — image loading and manipulation

## Where things live

- `artifacts/ai-watermark-detector/app.py` — single-file Streamlit app (all detection logic + UI)
- `artifacts/ai-watermark-detector/.streamlit/config.toml` — server config (port 5000, headless)

## Architecture decisions

- Single-file Streamlit app — stateless, no database, no backend needed
- `@st.cache_data` on `analyze_image()` — re-renders are instant after first analysis
- Four-method ensemble with calibrated weights (FFT 40%, color 25%, LSB 20%, noise 15%)
- Sigmoid stretch on combined score to push borderline results toward a clear decision
- OpenCV used for HSV saturation boost; graceful numpy fallback if cv2 unavailable

## Product

Users upload any JPG/PNG/WEBP image (up to 10 MB) and instantly see:
- A verdict (AI-generated / Real / Inconclusive) with a confidence percentage
- Original image, saturation-enhanced visualization, and FFT spectrum side by side
- Detailed sub-score breakdown with explanations
- An About page explaining the algorithms, detectable models, and limitations

## User preferences

_Populate as you build — explicit user instructions worth remembering across sessions._

## Gotchas

- The app must be run from `artifacts/ai-watermark-detector/` for the `.streamlit/config.toml` to be picked up
- OpenCV is required for the HSV saturation boost; a NumPy fallback exists but is less accurate
- Large images are auto-resized to 1024px max before analysis (keeps FFT fast)
- `scipy.signal.find_peaks` is used for harmonic peak detection — requires scipy >= 1.7

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
