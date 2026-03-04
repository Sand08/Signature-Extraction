"""
High-Precision Signature Recovery
Streamlit App — UNet + Pix2Pix Pipeline
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import time
import os

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Signature Recovery",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* Global */
[data-testid="stAppViewContainer"] {
    background: #0f1117;
    color: #e8eaf0;
}
[data-testid="stSidebar"] {
    background: #161b26;
    border-right: 1px solid #2a2f3e;
}
/* Cards */
.card {
    background: #1a1f2e;
    border: 1px solid #2a2f3e;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.metric-card {
    background: linear-gradient(135deg, #1e2540, #252b40);
    border: 1px solid #3a4060;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #7c9ef5;
}
.metric-label {
    font-size: 0.78rem;
    color: #8890a8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}
.metric-range {
    font-size: 0.7rem;
    color: #5a6278;
    margin-top: 0.15rem;
}
/* Pipeline step */
.pipeline-step {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.5rem 0.8rem;
    border-radius: 8px;
    margin: 0.3rem 0;
    font-size: 0.85rem;
}
.step-done   { background: #1a3a2a; border: 1px solid #2a5a3a; color: #4edb8a; }
.step-active { background: #1a2a4a; border: 1px solid #3a4a7a; color: #6ab0ff; }
.step-wait   { background: #1a1f2e; border: 1px solid #2a2f3e; color: #5a6278; }
/* Tabs */
[data-baseweb="tab-list"] { background: #161b26; border-radius: 10px; padding: 4px; }
[data-baseweb="tab"]      { color: #8890a8; border-radius: 8px; }
[aria-selected="true"]    { background: #2a3050 !important; color: #7c9ef5 !important; }
/* Headers */
h1 { color: #e8eaf0 !important; }
h2 { color: #c8cadc !important; }
h3 { color: #a8aabc !important; }
/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #3a5fc8, #5a3fc8);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
    width: 100%;
    transition: all .2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #4a6fd8, #6a4fd8);
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(90,63,200,.35);
}
/* Image labels */
.img-label {
    text-align: center;
    font-size: 0.78rem;
    color: #8890a8;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 0.3rem;
}
/* Separator */
hr { border-color: #2a2f3e; }
/* Alerts */
.stAlert { border-radius: 10px; }
/* File uploader */
[data-testid="stFileUploader"] {
    background: #1a1f2e;
    border: 2px dashed #3a4060;
    border-radius: 12px;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Model loading  (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models(unet_path: str, pix2pix_path: str):
    import tensorflow as tf
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    UNET_PATH = os.path.join(BASE_DIR, "models", "unet_soft_best-3.keras")
    PIX2PIX_PATH = os.path.join(BASE_DIR, "models", "pix2pix_gen_e3.keras")
    DEEPLAB_PATH = os.path.join(BASE_DIR, "models", "deeplab_best-2.keras")

    unet_model = tf.keras.models.load_model(UNET_PATH, compile=False)
    pix2pix_model = tf.keras.models.load_model(PIX2PIX_PATH, compile=False)
    deeplab_model = tf.keras.models.load_model(DEEPLAB_PATH, compile=False)
    return unet_model, pix2pix_model, deeplab_model

# ─────────────────────────────────────────────
# Image processing helpers
# ─────────────────────────────────────────────
IMG_SIZE = (512, 512)

def pil_to_array(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"))

def preprocess_for_unet(img_rgb: np.ndarray) -> np.ndarray:
    """Resize + normalise to [0,1], add batch dim."""
    resized = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    return np.expand_dims(x, 0)   # (1,512,512,3)

def preprocess_for_pix2pix(img_rgb: np.ndarray) -> np.ndarray:
    """Resize + normalise to [-1,1], add batch dim."""
    resized = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 127.5 - 1.0
    return np.expand_dims(x, 0)   # (1,512,512,3)

def postprocess_mask(prob: np.ndarray, thr: float = 0.35,
                     close_ksize: int = 3, close_iter: int = 1) -> np.ndarray:
    """prob: (512,512) float → binary uint8 {0,1}"""
    binary = (prob >= thr).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    b255 = (binary * 255).astype(np.uint8)
    b255 = cv2.morphologyEx(b255, cv2.MORPH_CLOSE, k, iterations=close_iter)
    return (b255 > 127).astype(np.uint8)

def extract_signature(rgb_01: np.ndarray, mask_01: np.ndarray) -> np.ndarray:
    """White background, signature pixels kept."""
    out = (rgb_01 * 255).astype(np.uint8).copy()
    out[mask_01 == 0] = 255
    return out

def compute_metrics(img_a: np.ndarray, img_b: np.ndarray):
    """
    Compute PSNR, SSIM, IoU between two (H,W,3) uint8 images.
    Returns dict with float values.
    """
    import tensorflow as tf

    a = tf.cast(img_a, tf.float32)[None] / 255.0
    b = tf.cast(img_b, tf.float32)[None] / 255.0

    psnr = float(tf.image.psnr(a, b, max_val=1.0).numpy())
    ssim = float(tf.image.ssim(a, b, max_val=1.0).numpy())

    # IoU on grayscale thresholded
    ga = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
    ma = (ga < 200).astype(np.uint8)   # foreground ~ dark pixels
    mb = (gb < 200).astype(np.uint8)
    inter = np.logical_and(ma, mb).sum()
    union = np.logical_or(ma, mb).sum()
    iou   = float(inter / union) if union > 0 else 0.0

    return {"psnr": psnr, "ssim": ssim, "iou": iou}

def array_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8))

def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✍️ Signature Recovery")
    st.markdown("<small style='color:#5a6278'>SRH University Heidelberg — MSc Data Science</small>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 🗂 Model Weights")
    unet_path    = st.text_input("U-Net weights (.keras)",    value="unet_soft_best-3.keras")
    pix2pix_path = st.text_input("Pix2Pix weights (.keras)", value="pix2pix_gen_e3.keras")
    deeplab_path  = st.text_input("DeepLab weights (.keras)", value="deeplab_best-2.keras")
    st.markdown("---")
    st.markdown("### ⚙️ Inference Settings")
    mask_threshold = st.slider("Mask threshold (U-Net)",  0.10, 0.80, 0.35, 0.05,
                               help="Pixel prob ≥ threshold → signature foreground")
    run_pipeline   = st.selectbox("Pipeline mode", ["U-Net only", "Pix2Pix only", "DeepLab only"])

    st.markdown("---")
    st.markdown("### 🔬 Pipeline")
    steps = [
        ("1. Upload image",         True),
        ("2. Pix2Pix denoising",    False),
        ("3. U-Net segmentation",   False),
        ("4. Signature extraction", False),
        ("5. Metrics",              False),
    ]
    for label, _ in steps:
        st.markdown(f"<div class='pipeline-step step-wait'>⬜ {label}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<small style='color:#5a6278'>Built with TensorFlow · OpenCV · Streamlit</small>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────
st.markdown("# High-Precision Signature Recovery")
st.markdown(
    "<p style='color:#8890a8;margin-top:-0.5rem;'>Semantic & Generative Pipeline — "
    "DeepLab U-Net + Pix2Pix cGAN</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Architecture diagram
with st.expander("📐 Pipeline Architecture", expanded=False):
    col_a, col_b, col_c, col_d, col_e = st.columns([2,1,2,1,2])
    with col_a:
        st.markdown("""
<div class='card' style='text-align:center;'>
<div style='font-size:2rem;'>🖼️</div>
<strong>Input</strong><br>
<small style='color:#8890a8'>Noisy scanned document<br>with stamps / artifacts</small>
</div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("<div style='text-align:center;padding-top:2rem;font-size:1.5rem;color:#7c9ef5'>→</div>", unsafe_allow_html=True)
    with col_c:
        st.markdown("""
<div class='card' style='text-align:center;'>
<div style='font-size:2rem;'>🔄</div>
<strong>Pix2Pix (cGAN)</strong><br>
<small style='color:#8890a8'>U-Net Generator + PatchGAN<br>Removes stamps · restores strokes</small>
</div>""", unsafe_allow_html=True)
    with col_d:
        st.markdown("<div style='text-align:center;padding-top:2rem;font-size:1.5rem;color:#7c9ef5'>→</div>", unsafe_allow_html=True)
    with col_e:
        st.markdown("""
<div class='card' style='text-align:center;'>
<div style='font-size:2rem;'>🎯</div>
<strong>U-Net Segmentation</strong><br>
<small style='color:#8890a8'>MobileNetV2 backbone<br>ASPP · pixel-wise mask</small>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div class='card' style='margin-top:0.5rem;'>
<b>Loss functions:</b> Pix2Pix = L1 loss + Adversarial (PatchGAN) · U-Net = BCE + Dice<br>
<b>Training:</b> Two-stage transfer learning — warm-up (frozen backbone) → full fine-tune<br>
<b>Metrics:</b> PSNR · SSIM · IoU · Pixel Accuracy
</div>
""", unsafe_allow_html=True)

st.markdown("")

# ── Upload
st.markdown("### 📤 Upload Document Image")
uploaded = st.file_uploader(
    "Upload a scanned signature document (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.markdown("""
<div class='card' style='text-align:center;padding:2.5rem;'>
<div style='font-size:3rem;'>📄</div>
<p style='color:#8890a8;margin-bottom:0;'>Upload a document image to begin signature extraction</p>
<p style='color:#5a6278;font-size:0.8rem;'>Supported: JPG · PNG · Max recommended: 4 MP</p>
</div>""", unsafe_allow_html=True)
    st.stop()

# ── Load image
pil_input = Image.open(uploaded).convert("RGB")
img_rgb   = pil_to_array(pil_input)
H, W      = img_rgb.shape[:2]

st.markdown("---")
st.markdown("### 📷 Input Image")
col1, col2 = st.columns([1, 2])
with col1:
    st.image(pil_input, use_column_width=True)
    st.markdown(f"<div class='img-label'>Original · {W}×{H}px</div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
<div class='card'>
<b>File:</b> {uploaded.name}<br>
<b>Dimensions:</b> {W} × {H} px<br>
<b>Mode:</b> Pipeline → <code>{run_pipeline}</code><br>
<b>Mask threshold:</b> {mask_threshold}<br><br>
<small style='color:#5a6278'>Images are resized to 512×512 for inference, then results are upscaled back to original resolution.</small>
</div>
""", unsafe_allow_html=True)

# ── Run button
st.markdown("")
run_btn = st.button("🚀  Run Signature Recovery", use_container_width=True)

if not run_btn:
    st.stop()

# ─────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────
with st.spinner("Loading model weights …"):
    try:
        unet_model, pix2pix_model, deep = load_models(unet_path, pix2pix_path)
    except Exception as e:
        st.error(f"❌ Failed to load models: {e}")
        st.info(
            "Make sure `unet_soft_best-3.keras`, `pix2pix_gen_e3.keras`, and `deeplab_best-2.keras` "
            "are in the same directory as this script, or update the paths in the sidebar."
        )
        st.stop()

st.success("✅ Models loaded successfully")
st.markdown("---")

# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
progress = st.progress(0, text="Starting pipeline …")
results  = {}
timing   = {}

# ── Step 1 — Pix2Pix denoising
if run_pipeline == "Pix2Pix only":
    progress.progress(15, text="⚡ Step 1/3 — Pix2Pix denoising …")
    t0 = time.time()

    x_p2p     = preprocess_for_pix2pix(img_rgb)
    y_p2p_m11 = pix2pix_model(x_p2p, training=False)[0].numpy()
    y_p2p_01  = np.clip((y_p2p_m11 + 1.0) / 2.0, 0, 1)
    y_p2p_rgb = (y_p2p_01 * 255).astype(np.uint8)

    # resize back to original
    y_p2p_orig = cv2.resize(y_p2p_rgb, (W, H), interpolation=cv2.INTER_LINEAR)

    results["pix2pix_512"] = y_p2p_rgb
    results["pix2pix_orig"] = y_p2p_orig
    timing["pix2pix"] = time.time() - t0
    progress.progress(40, text="✅ Pix2Pix done")

# ── Step 2 — Segmentation (U-Net / DeepLab)
if run_pipeline in ("U-Net only", "DeepLab only"):
    seg_model_name = "U-Net" if run_pipeline == "U-Net only" else "DeepLab"
    seg_model = unet_model if run_pipeline == "U-Net only" else deep
    progress.progress(50, text=f"⚡ Step 2/3 — {seg_model_name} segmentation …")
    t0 = time.time()

    seg_input_rgb = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    seg_input_rgb_orig = img_rgb

    x_seg  = np.expand_dims(seg_input_rgb.astype(np.float32) / 255.0, 0)
    prob   = seg_model.predict(x_seg, verbose=0)[0, ..., 0]   # (512,512)
    mask512 = postprocess_mask(prob, thr=mask_threshold)

    # resize mask back to original
    mask_orig = cv2.resize(mask512, (W, H), interpolation=cv2.INTER_NEAREST)

    results["prob_map"]    = prob
    results["mask_512"]    = mask512
    results["mask_orig"]   = mask_orig
    results["seg_input"]   = seg_input_rgb_orig
    results["seg_model_name"] = seg_model_name
    timing["segmentation"] = time.time() - t0
    progress.progress(75, text=f"✅ {seg_model_name} done")

# ── Step 3 — Extract signature
progress.progress(85, text="⚡ Step 3/3 — Extracting signature …")
t0 = time.time()

if run_pipeline in ("U-Net only", "DeepLab only"):
    base_rgb = img_rgb
    base_01  = base_rgb.astype(np.float32) / 255.0
    mask_for_extract = results["mask_orig"]
else:  # Pix2Pix only — no mask, just show denoised
    base_rgb = results["pix2pix_orig"]
    base_01  = base_rgb.astype(np.float32) / 255.0
    mask_for_extract = None

if mask_for_extract is not None:
    extracted_rgb = extract_signature(base_01, mask_for_extract)
    # black on white (monochrome)
    gray = cv2.cvtColor(extracted_rgb, cv2.COLOR_RGB2GRAY)
    _, sig_bw = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
    sig_bw = 255 - sig_bw   # invert: ink=255, bg=0 ... then flip for display
    results["extracted_rgb"] = extracted_rgb
    results["sig_bw"]        = sig_bw

timing["extract"] = time.time() - t0
progress.progress(100, text="✅ Pipeline complete!")
time.sleep(0.4)
progress.empty()

# ─────────────────────────────────────────────
# Results display
# ─────────────────────────────────────────────
st.markdown("## 🎯 Results")

tabs = st.tabs(["🖼️ Side-by-Side", "🔬 Detailed Steps", "📊 Metrics", "💾 Downloads"])

# ── TAB 1: Side-by-Side
with tabs[0]:
    if run_pipeline in ("U-Net only", "DeepLab only"):
        seg_model_name = results.get("seg_model_name", "Segmentation")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(pil_input, use_column_width=True)
            st.markdown("<div class='img-label'>Input (Noisy)</div>", unsafe_allow_html=True)
        with c2:
            mask_disp = (results["mask_orig"] * 255).astype(np.uint8)
            st.image(mask_disp, use_column_width=True, clamp=True)
            st.markdown(f"<div class='img-label'>{seg_model_name} Mask</div>", unsafe_allow_html=True)
        with c3:
            st.image(results["extracted_rgb"], use_column_width=True)
            st.markdown("<div class='img-label'>Extracted Signature</div>", unsafe_allow_html=True)

    else:  # Pix2Pix only
        c1, c2 = st.columns(2)
        with c1:
            st.image(pil_input, use_column_width=True)
            st.markdown("<div class='img-label'>Input (Noisy)</div>", unsafe_allow_html=True)
        with c2:
            st.image(results["pix2pix_orig"], use_column_width=True)
            st.markdown("<div class='img-label'>Pix2Pix Cleaned</div>", unsafe_allow_html=True)

# ── TAB 2: Detailed Steps
with tabs[1]:
    st.markdown("#### Step-by-step inspection")

    if "pix2pix_512" in results:
        st.markdown("**Pix2Pix Denoising**")
        c1, c2 = st.columns(2)
        with c1:
            inp_512 = cv2.resize(img_rgb, IMG_SIZE)
            st.image(inp_512, use_column_width=True)
            st.markdown("<div class='img-label'>Input @ 512×512</div>", unsafe_allow_html=True)
        with c2:
            st.image(results["pix2pix_512"], use_column_width=True)
            st.markdown("<div class='img-label'>Pix2Pix Output @ 512×512</div>", unsafe_allow_html=True)

        # Absolute difference
        diff = np.abs(inp_512.astype(np.int32) - results["pix2pix_512"].astype(np.int32))
        diff_norm = ((diff / diff.max()) * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)
        st.markdown("**Absolute difference (noisy − cleaned)**")
        st.image(diff_norm, use_column_width=False, width=400)
        st.markdown("<div class='img-label'>Artifact map — brighter = more correction</div>", unsafe_allow_html=True)
        st.markdown("---")

    if "prob_map" in results:
        seg_model_name = results.get("seg_model_name", "Segmentation")
        st.markdown(f"**{seg_model_name} Segmentation**")
        c1, c2, c3 = st.columns(3)
        with c1:
            seg_disp = cv2.resize(results["seg_input"], IMG_SIZE)
            st.image(seg_disp, use_column_width=True)
            st.markdown("<div class='img-label'>Seg input @ 512×512</div>", unsafe_allow_html=True)
        with c2:
            prob_disp = (results["prob_map"] * 255).astype(np.uint8)
            st.image(prob_disp, use_column_width=True, clamp=True)
            st.markdown("<div class='img-label'>Probability map</div>", unsafe_allow_html=True)
        with c3:
            mask_disp = (results["mask_512"] * 255).astype(np.uint8)
            st.image(mask_disp, use_column_width=True, clamp=True)
            st.markdown(f"<div class='img-label'>Binary mask (thr={mask_threshold})</div>", unsafe_allow_html=True)
        st.markdown("---")

    if "extracted_rgb" in results:
        st.markdown("**Final Extracted Signature**")
        c1, c2 = st.columns(2)
        with c1:
            st.image(results["extracted_rgb"], use_column_width=True)
            st.markdown("<div class='img-label'>Extracted (colour, white BG)</div>", unsafe_allow_html=True)
        with c2:
            sig_bw_disp = 255 - results["sig_bw"]   # black ink on white for display
            st.image(sig_bw_disp, use_column_width=True, clamp=True)
            st.markdown("<div class='img-label'>Binary (black ink / white BG)</div>", unsafe_allow_html=True)

# ── TAB 3: Metrics
with tabs[2]:
    st.markdown("#### Quantitative Metrics")

    # Timing metrics
    st.markdown("**⏱ Inference timing**")
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        st.markdown(f"""
<div class='metric-card'>
<div class='metric-value'>{timing.get('pix2pix', 0):.2f}s</div>
<div class='metric-label'>Pix2Pix</div>
<div class='metric-range'>Denoising pass</div>
</div>""", unsafe_allow_html=True)
    with tc2:
        seg_label = results.get("seg_model_name", "Segmentation")
        st.markdown(f"""
<div class='metric-card'>
<div class='metric-value'>{timing.get('segmentation', 0):.2f}s</div>
<div class='metric-label'>{seg_label}</div>
<div class='metric-range'>Segmentation pass</div>
</div>""", unsafe_allow_html=True)
    with tc3:
        total = sum(timing.values())
        st.markdown(f"""
<div class='metric-card'>
<div class='metric-value'>{total:.2f}s</div>
<div class='metric-label'>Total</div>
<div class='metric-range'>End-to-end pipeline</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Image quality metrics (Pix2Pix input vs output if available)
    if "pix2pix_512" in results:
        st.markdown("**📈 Image Quality Metrics — Input vs Pix2Pix Output**")
        inp_512 = cv2.resize(img_rgb, IMG_SIZE)
        m = compute_metrics(inp_512, results["pix2pix_512"])

        mc1, mc2, mc3 = st.columns(3)
        psnr_col = "#4edb8a" if m["psnr"] >= 27 else ("#f5c842" if m["psnr"] >= 20 else "#f56c6c")
        ssim_col = "#4edb8a" if m["ssim"] >= 0.88 else ("#f5c842" if m["ssim"] >= 0.7 else "#f56c6c")

        with mc1:
            st.markdown(f"""
<div class='metric-card'>
<div class='metric-value' style='color:{psnr_col}'>{m['psnr']:.2f} dB</div>
<div class='metric-label'>PSNR</div>
<div class='metric-range'>Target: 27–30 dB</div>
</div>""", unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""
<div class='metric-card'>
<div class='metric-value' style='color:{ssim_col}'>{m['ssim']:.4f}</div>
<div class='metric-label'>SSIM</div>
<div class='metric-range'>Target: 0.88–0.94</div>
</div>""", unsafe_allow_html=True)
        with mc3:
            st.markdown(f"""
<div class='metric-card'>
<div class='metric-value'>{m['iou']:.4f}</div>
<div class='metric-label'>IoU (ink pixels)</div>
<div class='metric-range'>Foreground overlap</div>
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div class='card' style='margin-top:0.8rem;'>
<small style='color:#8890a8'>
<b>PSNR</b> (Peak Signal-to-Noise Ratio): measures pixel-level fidelity. ≥27 dB indicates strong reconstruction.<br>
<b>SSIM</b> (Structural Similarity Index): perceptual similarity. ≥0.88 validates morphological continuity.<br>
<b>IoU</b>: foreground ink-pixel overlap between input and output. High values indicate minimal stroke loss.
</small>
</div>""", unsafe_allow_html=True)

    if "mask_orig" in results:
        st.markdown("**🎭 Segmentation Mask Stats**")
        mask = results["mask_orig"]
        fg_pct = mask.mean() * 100
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown(f"""
<div class='metric-card'>
<div class='metric-value'>{fg_pct:.1f}%</div>
<div class='metric-label'>Foreground</div>
<div class='metric-range'>Signature pixels</div>
</div>""", unsafe_allow_html=True)
        with sc2:
            st.markdown(f"""
<div class='metric-card'>
<div class='metric-value'>{mask_threshold}</div>
<div class='metric-label'>Threshold</div>
<div class='metric-range'>Applied cutoff</div>
</div>""", unsafe_allow_html=True)
        with sc3:
            st.markdown(f"""
<div class='metric-card'>
<div class='metric-value'>{W}×{H}</div>
<div class='metric-label'>Resolution</div>
<div class='metric-range'>Output size (px)</div>
</div>""", unsafe_allow_html=True)

    if "pix2pix_512" not in results and "mask_orig" not in results:
        st.info("Run the Full or individual pipelines to see detailed metrics.")

# ── TAB 4: Downloads
with tabs[3]:
    st.markdown("#### 💾 Download Outputs")

    dl_cols = []
    dl_items = []

    if "pix2pix_orig" in results:
        dl_items.append(("Pix2Pix cleaned (PNG)", results["pix2pix_orig"], "pix2pix_cleaned.png"))
    if "mask_orig" in results:
        mask_save = (results["mask_orig"] * 255).astype(np.uint8)
        seg_model_name = results.get("seg_model_name", "segmentation").lower()
        dl_items.append((f"{seg_model_name.title()} mask (PNG)", mask_save, f"{seg_model_name}_mask.png"))
    if "extracted_rgb" in results:
        dl_items.append(("Extracted signature (PNG)", results["extracted_rgb"], "signature_extracted.png"))
    if "sig_bw" in results:
        bw_save = 255 - results["sig_bw"]
        dl_items.append(("Binary signature (PNG)", bw_save, "signature_binary.png"))

    if dl_items:
        for label, arr, fname in dl_items:
            pil_out   = array_to_pil(arr) if arr.ndim == 3 else Image.fromarray(arr)
            img_bytes = pil_to_bytes(pil_out)
            st.download_button(
                label=f"⬇️ {label}",
                data=img_bytes,
                file_name=fname,
                mime="image/png",
                use_container_width=True,
            )
    else:
        st.info("Run the pipeline first to generate downloadable outputs.")

st.markdown("---")
st.markdown(
    "<small style='color:#5a6278'>High-Precision Signature Recovery · SRH University Heidelberg · "
    "Sandeep Sharma · 2025 — "
    "<a href='https://github.com/Sand08/Signature-Extraction' style='color:#5a6278'>GitHub</a></small>",
    unsafe_allow_html=True,
)
