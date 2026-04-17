import streamlit as st
import numpy as np
import nibabel as nib
import tempfile
import zipfile
import io
import base64
import textwrap
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Keeping your original logic imports
from preprocessing.preprocess_ct import preprocess_ct
from preprocessing.preprocess_mri import preprocess_slice
from models.ct_model import predict_ct, load_ct_model
from models.mri_model import predict_mri
from xai.xai_ct import get_gradcam_overlay
from xai.xai_mri import get_mri_overlay
from utils.severity import ct_severity, mri_severity
from utils.report import generate_report

st.set_page_config(
    page_title="NeuroVision AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- SESSION STATE ---
if "page" not in st.session_state: st.session_state.page = "clinician"
if "patient_unlocked" not in st.session_state: st.session_state.patient_unlocked = False
if "last_modality" not in st.session_state: st.session_state.last_modality = None
if "last_label" not in st.session_state: st.session_state.last_label = None
if "last_severity" not in st.session_state: st.session_state.last_severity = None
if "last_prob" not in st.session_state: st.session_state.last_prob = None
if "last_tumor_pct" not in st.session_state: st.session_state.last_tumor_pct = None
if "last_overall_tumor_pct" not in st.session_state: st.session_state.last_overall_tumor_pct = None

# Persist visuals and summaries so clinician view can be restored
if "last_ct_overlay" not in st.session_state: st.session_state.last_ct_overlay = None
if "last_mri_fig" not in st.session_state: st.session_state.last_mri_fig = None
if "last_mri_overlay" not in st.session_state: st.session_state.last_mri_overlay = None
if "last_dice" not in st.session_state: st.session_state.last_dice = None
if "last_iou" not in st.session_state: st.session_state.last_iou = None
if "last_clinical_summary_html" not in st.session_state: st.session_state.last_clinical_summary_html = ""
if "patient_profile" not in st.session_state:
    st.session_state.patient_profile = {
        "name": "Rahul Verma",
        "patient_id": "NV-2026-04871",
        "dob": "1992-09-18",
        "age_gender": "33 / Male",
        "physician": "Dr. Meera Kapoor",
        "facility": "NeuroVision Partner Hospital",
    }

# Rotating tips counter (0=MRI, 1=CT, 2=General)
@st.cache_resource
def _tip_rotation_store():
    return {"idx": -1}

def next_tip_idx():
    store = _tip_rotation_store()
    store["idx"] = (store["idx"] + 1) % 3
    return store["idx"]


def _normalize_preview_slice(arr):
    x = np.asarray(arr, dtype=np.float32)
    if x.size == 0:
        return np.zeros((128, 128), dtype=np.float32)
    lo = np.percentile(x, 1)
    hi = np.percentile(x, 99)
    if hi <= lo:
        lo, hi = float(np.min(x)), float(np.max(x))
    x = np.clip(x, lo, hi)
    den = float(np.max(x) - np.min(x))
    if den <= 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - np.min(x)) / den


@st.cache_data(show_spinner=False)
def _build_real_scan_preview_tiles_b64():
    try:
        base_dir = Path(__file__).resolve().parent

        # MRI sample (BraTS case)
        mri_root = base_dir / "assets" / "brats_subset"
        case_dirs = sorted([p for p in mri_root.iterdir() if p.is_dir()]) if mri_root.exists() else []
        if not case_dirs:
            return None
        case = case_dirs[0]

        flair_path = next(case.glob("*flair*.nii*"), None)
        t1ce_path = next(case.glob("*t1ce*.nii*"), None)
        seg_path = next(case.glob("*seg*.nii*"), None)
        if flair_path is None:
            return None

        flair_img = nib.load(str(flair_path))
        flair_vol = np.asarray(flair_img.dataobj)
        z = int(flair_vol.shape[2] // 2)
        flair_slice = _normalize_preview_slice(flair_vol[:, :, z])

        t1ce_slice = None
        if t1ce_path is not None:
            t1ce_vol = np.asarray(nib.load(str(t1ce_path)).dataobj)
            z2 = min(z, t1ce_vol.shape[2] - 1)
            t1ce_slice = _normalize_preview_slice(t1ce_vol[:, :, z2])

        seg_mask = None
        if seg_path is not None:
            seg_vol = np.asarray(nib.load(str(seg_path)).dataobj)
            z3 = min(z, seg_vol.shape[2] - 1)
            seg_mask = (seg_vol[:, :, z3] > 0).astype(np.float32)

        # CT samples (positive and negative DICOM)
        ct_root = base_dir / "assets" / "ct_subset_30"
        ct_pos_path = next((ct_root / "positive").glob("*.dcm"), None) if (ct_root / "positive").exists() else None
        ct_neg_path = next((ct_root / "negative").glob("*.dcm"), None) if (ct_root / "negative").exists() else None

        if ct_pos_path is None or ct_neg_path is None:
            return None

        ct_pos_img = preprocess_ct(str(ct_pos_path))
        ct_neg_img = preprocess_ct(str(ct_neg_path))
        if ct_pos_img is None or ct_neg_img is None:
            return None

        ct_pos_slice = np.asarray(ct_pos_img[:, :, 0], dtype=np.float32)
        ct_neg_slice = np.asarray(ct_neg_img[:, :, 0], dtype=np.float32)

        def _to_u8(x):
            x = np.asarray(x, dtype=np.float32)
            x = np.clip(x, 0.0, 1.0)
            return (x * 255.0).astype(np.uint8)

        def _to_bgr(x):
            return cv2.cvtColor(_to_u8(x), cv2.COLOR_GRAY2BGR)

        def _label(tile, txt):
            cv2.rectangle(tile, (6, 6), (tile.shape[1] - 6, 24), (235, 243, 249), -1)
            cv2.putText(tile, txt, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (42, 76, 104), 1, cv2.LINE_AA)
            return tile

        tile_size = 180
        gap = 10

        mri_flair = cv2.resize(_to_bgr(flair_slice), (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)

        mri_seg_base = t1ce_slice if t1ce_slice is not None else flair_slice
        mri_seg = cv2.resize(_to_bgr(mri_seg_base), (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
        if seg_mask is not None:
            seg_u8 = cv2.resize((seg_mask * 255).astype(np.uint8), (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)
            seg_heat = cv2.applyColorMap(seg_u8, cv2.COLORMAP_JET)
            mri_seg = cv2.addWeighted(mri_seg, 0.74, seg_heat, 0.42, 0)

        ct_plain = cv2.resize(_to_bgr(ct_neg_slice), (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)

        ct_src = cv2.resize(_to_u8(ct_pos_slice), (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
        ct_heat_seed = cv2.GaussianBlur(ct_src, (0, 0), 5)
        ct_heat_seed = cv2.normalize(ct_heat_seed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        ct_heat = cv2.applyColorMap(ct_heat_seed, cv2.COLORMAP_JET)
        ct_base = cv2.cvtColor(ct_src, cv2.COLOR_GRAY2BGR)
        ct_xai = cv2.addWeighted(ct_base, 0.66, ct_heat, 0.5, 0)

        mri_flair = _label(mri_flair, "MRI FLAIR")
        mri_seg = _label(mri_seg, "MRI Seg Overlay")
        ct_plain = _label(ct_plain, "CT Slice")
        ct_xai = _label(ct_xai, "CT XAI-like Overlay")

        def _encode_tile(tile):
            ok, enc = cv2.imencode(".png", tile)
            return base64.b64encode(enc.tobytes()).decode("utf-8") if ok else None

        tiles = {
            "mri_flair": _encode_tile(mri_flair),
            "mri_seg": _encode_tile(mri_seg),
            "ct_slice": _encode_tile(ct_plain),
            "ct_xai": _encode_tile(ct_xai),
        }
        if any(v is None for v in tiles.values()):
            return None
        return tiles
    except Exception:
        return None


def _hero_scan_preview_html():
    tiles = _build_real_scan_preview_tiles_b64()
    if tiles:
        return textwrap.dedent(f"""
<div class=\"hero-media\">
    <div class=\"hero-media-head\">Scan Preview Collage</div>
    <div class=\"hero-collage-frame\">
        <div class=\"hero-collage-tile\">
            <img class=\"hero-collage-img\" src=\"data:image/png;base64,{tiles['mri_flair']}\" alt=\"MRI FLAIR\" />
            <span class=\"hero-collage-cap\">MRI FLAIR</span>
        </div>
        <div class=\"hero-collage-tile\">
            <img class=\"hero-collage-img\" src=\"data:image/png;base64,{tiles['mri_seg']}\" alt=\"MRI segmentation overlay\" />
            <span class=\"hero-collage-cap\">MRI Seg</span>
        </div>
        <div class=\"hero-collage-tile\">
            <img class=\"hero-collage-img\" src=\"data:image/png;base64,{tiles['ct_slice']}\" alt=\"CT slice\" />
            <span class=\"hero-collage-cap\">CT Slice</span>
        </div>
        <div class=\"hero-collage-tile\">
            <img class=\"hero-collage-img\" src=\"data:image/png;base64,{tiles['ct_xai']}\" alt=\"CT XAI-like overlay\" />
            <span class=\"hero-collage-cap\">AI Overlay</span>
        </div>
    </div>
</div>
""").strip()

    return textwrap.dedent("""
<div class=\"hero-media\">
    <div class=\"hero-media-head\">Scan Preview Collage</div>
    <div class=\"hero-media-grid\">
        <div class=\"hero-scan-card\"><span class=\"hero-scan-label\">MRI Slice</span></div>
        <div class=\"hero-scan-card seg\"><span class=\"hero-scan-label\">Segmentation</span></div>
        <div class=\"hero-scan-card\"><span class=\"hero-scan-label\">CT Slice</span></div>
        <div class=\"hero-scan-card seg\"><span class=\"hero-scan-label\">AI Overlay</span></div>
    </div>
</div>
""").strip()

# --- MODERN UI STYLING (CSS) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --primary: #0d6efd;
    --teal: #13a3a9;
    --navy: #0a192f;
    --glass: rgba(255, 255, 255, 0.85);
    --border: rgba(200, 210, 220, 0.5);
    --accent: #0d5e8d;
    --muted: #516579;
    --green: #1f7a47;
}

@keyframes riseFade {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes scanSweep {
    0% { transform: translateY(-110%); opacity: 0; }
    10% { opacity: 0.42; }
    100% { transform: translateY(260%); opacity: 0; }
}
@keyframes collagePulse {
    0%, 100% { transform: scale(1); filter: saturate(1); }
    50% { transform: scale(1.008); filter: saturate(1.03); }
}

.stApp {
    background:
        radial-gradient(ellipse 70% 55% at 90% -8%, rgba(13,94,141,0.18), transparent),
        radial-gradient(ellipse 62% 42% at 8% 102%, rgba(19,111,115,0.14), transparent),
        radial-gradient(circle 520px at 50% 0%, rgba(255,255,255,0.8), transparent 62%),
        linear-gradient(160deg, #f3f8fb 0%, #ecf3f7 56%, #f8fbfd 100%);
    color: var(--ink);
}

#MainMenu, footer { visibility: hidden; }

h1,h2,h3,h4 { font-family: 'Inter', sans-serif; color: var(--ink); letter-spacing: -0.02em; }
p,span,label,div,li { font-family: 'Inter', sans-serif; }
code,pre,.stCode { font-family: 'JetBrains Mono', monospace !important; }

[data-testid="stSidebar"] h3 { font-weight: 700 !important; }

.side-nav {
    border: 1px solid rgba(191,219,236,0.22);
    border-radius: 12px;
    background: rgba(241,249,255,0.07);
    overflow: hidden;
}
.side-nav-item {
    padding: 0.56rem 0.68rem;
    border-bottom: 1px solid rgba(191,219,236,0.16);
    font-size: 0.84rem;
}
.side-nav-item:last-child { border-bottom: 0; }
.side-nav-active {
    background: rgba(169,217,246,0.2);
    color: #f6fbff;
    font-weight: 700;
}
.side-nav-disabled {
    opacity: 0.6;
}

.modality-card {
    border: 1px solid rgba(175,194,208,0.95);
    border-radius: 16px;
    background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(246,250,252,0.93));
    box-shadow: 0 12px 22px rgba(15,34,52,0.06);
    padding: 0.95rem 1rem;
    margin-bottom: 0.9rem;
}
.modality-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    border-radius: 999px;
    padding: 0.35rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 700;
    border: 1px solid;
}
.modality-card.mri {
    border: 1px solid rgba(131,177,194,0.95);
    background: linear-gradient(140deg, rgba(232,246,252,0.96), rgba(243,252,255,0.92));
    box-shadow: 0 12px 24px rgba(18,82,107,0.10);
}
.modality-pill.mri {
    border-color: #7fb6cb;
    background: linear-gradient(120deg, #dff3f9, #e9f8fc);
    color: #155a73;
}
.modality-note {
    margin-top: 0.52rem;
    font-size: 0.8rem;
    color: #40647a;
}
.decision-card {
    border: 1px solid rgba(151,181,204,0.95);
    border-radius: 20px;
    background: linear-gradient(145deg, rgba(255,255,255,0.99), rgba(238,246,252,0.96));
    box-shadow: 0 16px 34px rgba(15,34,52,0.09);
    padding: 1.35rem 1.45rem;
    margin: 0.85rem 0 1.1rem;
}
.decision-head {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    margin-bottom: 1rem;
}
.decision-badge {
    border-radius: 999px;
    padding: 0.28rem 0.66rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.64rem;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    border: 1px solid;
    font-weight: 700;
}
.decision-badge.ct { border-color: #8cb8d9; background: #e8f3fb; color: #1e537a; }
.decision-badge.mri { border-color: #86c0cd; background: #e4f6f8; color: #1b656f; }
.decision-title {
    font-size: 1.28rem;
    color: #16364f;
    font-weight: 700;
}
.decision-strip {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.55rem;
    margin-bottom: 0.9rem;
}
.decision-chip {
    border: 1px solid rgba(167,191,209,0.85);
    border-radius: 11px;
    background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(244,249,253,0.95));
    padding: 0.45rem 0.55rem;
}
.decision-chip .k {
    font-family: 'Space Mono', monospace;
    font-size: 0.66rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #5a748a;
}
.decision-chip .v {
    margin-top: 0.15rem;
    color: #16354f;
    font-family: 'Fraunces', serif;
    font-size: 1.08rem;
}
.decision-grid {
    display: grid;
    grid-template-columns: 1.08fr 0.92fr;
    gap: 1rem;
    align-items: start;
}
.decision-left {
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
}
.decision-side {
    display: grid;
    justify-items: stretch;
    gap: 0.75rem;
}
.decision-side-grid {
    width: 100%;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.55rem;
}
.decision-mini {
    border: 1px solid rgba(167,191,209,0.85);
    border-radius: 11px;
    background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(242,249,253,0.95));
    padding: 0.58rem 0.62rem;
    min-height: 84px;
}
.decision-mini .k {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    color: #5a748a;
}
.decision-mini .v {
    margin-top: 0.14rem;
    color: #1a4260;
    font-family: 'Fraunces', serif;
    font-size: 1.08rem;
}
.decision-mini-bar {
    margin-top: 0.26rem;
    width: 100%;
    height: 6px;
    border-radius: 999px;
    background: #d8e7f2;
    overflow: hidden;
}
.decision-mini-bar > span {
    display: block;
    height: 100%;
    border-radius: 999px;
    background: #2a8a54;
}
.decision-mini-note {
    margin-top: 0.12rem;
    font-size: 0.72rem;
    color: #5a748a;
    font-family: 'Space Mono', monospace;
}
.clinical-board {
    border: 1px solid rgba(165,191,210,0.92);
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(242,248,252,0.95));
    padding: 0.95rem 1rem 1rem;
    box-shadow: 0 12px 24px rgba(15,34,52,0.06);
    margin-top: 0.85rem;
}
.clinical-board h3 {
    margin: 0 0 0.7rem;
    font-size: 1.05rem;
    color: #17364f;
}
.clinical-board-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.7rem;
}
.clinical-stat {
    border: 1px solid rgba(167,191,209,0.85);
    border-radius: 14px;
    background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(244,249,253,0.95));
    padding: 0.72rem 0.8rem;
}
.clinical-stat .k {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #5a748a;
}
.clinical-stat .v {
    margin-top: 0.2rem;
    color: #16354f;
    font-family: 'Fraunces', serif;
    font-size: 1rem;
}
.clinical-stat .m {
    margin-top: 0.14rem;
    color: #4b647b;
    font-size: 0.78rem;
    line-height: 1.45;
}
.clinical-insight {
    margin-top: 0.8rem;
    border-radius: 12px;
    border: 1px solid rgba(164,190,208,0.9);
    background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(242,248,252,0.95));
    padding: 0.72rem 0.82rem;
}
.clinical-insight .k {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #567086;
}
.clinical-insight .v {
    margin-top: 0.16rem;
    color: #2c4b62;
    font-size: 0.84rem;
    line-height: 1.5;
}
.error-ring {
    width: 56px;
    height: 56px;
    border-radius: 50%;
    display: grid;
    place-items: center;
    margin: 0.1rem auto 0.15rem;
    background: conic-gradient(var(--error-color, #2a8a54) calc(var(--error-pct, 0) * 1%), #d9e7f1 0);
    position: relative;
}
.error-ring::after {
    content: '';
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: linear-gradient(180deg, #ffffff, #f2f8fc);
    border: 1px solid rgba(166,191,209,0.7);
    position: absolute;
}
.error-ring span {
    position: relative;
    z-index: 1;
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem;
    color: #16354f;
}
.decision-k {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #5b748a;
}
.decision-v {
    margin: 0.2rem 0 0.5rem;
    color: #15354f;
    font-family: 'Fraunces', serif;
    font-size: 1.34rem;
}
.mini-conf {
    width: 100%;
    max-width: 420px;
    height: 10px;
    border-radius: 999px;
    background: #dceaf4;
    overflow: hidden;
    border: 1px solid rgba(154,184,206,0.6);
}
.mini-conf > span {
    display: block;
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--accent-color, #0d5e8d), #19a3ad);
}
.decision-ring {
    --p: 65;
    width: 178px;
    height: 178px;
    border-radius: 50%;
    background: conic-gradient(var(--ring-color, #0d5e8d) calc(var(--p) * 1%), #d7e6f1 0);
    display: grid;
    place-items: center;
    margin: 0 auto 0.2rem;
    position: relative;
}
.decision-ring::after {
    content: '';
    width: 124px;
    height: 124px;
    border-radius: 50%;
    background: linear-gradient(180deg, #ffffff, #f1f8fc);
    border: 1px solid rgba(166,191,209,0.7);
    position: absolute;
}
.decision-ring-label {
    position: relative;
    z-index: 1;
    text-align: center;
    color: #18425f;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
}
.decision-ring-label b {
    display: block;
    font-family: 'Fraunces', serif;
    font-size: 1.24rem;
    margin-top: 0.08rem;
}
.decision-ring-sub {
    margin-top: 0.12rem;
    text-align: center;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #5a748a;
}
.coverage-viz {
    border: 1px solid rgba(167,191,209,0.85);
    border-radius: 12px;
    background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(242,249,253,0.95));
    padding: 0.62rem 0.68rem;
}
.coverage-viz .k {
    font-family: 'Space Mono', monospace;
    font-size: 0.64rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #5a748a;
}
.coverage-viz .v {
    margin-top: 0.16rem;
    color: #16354f;
    font-family: 'Fraunces', serif;
    font-size: 1.1rem;
}
.coverage-stack {
    margin-top: 0.36rem;
    height: 16px;
    width: 100%;
    border-radius: 999px;
    overflow: hidden;
    border: 1px solid rgba(160,186,205,0.75);
    display: flex;
    background: #d7e6f1;
}
.coverage-tumor {
    height: 100%;
    min-width: 2px;
    background: linear-gradient(90deg, #c48b10, #d39a13);
}
.coverage-normal {
    height: 100%;
    background: #d7e6f1;
}
.coverage-legend {
    margin-top: 0.34rem;
    display: flex;
    justify-content: space-between;
    gap: 0.35rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.66rem;
    color: #4f687f;
}
.coverage-row {
    margin-top: 0.42rem;
}
.coverage-row:first-of-type {
    margin-top: 0.28rem;
}
.coverage-row-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 0.4rem;
}
.coverage-row-head .label {
    color: #3f5a72;
    font-size: 0.76rem;
}
.coverage-row-head .value {
    color: #17364f;
    font-family: 'Space Mono', monospace;
    font-size: 0.74rem;
}
.coverage-note {
    margin-top: 0.34rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.64rem;
    color: #5a748a;
}
.coverage-pie {
    --tumor-pct: 0;
    width: 144px;
    height: 144px;
    margin: 0.45rem auto 0.2rem;
    border-radius: 50%;
    background: conic-gradient(#d63c3c calc(var(--tumor-pct) * 1%), #d8e4ee 0);
    display: grid;
    place-items: center;
    position: relative;
}
.coverage-pie::after {
    content: '';
    width: 94px;
    height: 94px;
    border-radius: 50%;
    background: linear-gradient(180deg, #ffffff, #f2f8fc);
    border: 1px solid rgba(166,191,209,0.7);
    position: absolute;
}
.coverage-pie-label {
    position: relative;
    z-index: 1;
    text-align: center;
    color: #17364f;
    font-family: 'Space Mono', monospace;
    font-size: 0.66rem;
}
.coverage-pie-label b {
    display: block;
    font-family: 'Fraunces', serif;
    font-size: 1.04rem;
    margin-top: 0.06rem;
}
.coverage-pie-legend {
    margin-top: 0.34rem;
    display: flex;
    justify-content: space-between;
    gap: 0.42rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.63rem;
    color: #4f687f;
}
.coverage-metrics {
    margin-top: 0.42rem;
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.4rem;
}
.coverage-metric {
    border: 1px solid rgba(166,191,209,0.8);
    border-radius: 9px;
    background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(244,249,253,0.95));
    padding: 0.35rem 0.45rem;
}
.coverage-metric .k {
    font-family: 'Space Mono', monospace;
    font-size: 0.56rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #5a748a;
}
.coverage-metric .v {
    margin-top: 0.1rem;
    color: #17364f;
    font-family: 'Fraunces', serif;
    font-size: 0.86rem;
}
.coverage-ruler {
    margin-top: 0.42rem;
    position: relative;
    height: 34px;
}
.coverage-ruler-track {
    position: absolute;
    left: 0;
    right: 0;
    top: 14px;
    height: 8px;
    border-radius: 999px;
    border: 1px solid rgba(160,186,205,0.78);
    background: linear-gradient(90deg, #e7f1f8, #d5e6f2);
}
.coverage-ruler-pin {
    position: absolute;
    top: 0;
    transform: translateX(-50%);
    display: grid;
    justify-items: center;
    gap: 0.12rem;
}
.coverage-ruler-pin .dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    border: 2px solid #ffffff;
    box-shadow: 0 0 0 1px rgba(21,62,88,0.28);
}
.coverage-ruler-pin.slice .dot {
    background: #c48b10;
}
.coverage-ruler-pin.volume .dot {
    background: #1f7aa3;
}
.coverage-ruler-pin .val {
    font-family: 'Space Mono', monospace;
    font-size: 0.61rem;
    color: #274760;
    white-space: nowrap;
}
.coverage-ruler-scale {
    margin-top: 0.2rem;
    display: flex;
    justify-content: space-between;
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: #5b758b;
}
.coverage-ruler-legend {
    margin-top: 0.3rem;
    display: flex;
    justify-content: space-between;
    gap: 0.35rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.63rem;
    color: #4f687f;
}
.risk-wrap {
    margin-top: 0.6rem;
}
.risk-scale {
    position: relative;
    height: 8px;
    width: 100%;
    max-width: 440px;
    border-radius: 999px;
    background: linear-gradient(90deg, #2a8a54, #c48b10, #b43b32);
}
.risk-marker {
    position: absolute;
    top: -4px;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #ffffff;
    border: 2px solid #0f5179;
    transform: translateX(-50%);
    box-shadow: 0 0 8px rgba(15,81,121,0.25);
}
.risk-labels {
    width: 100%;
    max-width: 440px;
    display: flex;
    justify-content: space-between;
    margin-top: 0.22rem;
    color: #4b647b;
    font-family: 'Space Mono', monospace;
    font-size: 0.74rem;
}
.sev-scale {
    margin-top: 0.35rem;
    width: 100%;
    max-width: 440px;
    border: 1px solid rgba(165,191,210,0.82);
    border-radius: 999px;
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    overflow: hidden;
    position: relative;
    background: #f4f9fc;
}
.sev-scale span {
    text-align: center;
    padding: 0.34rem 0.2rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #4d667d;
    border-right: 1px solid rgba(165,191,210,0.6);
}
.sev-scale span:last-child { border-right: 0; }
.sev-scale span.active {
    background: var(--sev-color, rgba(42,138,84,0.25));
    color: #254c3a;
    font-weight: 700;
}
.summary-box {
    margin-top: 1rem;
    border: 1px solid rgba(164,190,208,0.9);
    border-radius: 12px;
    background: linear-gradient(180deg, rgba(255,255,255,0.97), rgba(243,249,253,0.95));
    padding: 0.68rem 0.8rem;
}
.summary-box .k {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #567086;
}
.summary-box .v {
    margin-top: 0.16rem;
    color: #2c4b62;
    font-size: 1rem;
    line-height: 1.5;
}

.seg-error-value {
    margin-top: 0.12rem;
    color: #1a4260;
    font-family: 'Fraunces', serif;
    font-size: 1.12rem;
}
.xai-tag-row {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
    margin: 0.35rem 0 0.6rem;
}
.xai-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    border-radius: 999px;
    border: 1px solid rgba(140,176,200,0.78);
    background: #eaf4fb;
    color: #1d4f73;
    padding: 0.2rem 0.55rem;
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
}

.legend-grid {
    display: grid;
    grid-template-columns: 1.4fr 1fr;
    gap: 0.9rem;
    align-items: start;
}
.legend-list { margin-top: 0.2rem; }
.legend-item {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    margin-bottom: 0.35rem;
    font-size: 0.82rem;
    color: #375067;
}
.legend-dot { width: 10px; height: 10px; border-radius: 3px; }

.ct-focus-card {
    border: 1px solid rgba(176, 196, 211, 0.9);
    border-radius: 14px;
    background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(245,250,253,0.95));
    padding: 0.75rem 0.8rem;
}
.ct-focus-card .k {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: #5d7489;
}
.ct-focus-card .v {
    margin-top: 0.22rem;
    color: #1f3f58;
    font-size: 0.86rem;
    line-height: 1.55;
}
.ct-focus-list {
    margin: 0.55rem 0 0;
    padding-left: 1rem;
}
.ct-focus-list li {
    margin: 0.24rem 0;
    color: #375067;
    font-size: 0.8rem;
}

.clinical-card {
    border: 1px solid rgba(175,194,208,0.95);
    border-radius: 16px;
    background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(245,250,253,0.96));
    padding: 1rem 1.05rem;
    box-shadow: 0 12px 24px rgba(15,34,52,0.06);
    margin-top: 0.85rem;
}
.clinical-card ul { margin: 0.35rem 0 0; padding-left: 1.1rem; }
.clinical-card li { color: #3f566b; font-size: 0.86rem; margin: 0.25rem 0; }

[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(7,25,38,0.96), rgba(13,43,64,0.96)),
        linear-gradient(180deg, #eef4f8, #e7eef5) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stCaption { color: rgba(230,240,248,0.74) !important; }
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] li { color: #edf5fb !important; }

.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: 0.95rem 1.2rem;
    margin-bottom: 1rem;
    border: 1px solid rgba(13,94,141,0.16);
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(8,27,41,0.98), rgba(12,57,84,0.94));
    box-shadow: 0 16px 32px rgba(7,25,38,0.18);
}
.topbar-brand {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    min-width: 0;
}
.brand-mark {
    width: 2.6rem;
    height: 2.6rem;
    border-radius: 14px;
    display: grid;
    place-items: center;
    background: linear-gradient(135deg, #eaf5ff, #a8d3ef);
    color: #09324d;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.08em;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.35);
}
.topbar-kicker {
    font-family: 'Space Mono', monospace;
    font-size: 0.63rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(203,231,247,0.8);
    margin-bottom: 0.15rem;
}
.topbar-title {
    color: #f5fbff;
    font-family: 'Fraunces', serif;
    font-size: 1.05rem;
    line-height: 1.1;
}
.topbar-meta {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
    justify-content: flex-end;
}
.topbar-chip,
.topbar-chip-muted {
    display: inline-flex;
    align-items: center;
    padding: 0.38rem 0.75rem;
    border-radius: 999px;
    font-family: 'Space Mono', monospace;
    font-size: 0.66rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border: 1px solid rgba(185,216,234,0.24);
}
.topbar-chip {
    background: rgba(235,247,255,0.12);
    color: #eff8fe;
}
.topbar-chip-muted {
    background: rgba(235,247,255,0.08);
    color: rgba(230,241,249,0.78);
}

.dashboard-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.85rem;
    margin: 0 0 1rem;
}
.dash-stat {
    border: 1px solid rgba(177,196,210,0.85);
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(245,250,253,0.92));
    box-shadow: 0 12px 24px rgba(15,34,52,0.06);
    padding: 0.95rem 1rem;
}
.dash-stat .k {
    font-family: 'Space Mono', monospace;
    color: #53708a;
    font-size: 0.66rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}
.dash-stat .v {
    font-family: 'Fraunces', serif;
    color: #15324a;
    font-size: 1.06rem;
    margin-top: 0.35rem;
}
.dash-stat .m {
    color: var(--muted);
    font-size: 0.82rem;
    line-height: 1.45;
    margin-top: 0.35rem;
}

[data-testid="stFileUploader"] {
    border-radius: 14px !important;
    border: 1.5px dashed #8aa8bf !important;
    background: var(--surface-soft) !important;
}

[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 20px rgba(15,34,52,0.05);
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-family: 'Space Mono', monospace !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-size: 0.68rem !important;
}
[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: 'Fraunces', serif !important;
    font-size: 1.52rem !important;
}

.stButton > button {
    border-radius: 12px !important;
    border: 1px solid #91acc2 !important;
    background: linear-gradient(135deg, #ffffff, #eef5fa) !important;
    color: var(--accent) !important;
    font-family: 'Manrope', sans-serif !important;
    font-weight: 700 !important;
    transition: all 0.22s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    border-color: #5f86a7 !important;
    box-shadow: 0 10px 24px rgba(13,94,141,0.14) !important;
}

/* Sidebar buttons need stronger contrast on dark background */
[data-testid="stSidebar"] .stButton > button {
    border: 1px solid rgba(147, 193, 232, 0.42) !important;
    background: linear-gradient(135deg, #1d4f78, #2b6f9e) !important;
    color: #f5fbff !important;
    box-shadow: 0 8px 18px rgba(8, 28, 47, 0.28) !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: rgba(184, 221, 248, 0.75) !important;
    background: linear-gradient(135deg, #255e8d, #377db0) !important;
    color: #ffffff !important;
    box-shadow: 0 10px 22px rgba(8, 28, 47, 0.34) !important;
}
[data-testid="stSidebar"] .stButton > button:disabled,
[data-testid="stSidebar"] .stButton > button[disabled] {
    background: linear-gradient(135deg, #2a3f50, #304a5f) !important;
    color: rgba(231, 242, 252, 0.68) !important;
    border-color: rgba(154, 178, 197, 0.28) !important;
    box-shadow: none !important;
    opacity: 1 !important;
}

[data-testid="stNotification"] {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
}

.stSuccess {
    background: rgba(31,122,71,0.10) !important;
    border: 1px solid rgba(31,122,71,0.27) !important;
    color: #165b35 !important;
}
.stError {
    background: rgba(180,59,50,0.10) !important;
    border: 1px solid rgba(180,59,50,0.28) !important;
}

.stCode,[data-testid="stCodeBlock"] {
    background: #f6fbff !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

hr { border-color: var(--border) !important; margin: 1rem 0 !important; }
.stCaption {
    color: var(--muted) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.76rem !important;
}

[data-testid="stImage"] img,.element-container img {
    border-radius: 14px !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 10px 24px rgba(15,34,52,0.09);
}

/* Segmentation / Grad-CAM overlay image sizing */
.visual-stage img {
    max-height: 220px;
    width: 100%;
    height: auto;
    object-fit: contain;
}

::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #e9f0f6; }
::-webkit-scrollbar-thumb { background: #adc1d3; border-radius: 8px; }

/* custom components */
.hero-wrap,
.scan-card,
.result-panel,
.pat-summary-card,
.pat-section,
.report-header {
    animation: riseFade 0.45s ease-out;
}

.hero-wrap {
    position: relative;
    border: 1px solid var(--border-strong);
    border-radius: 24px;
    padding: 2.2rem 2.35rem 1.9rem;
    background: linear-gradient(135deg, #ffffff, #f1f7fb);
    box-shadow: 0 18px 36px rgba(15,34,52,0.10);
    margin-bottom: 1.4rem;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(circle 230px at 7% 50%, rgba(13,94,141,0.13), transparent),
        radial-gradient(circle 190px at 92% 18%, rgba(19,111,115,0.13), transparent);
    pointer-events: none;
}

.hero-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 0.55rem;
    opacity: 0.9;
}
.hero-title {
    font-family: 'Inter', sans-serif;
    font-size: clamp(1.8rem, 3vw, 2.7rem);
    line-height: 1.16;
    color: #10283d;
    margin: 0 0 0.52rem;
}
.hero-title em { font-style: normal; }
.hero-sub {
    color: var(--muted);
    font-size: 0.98rem;
    max-width: 680px;
    line-height: 1.62;
    margin: 0 0 1.05rem;
}

.hero-layout {
    display: grid;
    grid-template-columns: 1.18fr 0.82fr;
    gap: 1rem;
    align-items: stretch;
}

.hero-media {
    border: 0;
    border-radius: 0;
    padding: 0;
    background: transparent;
    box-shadow: none;
    align-self: flex-start;
    max-width: none;
    width: fit-content;
    margin-left: auto;
}
.hero-media-head {
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #4f6f87;
    margin-bottom: 0.24rem;
}
.hero-media-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.55rem;
}
.hero-collage-frame {
    position: relative;
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.2rem;
    border: 1px solid rgba(166, 191, 209, 0.38);
    border-radius: 10px;
    overflow: hidden;
    background: #eef5fa;
    padding: 0.1rem;
    width: 100%;
    max-width: 190px;
    box-shadow: none;
}
.hero-collage-tile {
    position: relative;
    border-radius: 12px;
    overflow: hidden;
    background: #dfeaf2;
    border: 1px solid rgba(166,191,209,0.72);
    aspect-ratio: 1 / 1;
    min-height: 0;
}
.hero-collage-img {
    display: block;
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform 220ms ease, filter 220ms ease;
}
.hero-collage-cap {
    position: absolute;
    top: 6px;
    left: 6px;
    z-index: 1;
    padding: 0.12rem 0.34rem;
    border-radius: 999px;
    border: 1px solid rgba(186, 209, 226, 0.42);
    background: rgba(6, 20, 33, 0.56);
    backdrop-filter: blur(1.5px);
    font-family: 'DM Mono', monospace;
    font-size: 0.5rem;
    line-height: 1.25;
    letter-spacing: 0.03em;
    color: rgba(239, 247, 253, 0.94);
    pointer-events: none;
}
.hero-collage-tile:hover .hero-collage-img {
    transform: scale(1.035);
    filter: contrast(1.04);
}
.hero-collage-note {
    margin-top: 0.36rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: #58758b;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.hero-scan-card {
    position: relative;
    border: 1px solid rgba(165, 189, 208, 0.7);
    border-radius: 12px;
    background:
        radial-gradient(circle at 35% 45%, rgba(157, 171, 184, 0.75) 0 24%, transparent 25%),
        radial-gradient(circle at 35% 45%, rgba(121, 140, 156, 0.33) 0 42%, transparent 43%),
        linear-gradient(165deg, #111a23, #0a1018);
    aspect-ratio: 1 / 1;
    min-height: 0;
    overflow: hidden;
}
.hero-scan-card::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(180deg, rgba(255,255,255,0.1), transparent 35%);
    pointer-events: none;
}
.hero-scan-card.seg::before {
    content: '';
    position: absolute;
    left: 34%;
    top: 36%;
    width: 30px;
    height: 22px;
    border-radius: 58% 42% 55% 45% / 46% 58% 42% 54%;
    background: rgba(255, 70, 70, 0.78);
    border: 1px solid rgba(255, 166, 166, 0.72);
    box-shadow: 0 0 18px rgba(255, 60, 60, 0.28);
}
.hero-scan-label {
    position: absolute;
    left: 7px;
    top: 6px;
    border: 1px solid rgba(186, 209, 226, 0.38);
    border-radius: 999px;
    padding: 0.12rem 0.44rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.05em;
    color: rgba(239, 247, 253, 0.9);
    background: rgba(12, 33, 50, 0.52);
}

.pill-row { display: flex; gap: 0.52rem; flex-wrap: wrap; }
.pill {
    display: inline-flex;
    align-items: center;
    gap: 0.36rem;
    padding: 0.28rem 0.78rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-family: 'Manrope', sans-serif;
    font-weight: 700;
    letter-spacing: 0.02em;
    border: 1px solid;
}
.pill-blue   { border-color: #9fc2dd; background: #e8f3fb; color: #1e537a; }
.pill-purple { border-color: #9bc0c8; background: #e9f6f5; color: #1f6466; }
.pill-green  { border-color: #9dc9b2; background: #edf7f1; color: #1f6d43; }
.pill-amber  { border-color: #ddc79c; background: #faf3e5; color: #6c5317; }
.pill-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; opacity: 0.72; }

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.17em;
    text-transform: uppercase;
    color: #2f607f;
    margin-bottom: 0.34rem;
}

.scan-card {
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.2rem 1.4rem 0.95rem;
    background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(249,252,254,0.96));
    margin-bottom: 1rem;
    box-shadow: 0 10px 24px rgba(15,34,52,0.06);
}
.scan-card h3 { font-size: 1.05rem; margin: 0 0 0.22rem; color: #17324b; }
.scan-card p { color: var(--muted); font-size: 0.885rem; margin: 0; line-height: 1.5; }

.result-panel {
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.32rem 1.52rem;
    background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(248,251,253,0.9));
    box-shadow: 0 16px 36px rgba(15,34,52,0.10);
    backdrop-filter: blur(14px);
    margin-bottom: 1rem;
}
.result-panel h3 {
    font-size: 1rem;
    margin: 0 0 0.9rem;
    color: #173750;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.64rem;
}

.visual-stage {
    border: 1px solid rgba(178,196,210,0.95);
    border-radius: 22px;
    background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(244,249,252,0.92));
    box-shadow: 0 18px 40px rgba(15,34,52,0.09);
    padding: 1.15rem 1.2rem 1.05rem;
    margin: 1rem 0 0.85rem;
}
.visual-stage h3 {
    margin: 0 0 0.25rem;
    color: #123149;
    font-size: 1.05rem;
}
.visual-stage .subtle {
    color: var(--muted);
    font-size: 0.86rem;
    line-height: 1.55;
    margin-bottom: 0.9rem;
}

.result-grid {
    display: grid;
    grid-template-columns: 1.15fr 0.85fr;
    gap: 0.9rem;
    margin-top: 1rem;
}
.result-grid.single {
    grid-template-columns: 1fr;
}
.result-card {
    border: 1px solid rgba(178,196,210,0.9);
    border-radius: 18px;
    padding: 1rem 1.05rem;
    background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(247,251,253,0.92));
    box-shadow: 0 12px 26px rgba(15,34,52,0.06);
}
.result-card h3 {
    font-size: 0.98rem;
    margin: 0 0 0.75rem;
    color: #173750;
    border: 0;
    padding: 0;
}
.metric-row {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.75rem;
    margin: 0.95rem 0 1.1rem;
}
.metric-tile {
    border: 1px solid rgba(178,196,210,0.8);
    border-radius: 16px;
    background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(244,248,251,0.95));
    padding: 0.85rem 0.9rem;
    box-shadow: 0 10px 22px rgba(15,34,52,0.05);
}
.metric-tile .k {
    font-family: 'Space Mono', monospace;
    font-size: 0.63rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #5c768f;
}
.metric-tile .v {
    margin-top: 0.28rem;
    font-family: 'Fraunces', serif;
    font-size: 1.05rem;
    color: #16344d;
}
.metric-tile .m {
    margin-top: 0.2rem;
    color: var(--muted);
    font-size: 0.8rem;
    line-height: 1.45;
}


.sev-bar-wrap {
    height: 6px;
    border-radius: 999px;
    background: linear-gradient(90deg, #1f7a47, #c48b10, #b43b32);
    margin: 0.4rem 0 0.2rem;
    position: relative;
}
.sev-marker {
    position: absolute;
    top: -4px;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #ffffff;
    border: 2px solid var(--accent);
    box-shadow: 0 0 9px rgba(13,94,141,0.3);
    transform: translateX(-50%);
}

.conf-bar-outer { height: 8px; border-radius: 999px; background: #e3edf6; overflow: hidden; margin-top: 0.35rem; }
.conf-bar-inner { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #0d5e8d, #136f73); }

.ct-bullet-wrap { margin-top: 0.42rem; }
.ct-bullet-track {
    position: relative;
    height: 10px;
    border-radius: 999px;
    border: 1px solid rgba(155,180,199,0.86);
    overflow: hidden;
    background:
        linear-gradient(90deg,
            rgba(31,122,71,0.22) 0%, rgba(31,122,71,0.22) 33.3%,
            rgba(196,139,16,0.22) 33.3%, rgba(196,139,16,0.22) 66.6%,
            rgba(180,59,50,0.22) 66.6%, rgba(180,59,50,0.22) 100%);
}
.ct-bullet-marker {
    position: absolute;
    top: -3px;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #ffffff;
    border: 2px solid #174a6e;
    box-shadow: 0 0 0 2px rgba(255,255,255,0.72), 0 1px 6px rgba(14,46,70,0.28);
    transform: translateX(-50%);
}
.ct-bullet-scale {
    display: flex;
    justify-content: space-between;
    margin-top: 0.26rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.61rem;
    letter-spacing: 0.03em;
    color: #5b748b;
}

.ct-sev-variant {
    margin-top: 0.42rem;
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.32rem;
}
.ct-sev-chip {
    border: 1px solid rgba(164, 188, 207, 0.86);
    border-radius: 10px;
    padding: 0.28rem 0.26rem;
    text-align: center;
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.02em;
    color: #5d7489;
    background: #f2f7fb;
}
.ct-sev-chip.active-low {
    border-color: rgba(38, 138, 88, 0.55);
    background: rgba(38, 138, 88, 0.16);
    color: #1f6d45;
    font-weight: 700;
}
.ct-sev-chip.active-mod {
    border-color: rgba(196, 139, 16, 0.56);
    background: rgba(196, 139, 16, 0.15);
    color: #7b5a08;
    font-weight: 700;
}
.ct-sev-chip.active-high {
    border-color: rgba(180, 59, 50, 0.56);
    background: rgba(180, 59, 50, 0.15);
    color: #8f3028;
    font-weight: 700;
}

.pat-sev-badge { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.45rem 1rem; border-radius: 999px; font-weight: 700; font-size: 0.9rem; margin: 0.5rem 0 1rem; }
.ct-sev-neutral { background: rgba(125,145,162,0.13); border: 1px solid rgba(125,145,162,0.34); color: #234158; }
.severity-mild { background: rgba(31,122,71,0.12); border: 1px solid rgba(31,122,71,0.28); color: #195e37; }
.severity-moderate { background: rgba(203,147,17,0.12); border: 1px solid rgba(203,147,17,0.28); color: #7a5904; }
.severity-severe { background: rgba(180,59,50,0.12); border: 1px solid rgba(180,59,50,0.26); color: #8f2f27; }
.sev-mild { background: rgba(31,122,71,0.12); border: 1px solid rgba(31,122,71,0.25); color: #195e37; }
.sev-mod { background: rgba(138,100,0,0.12); border: 1px solid rgba(138,100,0,0.25); color: #664909; }
.sev-severe { background: rgba(180,59,50,0.12); border: 1px solid rgba(180,59,50,0.26); color: #8f2f27; }

.report-header {
    border: 1px solid rgba(44,146,109,0.22);
    border-radius: 18px;
    padding: 1.12rem 1.42rem 0.84rem;
    background: linear-gradient(135deg, rgba(234,249,241,0.9), rgba(255,255,255,0.98));
    margin-bottom: 0.8rem;
}
.report-header h3 { font-size: 1.05rem; margin: 0 0 0.22rem; color: #21553d; }
.report-header p { color: var(--muted); font-size: 0.86rem; margin: 0; }

.step-list { counter-reset: steps; list-style: none; padding: 0; margin: 0.6rem 0 0; }
.step-list li {
    counter-increment: steps;
    display: flex;
    align-items: center;
    gap: 0.65rem;
    padding: 0.46rem 0;
    color: var(--muted);
    font-size: 0.875rem;
    border-bottom: 1px solid #dde7f0;
}
.step-list li::before {
    content: counter(steps);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: #e4f0f8;
    border: 1px solid #b8cfdf;
    color: #1e567f;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    flex-shrink: 0;
}

/* patient page */
.pat-header {
    text-align: center;
    padding: 2.45rem 2rem 1.8rem;
    border: 1px solid rgba(159,190,212,0.95);
    border-radius: 22px;
    background:
        radial-gradient(circle 230px at 82% 14%, rgba(13,94,141,0.12), transparent),
        linear-gradient(135deg, #f7fbff, #eef6fb);
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
}
.pat-header::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(circle 220px at 78% 18%, rgba(13,94,141,0.09), transparent);
    pointer-events: none;
}
.pat-header .pat-icon { font-size: 2.4rem; margin-bottom: 0.6rem; }
.pat-header h1 { font-size: clamp(1.45rem, 2.5vw, 2.05rem); margin: 0 0 0.42rem; color: #16334c; }
.pat-header p { color: var(--muted); font-size: 0.93rem; margin: 0; }

.pat-section {
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    background: var(--surface);
    margin-bottom: 1rem;
    box-shadow: 0 8px 18px rgba(15,34,52,0.06);
}
.pat-section-icon { font-size: 1.3rem; margin-bottom: 0.4rem; }
.pat-section h3 { font-family: 'Inter', sans-serif; font-size: 1rem; margin: 0 0 0.55rem; color: #17324b; }
.pat-section p,.pat-section li { color: #394f66; font-size: 0.91rem; line-height: 1.66; margin: 0.25rem 0; }
.pat-section ul { padding-left: 1.2rem; }

.pat-summary-card {
    border: 1px solid #bdd2e1;
    border-radius: 18px;
    padding: 1rem 1.1rem;
    background:
        radial-gradient(circle 180px at 90% 10%, rgba(13,94,141,0.09), transparent),
        linear-gradient(132deg, #eaf4fd, #ffffff);
    margin-bottom: 1rem;
}
.pat-summary-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(170px,1fr)); gap: 0.65rem; margin-top: 0.7rem; }
.pat-chip {
    border: 1px solid #d2dfeb;
    border-radius: 12px;
    padding: 0.62rem 0.72rem;
    background: #f8fbff;
}
.pat-chip .k {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #5f768f;
}
.pat-chip .v { margin-top: 0.2rem; font-weight: 700; color: #17324b; font-size: 0.93rem; }

/* premium lock */
.premium-blur-wrap { position: relative; border-radius: 14px; overflow: hidden; }
.premium-blur-content { filter: blur(6px) brightness(0.56); pointer-events: none; user-select: none; }
.premium-overlay {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.8rem;
    background: rgba(243,248,252,0.74);
    backdrop-filter: blur(2px);
    border-radius: 14px;
    z-index: 10;
}
.premium-overlay .lock-icon { font-size: 2rem; }
.premium-overlay h4 { font-family: 'Fraunces', serif; color: #15314b; font-size: 1.1rem; margin: 0; text-align: center; }
.premium-overlay p { color: var(--muted); font-size: 0.82rem; margin: 0; text-align: center; max-width: 260px; line-height: 1.5; }
.premium-badge {
    background: linear-gradient(135deg, #d8b96d, #c5844f);
    color: #1c2f41;
    font-weight: 700;
    font-size: 0.78rem;
    letter-spacing: 0.05em;
    padding: 0.3rem 0.85rem;
    border-radius: 999px;
    font-family: 'Space Mono', monospace;
}

.sidebar-premium-box {
    background: linear-gradient(135deg, #edf5ff, #dcecff);
    border: 1px solid #9fc0e6;
    border-radius: 12px;
    padding: 0.85rem 1rem;
    margin-top: 0.4rem;
}
.sidebar-premium-tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.63rem;
    letter-spacing: 0.15em;
    color: #2b5a8c !important;
    opacity: 0.95;
    margin-bottom: 0.4rem;
    font-weight: 700;
}
.sidebar-premium-title {
    font-family: 'Fraunces', serif;
    color: #143e69 !important;
    font-size: 0.95rem;
    margin-bottom: 0.3rem;
    font-weight: 700;
}
.sidebar-premium-copy {
    color: #1f4d78 !important;
    font-size: 0.8rem;
    line-height: 1.5;
}
.sidebar-premium-price {
    margin-top: 0.7rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #0f3f6e !important;
    font-weight: 700;
}

/* stronger selector to beat global sidebar div color override */
[data-testid="stSidebar"] .sidebar-premium-box .sidebar-premium-tag { color: #2b5a8c !important; }
[data-testid="stSidebar"] .sidebar-premium-box .sidebar-premium-title { color: #143e69 !important; }
[data-testid="stSidebar"] .sidebar-premium-box .sidebar-premium-copy { color: #1f4d78 !important; }
[data-testid="stSidebar"] .sidebar-premium-box .sidebar-premium-price { color: #0f3f6e !important; }

.pat-disclaimer {
    margin-top: 1.5rem;
    padding: 1rem 1.2rem;
    border: 1px solid #d4e0ea;
    border-radius: 12px;
    background: linear-gradient(180deg, #f8fbfd, #ffffff);
}
.stDownloadButton > button {
    border-radius: 12px !important;
    border: 1px solid rgba(13,94,141,0.28) !important;
    background: linear-gradient(135deg, #0d5e8d, #136f73) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    box-shadow: 0 14px 28px rgba(13,94,141,0.18) !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 16px 32px rgba(13,94,141,0.24) !important;
}

.upload-note {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.65rem;
    margin: 0.1rem 0 1rem;
}
.upload-note-box {
    border: 1px solid rgba(175,194,208,0.9);
    border-radius: 14px;
    background: rgba(255,255,255,0.82);
    padding: 0.8rem 0.85rem;
}
.upload-note-box .k {
    font-family: 'Space Mono', monospace;
    color: #5d768e;
    font-size: 0.64rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}
.upload-note-box .v {
    color: #17324b;
    font-size: 0.86rem;
    line-height: 1.45;
    margin-top: 0.3rem;
}

.pat-hero-band {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    justify-content: center;
    margin-top: 1rem;
}

.patient-cta {
    display: flex;
    gap: 0.7rem;
    flex-wrap: wrap;
    justify-content: center;
    margin-top: 1rem;
}
.pat-disclaimer p {
    color: #4c647b;
    font-size: 0.76rem;
    font-family: 'Space Mono', monospace;
    line-height: 1.68;
    margin: 0;
}

@media (max-width: 900px) {
    .hero-wrap { padding: 1.5rem 1.2rem 1.2rem; border-radius: 18px; }
    .hero-layout { grid-template-columns: 1fr; }
    .hero-media { max-width: none; margin-left: 0; }
    .hero-collage-frame { aspect-ratio: 4 / 3; max-height: none; }
    .scan-card, .result-panel, .pat-section, .pat-summary-card { padding: 1rem; }
    .pat-header { padding: 1.7rem 1rem 1.2rem; }
    .pill-row { gap: 0.4rem; }
    .topbar { flex-direction: column; align-items: flex-start; }
    .dashboard-grid, .upload-note { grid-template-columns: 1fr; }
    .legend-grid { grid-template-columns: 1fr; }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PATIENT CONTENT DICTIONARY
# ─────────────────────────────────────────────────────────────────────────────
PATIENT_CONTENT = {
    "CT": {
        "what_found": {
            "Normal": "Great news — your brain scan looks normal. No signs of bleeding, unusual growths, or damage were found.",
            "Intracranial Hemorrhage Detected": "The scan shows a small area of bleeding inside the brain, called an intracranial hemorrhage. Blood has leaked from a vessel and is collecting in or around the brain tissue.",
            "Abnormality Detected": "The scan picked up something that looks different from a normal brain. It could be swelling, a structural change, or another finding that a doctor will need to look at closely.",
        },
        "plain_meaning": {
            "Normal": "Your brain appears healthy based on this scan. Share the results with your doctor for full peace of mind.",
            "Intracranial Hemorrhage Detected": "Think of it like a bruise, but inside the brain. Depending on size and location, this may need treatment or monitoring — a doctor needs to see this urgently.",
            "Abnormality Detected": "Something unusual was flagged. This does not automatically mean something serious, but a specialist should look at it as soon as possible.",
        },
        "causes": {
            "Normal": [],
            "Intracranial Hemorrhage Detected": ["High blood pressure (most common)", "A head injury or fall", "Blood-thinning medications", "A weak blood vessel (aneurysm)", "Certain clotting disorders"],
            "Abnormality Detected": ["Inflammation or infection in the brain", "Scarring from a previous injury", "A benign (non-cancerous) growth", "Early signs of a neurological condition", "Vascular changes related to age or blood pressure"],
        },
        "what_to_do": {
            "Normal": ["No urgent action needed — share results with your doctor", "Maintain regular health check-ups", "Report any new headaches or neurological symptoms"],
            "Intracranial Hemorrhage Detected": ["🚨 Go to the Emergency Department right now", "Do NOT drive — call someone or an ambulance", "Tell staff your scan shows a possible brain bleed", "Avoid blood thinners (aspirin, ibuprofen) until a doctor clears you", "Do not eat or drink in case surgery is needed"],
            "Abnormality Detected": ["See a neurologist within the next few days", "Bring this report and scan files to the appointment", "Write down your symptoms — headaches, dizziness, vision changes", "Many abnormalities are benign and very treatable"],
        },
        "lifestyle": {
            "Normal": ["Stay hydrated and get regular sleep", "Exercise regularly for brain health", "Manage stress and blood pressure"],
            "Intracranial Hemorrhage Detected": ["Rest completely until you have seen a doctor", "Avoid all physical strain or exertion", "No alcohol or smoking"],
            "Abnormality Detected": ["Avoid strenuous activity until reviewed by a doctor", "Get enough sleep — brain recovery happens during rest", "Keep a symptom diary for your neurologist"],
        },
    },
    "MRI": {
        "what_found_tumor": "The MRI scan detected an area in the brain that looks different from surrounding tissue. The AI has highlighted this as a possible tumor region — an abnormal growth of cells.",
        "plain_meaning_tumor": "Think of it as cells that grew where they should not. Some brain tumors are slow-growing and non-cancerous (benign), others may need treatment. Only a specialist can tell you which type after reviewing the full scan.",
        "causes_tumor": ["Uncontrolled growth of brain cells (primary brain tumor)", "Cancer that has spread from another part of the body", "Genetic or inherited factors", "Prior radiation exposure (rare)", "Often the cause is unknown — it is not your fault"],
        "what_to_do_tumor": {
            "Mild": ["Book an appointment with a neurologist this week", "Bring all scan files and this report", "Small findings are often monitored rather than treated immediately", "Ask about a follow-up MRI in 3–6 months"],
            "Moderate": ["See a neurologist or neurosurgeon within 48–72 hours", "Bring all scan files to the appointment", "Avoid alcohol, smoking, and unnecessary stress", "Note any new symptoms — headaches, vision changes, weakness", "You may be referred for a biopsy or further imaging"],
            "Severe": ["🚨 Seek specialist care urgently — contact a neurosurgeon today", "Go to Emergency if you have sudden severe headaches, seizures, or weakness", "Do not drive", "A treatment plan will likely be discussed immediately", "Bring a family member to appointments for support"],
        },
        "lifestyle_tumor": ["Rest as much as you need — fatigue is normal", "Eat a balanced diet with anti-inflammatory foods", "Avoid alcohol and smoking completely", "Lean on family and friends — emotional support matters greatly", "Ask your doctor about connecting with a support group"],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS (all logic untouched)
# ─────────────────────────────────────────────────────────────────────────────

# Rotating tips for the main UI
TIP_SETS = {
    0: [  # MRI-focused tips
        ("For MRI", "Include <b>flair</b>, <b>t1ce</b>, and <b>t2</b> from the same patient and study session."),
        ("Ground Truth", "Add a <b>seg</b> mask only when available to compute Dice and IoU against ground truth."),
        ("MRI Quality", "Ensure consistent spacing and alignment across all three modalities for accurate segmentation."),
    ],
    1: [  # CT-focused tips
        ("CT Upload", "Single CT DICOM (.dcm) or multiple DICOM files are automatically loaded and processed."),
        ("Hemorrhage Detection", "The model flags intracranial bleeding patterns; correlate with clinical presentation."),
        ("Scan Window", "Control window/level settings before upload to optimize tissue contrast for the AI model."),
    ],
    2: [  # General tips
        ("Data Quality", "Verify slice quality first: strong motion, truncation, or missing coverage can reduce reliability."),
        ("Batch Analysis", "Process multiple patients in one session; each analysis generates a separate patient report."),
        ("Export Reports", "Download plain-text or PDF reports for each patient for your clinical documentation."),
    ],
}

def save_temp(uploaded):
    suffix = "".join(Path(uploaded.name).suffixes)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        return tmp.name

def infer_mri_modality(filename):
    name = filename.lower()
    if "seg"  in name: return "seg"
    if "flair" in name: return "flair"
    if "t1ce" in name or "t1c" in name or "t1_ce" in name: return "t1ce"
    if "t2"   in name and "t1" not in name: return "t2"
    return None

def dice_score(y_true, y_pred):
    y_true = y_true.astype(np.float32).flatten()
    y_pred = (y_pred > 0.5).astype(np.float32).flatten()
    intersection = np.sum(y_true * y_pred)
    return (2 * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-6)

def iou_score(y_true, y_pred):
    y_true = y_true.astype(np.float32).flatten()
    y_pred = (y_pred > 0.5).astype(np.float32).flatten()
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / (union + 1e-6)

def render_mri_panel(img, pred_mask, prob_map, gt_mask=None):
    # Match the original Kaggle evaluation logic: compute tumor
    # area from the predicted mask and use Dice/IoU against the
    # resized ground-truth mask when available.
    tumor_percent = (np.sum(pred_mask) / pred_mask.size) * 100
    if gt_mask is not None:
        dice = dice_score(gt_mask, pred_mask)
        iou  = iou_score(gt_mask, pred_mask)
        error_map    = np.abs(gt_mask.astype(np.float32) - pred_mask.astype(np.float32))
        gt_title      = "Ground Truth"
        overlay_title = f"Overlay\nDice:{dice:.2f}"
        suptitle      = f"Tumor Area: {tumor_percent:.2f}% | IoU:{iou:.2f}"
    else:
        dice = iou = None
        error_map     = np.zeros_like(pred_mask, dtype=np.float32)
        gt_mask       = np.zeros_like(pred_mask, dtype=np.float32)
        gt_title      = "Ground Truth (N/A)"
        overlay_title = "Overlay"
        suptitle      = f"Tumor Area: {tumor_percent:.2f}%"

    fig, ax = plt.subplots(1, 5, figsize=(13.8, 3), dpi=80, constrained_layout=True)
    fig.patch.set_facecolor('#f8fbff')
    for a in ax: a.set_facecolor('#f8fbff')

    ax[0].imshow(img[:, :, 0], cmap="gray");        ax[0].set_title("MRI Slice",            color="#5c7087", fontsize=9); ax[0].axis("off")
    ax[1].imshow(gt_mask, cmap="gray");              ax[1].set_title(gt_title,               color="#5c7087", fontsize=9); ax[1].axis("off")
    ax[2].imshow(pred_mask, cmap="gray");            ax[2].set_title("Predicted Mask",        color="#5c7087", fontsize=9); ax[2].axis("off")
    ax[3].imshow(img[:, :, 0], cmap="gray"); ax[3].imshow(pred_mask, cmap="winter", alpha=0.55)
    ax[3].set_title(overlay_title, color="#5c7087", fontsize=9); ax[3].axis("off")
    ax[4].imshow(error_map, cmap="magma");           ax[4].set_title("Segmentation Error",   color="#5c7087", fontsize=9); ax[4].axis("off")

    fig.suptitle(suptitle, color="#1a3a58", fontsize=10)
    return fig, tumor_percent, dice, iou

def sev_cls(sev):
    s = (sev or "").lower()
    if "mild" in s:   return "sev-mild"
    if "severe" in s: return "sev-severe"
    return "sev-mod"

def sev_pct(sev):
    s = (sev or "").lower()
    if "mild" in s:   return 15
    if "severe" in s: return 85
    return 50

def sev_emoji(sev):
    s = (sev or "").lower()
    if "mild" in s:   return "🟢"
    if "severe" in s: return "🔴"
    return "🟡"


def _ct_severity_variant_html(severity):
    s = (severity or "").lower()
    low_cls = "active-low" if "mild" in s else ""
    high_cls = "active-high" if "severe" in s else ""
    mod_cls = "active-mod" if not low_cls and not high_cls else ""
    return (
        f'<div class="ct-sev-variant">'
        f'<span class="ct-sev-chip {low_cls}">Low</span>'
        f'<span class="ct-sev-chip {mod_cls}">Moderate</span>'
        f'<span class="ct-sev-chip {high_cls}">High</span>'
        f'</div>'
    )


def _show_mri_modality_card():
    st.markdown("""
<div class="modality-card mri">
    <div class="section-label">Detected Modality</div>
    <span class="modality-pill mri">🧬 MRI Scan</span>
    <div class="modality-note">Sequences loaded: FLAIR + T1CE + T2</div>
</div>
""", unsafe_allow_html=True)


def _show_ct_visual_stage(overlay, label, severity, prob):
    ct_prob_pct = f"{(prob or 0.0) * 100:.1f}%"
    st.markdown("""
<div class="visual-stage">
    <h3>Grad-CAM Visualisation</h3>
    <p class="subtle">Highlighted regions indicate areas influencing the model's prediction.</p>
</div>
""", unsafe_allow_html=True)
    st.markdown("""
<div class="xai-tag-row">
    <span class="xai-tag">🎯 Model focused here</span>
    <span class="xai-tag">🧠 Important region</span>
</div>
""", unsafe_allow_html=True)

    img_col, info_col = st.columns([0.66, 1.34], gap="small")
    with img_col:
        st.image(overlay, caption="Regions most influential to the model's prediction", width=340)
    with info_col:
        st.markdown(f"""
<div class="ct-focus-card">
    <div class="k">CT Focus Summary</div>
    <div class="v"><strong>Finding:</strong> {label}<br><strong>Severity:</strong> {severity or 'N/A'}<br><strong>Confidence:</strong> {ct_prob_pct}</div>
    <ul class="ct-focus-list">
        <li>Brighter regions indicate stronger influence on the AI decision.</li>
        <li>Use this as support context, not as a standalone diagnosis.</li>
        <li>Correlate with clinical signs and radiologist review.</li>
    </ul>
</div>
""", unsafe_allow_html=True)
        st.markdown("""
<div class="ct-focus-card" style="margin-top:0.6rem;">
    <div class="k">Heatmap Legend</div>
    <div class="legend-list" style="margin-top:0.35rem;">
        <div class="legend-item"><span class="legend-dot" style="background:#ff4a4a;"></span>Red → high importance / tumor</div>
        <div class="legend-item"><span class="legend-dot" style="background:#f1c232;"></span>Yellow → medium</div>
        <div class="legend-item"><span class="legend-dot" style="background:#8fb7dd;"></span>Blue / transparent → low</div>
    </div>
    <div class="v" style="margin-top:0.2rem;">Severity bar shows risk progression from low to high.</div>
</div>
""", unsafe_allow_html=True)


def _severity_bucket_for_ui(severity):
    s = (severity or "").lower()
    if any(k in s for k in ["normal", "no tumor", "mild", "small", "low"]):
        return "Mild", 16
    if any(k in s for k in ["moderate", "medium"]):
        return "Moderate", 50
    return "Severe", 84


def _mri_severity_from_coverage(tp):
    # Unified MRI thresholds used across inference + UI:
    # <1% no tumor, 1-3.5% low, 3.5-10% moderate, >=10% high.
    if tp < 1.0:
        return "No Tumor"
    if tp < 3.5:
        return "Small Tumor"
    if tp < 10.0:
        return "Medium Tumor"
    return "Large Tumor"


def _mri_overall_coverage(seg_obj, is_nib_flair):
    if seg_obj is None:
        return None
    if is_nib_flair:
        seg_volume = np.asarray(seg_obj.dataobj)
    else:
        seg_volume = np.asarray(seg_obj)
    if seg_volume.size == 0:
        return None
    tumor_voxels = float(np.sum(seg_volume > 0))
    return (tumor_voxels / float(seg_volume.size)) * 100.0


def _risk_score_for_ui(modality, confidence=None, tumor_pct=None, severity=""):
    if modality == "CT" and confidence is not None:
        return max(1.0, min(99.0, float(confidence) * 100.0))
    if tumor_pct is not None:
        tp = float(tumor_pct)
        if tp < 1.0:
            return 16.0
        if tp < 3.5:
            return 22.0 + ((tp - 1.0) / 2.5) * 10.0
        if tp < 10.0:
            return 35.0 + ((tp - 3.5) / 6.5) * 30.0
        return min(95.0, 70.0 + ((tp - 10.0) / 20.0) * 20.0)
    bucket, _ = _severity_bucket_for_ui(severity)
    return {"Mild": 24.0, "Moderate": 56.0, "Severe": 84.0}[bucket]


def _show_decision_snapshot(modality, disease, severity, confidence=None, tumor_pct=None, overall_tumor_pct=None, dice=None, iou=None):
    mode_cls = "ct" if modality == "CT" else "mri"
    sev_bucket, _ = _severity_bucket_for_ui(severity)
    risk_pct = _risk_score_for_ui(modality, confidence, tumor_pct, severity)

    sev_colors = {
        "Mild": ("#2a8a54", "rgba(42,138,84,0.22)", "33.4%"),
        "Moderate": ("#c48b10", "rgba(196,139,16,0.24)", "66.7%"),
        "Severe": ("#b43b32", "rgba(180,59,50,0.24)", "100%"),
    }
    ring_color, sev_fill_color, _ = sev_colors.get(sev_bucket, sev_colors["Moderate"])
    risk_tier = "Low" if risk_pct < 35 else ("Moderate" if risk_pct < 70 else "High")
    follow_up_window = "Within 3-7 days" if sev_bucket == "Mild" else ("Within 48-72 hours" if sev_bucket == "Moderate" else "Immediate review today")
    follow_up_reason = "Severity-based recommendation"
    sev_active = {
        "Mild": ("active", "", ""),
        "Moderate": ("", "active", ""),
        "Severe": ("", "", "active"),
    }.get(sev_bucket, ("", "active", ""))
    ring_subtext = ""
    coverage_visual_html = ""

    if modality == "CT" and confidence is not None:
        metric_label = "Confidence"
        confidence_text = f"{confidence:.2f} ({confidence*100:.0f}%)"
        conf_fill = max(0.0, min(100.0, float(confidence) * 100.0))
        conf_bar_html = f'<div class="mini-conf"><span style="width:{conf_fill:.1f}%;"></span></div>'
        ring_label = "Confidence"
        ring_value = f"{conf_fill:.0f}%"
        ring_pct = conf_fill
        ring_subtext = "Model probability"
        coverage_visual_html = (
            f'<div class="decision-ring" style="--p:{ring_pct:.1f};">'
            f'<div class="decision-ring-label">{ring_label}<b>{ring_value}</b></div>'
            f'</div>'
            f'<div class="decision-ring-sub">{ring_subtext}</div>'
        )
        quality_text = "Classification task"
        validation_text = "Probability-driven"
        summary_line = f"Detected likely {str(disease).lower()} pattern with {sev_bucket.lower()} severity signal."
        strip_html = (
            f'<div class="decision-chip"><div class="k">Risk Tier</div><div class="v">{risk_tier}</div></div>'
            f'<div class="decision-chip"><div class="k">Severity</div><div class="v">{severity}</div></div>'
            f'<div class="decision-chip"><div class="k">Confidence</div><div class="v">{conf_fill:.0f}%</div></div>'
        )
    else:
        metric_label = "Slice Coverage"
        coverage = float(tumor_pct or 0.0)
        remaining_pct = max(0.0, 100.0 - coverage)
        overall_pct_num = float(overall_tumor_pct) if overall_tumor_pct is not None else 0.0
        overall_pct_num = max(0.0, min(100.0, overall_pct_num))
        overall_pct_text = f"{overall_pct_num:.1f}%" if overall_tumor_pct is not None else "N/A"
        confidence_text = f"{coverage:.1f}% of analyzed MRI slice"
        conf_fill = max(8.0, min(100.0, coverage))
        conf_bar_html = ""
        ring_label = "Tumor Coverage"
        ring_value = f"{coverage:.1f}%"
        ring_pct = max(12.0, min(100.0, coverage))
        ring_subtext = f"Remaining normal tissue {remaining_pct:.1f}%"
        coverage_visual_html = (
            f'<div class="coverage-viz">'
            f'<div class="k">Clinical Composition</div>'
            f'<div class="v">Tumor share in analyzed slice</div>'
            f'<div class="coverage-pie" style="--tumor-pct:{coverage:.1f};">'
            f'<div class="coverage-pie-label">Tumor<b>{coverage:.1f}%</b></div>'
            f'</div>'
            f'<div class="coverage-pie-legend"><span>Tumor (red): {coverage:.1f}%</span><span>Normal tissue: {remaining_pct:.1f}%</span></div>'
            f'<div class="coverage-metrics">'
            f'<div class="coverage-metric"><div class="k">Slice Tumor</div><div class="v">{coverage:.1f}%</div></div>'
            f'<div class="coverage-metric"><div class="k">Slice Normal</div><div class="v">{remaining_pct:.1f}%</div></div>'
            f'<div class="coverage-metric"><div class="k">Whole MRI</div><div class="v">{overall_pct_text}</div></div>'
            f'</div>'
            f'<div class="coverage-note">Neutral section of donut indicates non-tumor tissue (not highlighted).</div>'
            f'</div>'
        )
        quality_text = f"Dice {dice:.2f} / IoU {iou:.2f}" if (dice is not None and iou is not None) else "Dice/IoU not available"
        validation_text = "Ground truth matched" if (dice is not None and iou is not None) else "No seg mask"
        summary_line = f"Detected abnormal region consistent with tumor segmentation and {sev_bucket.lower()} severity."
        overall_text = f"{float(overall_tumor_pct):.1f}% of full MRI volume" if overall_tumor_pct is not None else "N/A"
        strip_html = (
            f'<div class="decision-chip"><div class="k">Risk Tier</div><div class="v">{risk_tier}</div></div>'
            f'<div class="decision-chip"><div class="k">Severity</div><div class="v">{severity}</div></div>'
            f'<div class="decision-chip"><div class="k">Overall Burden</div><div class="v">{overall_text}</div></div>'
        )

    seg_error_block = ""
    if modality == "MRI":
        if dice is not None:
            seg_err = max(0.0, min(100.0, (1.0 - float(dice)) * 100.0))
            err_color = "#2a8a54" if seg_err < 10 else ("#c48b10" if seg_err < 25 else "#b43b32")
            seg_error_block = (
                f'<div class="decision-mini">'
                f'<div class="k">Seg Error</div>'
                f'<div class="seg-error-value">{seg_err:.1f}%</div>'
                f'<div class="decision-mini-bar"><span style="width:{seg_err:.1f}%;background:{err_color};"></span></div>'
                f'<div class="decision-mini-note">Lower is better</div>'
                f'</div>'
            )
        else:
            seg_error_block = (
                '<div class="decision-mini">'
                '<div class="k">Seg Error</div>'
                '<div class="v">N/A</div>'
                '<div class="decision-mini-note">Ground truth mask required</div>'
                '</div>'
            )

    if modality == "MRI":
        right_cards_html = (
            f'<div class="decision-mini"><div class="k">Recommended Follow-up</div><div class="v">{follow_up_window}</div><div class="decision-mini-note">{follow_up_reason}</div></div>'
            f'<div class="decision-mini"><div class="k">Validation</div><div class="v">{validation_text}</div><div class="decision-mini-note">Mask availability and match status</div></div>'
            f'<div class="decision-mini"><div class="k">Quality</div><div class="v">{quality_text}</div><div class="decision-mini-note">Dice and IoU overlap metrics</div></div>'
            f'{seg_error_block}'
        )
    else:
        right_cards_html = (
            f'<div class="decision-mini"><div class="k">Decision Basis</div><div class="v">{validation_text}</div><div class="decision-mini-note">Model-derived probability signal</div></div>'
            f'<div class="decision-mini"><div class="k">Recommended Follow-up</div><div class="v">{follow_up_window}</div><div class="decision-mini-note">{follow_up_reason}</div></div>'
            f'<div class="decision-mini"><div class="k">Quality</div><div class="v">{quality_text}</div><div class="decision-mini-note">Classification confidence context</div></div>'
            f'<div class="decision-mini"><div class="k">Primary Metric</div><div class="v">{confidence_text}</div><div class="decision-mini-note">Predicted class confidence</div></div>'
        )

    st.markdown(f"""<div class="decision-card" style="--accent-color:{ring_color};--ring-color:{ring_color};--sev-color:{sev_fill_color};--sev-line:{ring_color};">
<div class="decision-head">
<span class="decision-badge {mode_cls}">{modality}</span>
<span class="decision-title">Prediction Card</span>
</div>
<div class="decision-strip">{strip_html}</div>
<div class="decision-grid">
<div class="decision-left">
<div class="decision-k">Disease</div>
<div class="decision-v">{disease}</div>
<div class="decision-k">Severity</div>
<div class="decision-v">{severity}</div>
<div class="decision-k">{metric_label}</div>
<div class="decision-v" style="font-size:1.12rem;">{confidence_text}</div>
{conf_bar_html}
<div class="risk-wrap">
<div class="decision-k" style="margin-bottom:0.25rem;">Probability / Risk Indicator</div>
<div class="risk-scale"><span class="risk-marker" style="left:{risk_pct:.1f}%;"></span></div>
<div class="risk-labels"><span>Low Risk</span><span>High Risk</span></div>
</div>
<div class="risk-wrap">
<div class="decision-k" style="margin-bottom:0.25rem;">Severity Bar</div>
<div class="sev-scale">
<span class="{sev_active[0]}">Mild</span><span class="{sev_active[1]}">Moderate</span><span class="{sev_active[2]}">Severe</span>
</div>
</div>
</div>
<div class="decision-side">
{coverage_visual_html}
<div class="decision-side-grid">{right_cards_html}</div>
</div>
</div>
<div class="summary-box">
<div class="k">Summary</div>
<div class="v">{summary_line}</div>
</div>
</div>""", unsafe_allow_html=True)


# MRI helpers used in clinician view
def _run_mri(flair_obj, t1ce_obj, t2_obj, seg_obj):
    # Handle both numpy arrays and nibabel objects
    is_nib_flair = not isinstance(flair_obj, np.ndarray)

    # All modalities are expected to share the same depth
    depth = flair_obj.shape[2]

    if is_nib_flair:
        # Working with nibabel images: use on-demand slice loading to reduce memory
        if seg_obj is not None:
            tumor_slices = []
            for k in range(depth):
                seg_slice = np.asarray(seg_obj.dataobj[:, :, k])
                if np.sum(seg_slice) > 500:
                    tumor_slices.append(k)
            tumor_slices = np.array(tumor_slices, dtype=int)

            if len(tumor_slices) > 0:
                if len(tumor_slices) <= 5:
                    selected = tumor_slices
                else:
                    pos = np.linspace(0, len(tumor_slices) - 1, 5, dtype=int)
                    selected = tumor_slices[pos]
            else:
                center = depth // 2
                selected = np.array([max(0, center - 1), center, min(depth - 1, center + 1)])
        else:
            center = depth // 2
            selected = np.array([max(0, center - 1), center, min(depth - 1, center + 1)])
    else:
        # Working with numpy arrays (backward compatibility)
        flair_data = lambda: flair_obj
        t1ce_data = lambda: t1ce_obj
        t2_data = lambda: t2_obj
        seg_data = (lambda: seg_obj) if seg_obj is not None else None

        if seg_obj is not None:
            tumor_slices = np.where(np.sum(seg_obj, axis=(0, 1)) > 500)[0]
            if len(tumor_slices) > 0:
                if len(tumor_slices) <= 5:
                    selected = tumor_slices
                else:
                    pos = np.linspace(0, len(tumor_slices) - 1, 5, dtype=int)
                    selected = tumor_slices[pos]
            else:
                center = depth // 2
                selected = np.array([max(0, center - 1), center, min(depth - 1, center + 1)])
        else:
            center = depth // 2
            selected = np.array([max(0, center - 1), center, min(depth - 1, center + 1)])

    overall_tumor_pct = _mri_overall_coverage(seg_obj, is_nib_flair)

    slice_runs = []
    for slice_idx in selected:
        if is_nib_flair:
            # Load only this slice from each modality, as float32
            f_slice = np.asarray(flair_obj.dataobj[:, :, slice_idx], dtype=np.float32)
            t_slice = np.asarray(t1ce_obj.dataobj[:, :, slice_idx], dtype=np.float32)
            t2_slice = np.asarray(t2_obj.dataobj[:, :, slice_idx], dtype=np.float32)
        else:
            f_slice = flair_data()[:, :, slice_idx]
            t_slice = t1ce_data()[:, :, slice_idx]
            t2_slice = t2_data()[:, :, slice_idx]

        img = preprocess_slice(f_slice, t_slice, t2_slice)
        mask, prob_map = predict_mri(img)

        gt = None
        if seg_obj is not None:
            if is_nib_flair:
                seg_slice = np.asarray(seg_obj.dataobj[:, :, slice_idx])
            else:
                seg_slice = seg_data()[:, :, slice_idx] if seg_data is not None else None

            if seg_slice is not None:
                gt = (seg_slice > 0).astype(np.float32)
                gt = cv2.resize(gt, (128, 128), interpolation=cv2.INTER_NEAREST)

        fig, tp, dice, iou = render_mri_panel(img, mask, prob_map, gt)
        overlay_img = get_mri_overlay(img, mask)
        slice_runs.append((fig, tp, dice, iou, overlay_img))

    # Use the slice with highest predicted tumor area for visualization
    # and report metrics for that same slice so the numbers shown under
    # the figure match the title inside the panel.
    best_idx = int(np.argmax([x[1] for x in slice_runs]))
    fig, tp, dice, iou, best_overlay = slice_runs[best_idx]

    tp = float(tp)
    dice = float(dice) if dice is not None else None
    iou = float(iou) if iou is not None else None

    sev = _mri_severity_from_coverage(tp)

    return fig, best_overlay, tp, overall_tumor_pct, dice, iou, sev


def _show_mri(fig, best_overlay, tp, overall_tumor_pct, dice, iou, sev):
    st.session_state.last_modality  = "MRI"
    st.session_state.last_label     = "Tumor Segmentation"
    st.session_state.last_severity  = sev
    st.session_state.last_prob      = None
    st.session_state.last_tumor_pct = tp
    st.session_state.last_overall_tumor_pct = overall_tumor_pct
    st.session_state.last_mri_fig   = fig
    st.session_state.last_mri_overlay = best_overlay
    st.session_state.last_dice      = dice
    st.session_state.last_iou       = iou

    st.markdown("""
<div class="result-card" style="margin-top:1rem;">
    <h3>🩻 MRI Segmentation</h3>
""", unsafe_allow_html=True)
    # Do not clear the figure so it can be reused when
    # the clinician navigates away and comes back.
    st.pyplot(fig, clear_figure=False, width="stretch")

    st.markdown("""
<div class="xai-tag-row">
    <span class="xai-tag">🎯 Model focused here</span>
    <span class="xai-tag">🧠 Important region highlighted</span>
</div>
""", unsafe_allow_html=True)

    _show_decision_snapshot(
        "MRI",
        "Tumor Segmentation",
        sev,
        confidence=None,
        tumor_pct=tp,
        overall_tumor_pct=overall_tumor_pct,
        dice=dice,
        iou=iou,
    )

    dice_text = f"{dice:.2f}" if dice is not None else "N/A"
    iou_text = f"{iou:.2f}" if iou is not None else "N/A"
    validation_status = "Ground-truth matched" if (dice is not None and iou is not None) else "No ground-truth mask"

    interpretation = (
        f"Predicted lesion burden is {tp:.2f}% on the analyzed MRI slice. "
        f"Segmentation severity band is {sev}."
    )
    overall_text = f"{overall_tumor_pct:.2f}%" if overall_tumor_pct is not None else "N/A"
    clinical_summary = f"""
<div class="clinical-card">
    <h3>Clinical Summary</h3>
    <ul>
        <li><strong>Finding:</strong> Tumor Segmentation</li>
        <li><strong>Severity Tier:</strong> {sev}</li>
        <li><strong>Slice Coverage:</strong> {tp:.2f}%</li>
        <li><strong>Overall Coverage:</strong> {overall_text}</li>
        <li><strong>Quality Metrics:</strong> Dice {dice_text} | IoU {iou_text}</li>
        <li><strong>Validation Data:</strong> {validation_status}</li>
        <li><strong>Interpretation:</strong> {interpretation}</li>
    </ul>
</div>
"""
    st.markdown(clinical_summary, unsafe_allow_html=True)
    st.session_state.last_clinical_summary_html = clinical_summary
    return generate_report("MRI", "Tumor Segmentation", sev)

def patient_severity_bucket(severity, modality):
    s = (severity or "").lower()
    if modality == "MRI":
        if any(k in s for k in ["no tumor", "none", "normal"]):
            return "No Tumor"
        if any(k in s for k in ["small", "mild"]):
            return "Mild"
        if any(k in s for k in ["medium", "moderate"]):
            return "Moderate"
        if any(k in s for k in ["large", "severe"]):
            return "Severe"
        return "Moderate"
    if "normal" in s or "no hemorrhage" in s:
        return "Normal"
    if "severe" in s:
        return "Severe"
    if "moderate" in s:
        return "Moderate"
    return "Mild"

def patient_followup_window(modality, severity, label):
    bucket = patient_severity_bucket(severity, modality)
    if modality == "CT" and "normal" in (label or "").lower():
        return "Routine follow-up"
    if modality == "MRI" and bucket == "No Tumor":
        return "Routine follow-up"
    if bucket == "Severe":
        return "Urgent care today"
    if bucket == "Moderate":
        return "Within 48-72 hours"
    return "Within 3-7 days"

def patient_urgency_text(modality, severity):
    bucket = patient_severity_bucket(severity, modality)
    if bucket in ["Normal", "No Tumor"]:
        return "Low immediate risk, but continue routine medical follow-up."
    if bucket == "Mild":
        return "Lower-to-moderate urgency. Arrange specialist follow-up soon."
    if bucket == "Moderate":
        return "Needs timely medical follow-up and close symptom monitoring."
    return "High-priority finding. Seek urgent specialist care now."


def patient_initials(full_name):
    parts = [p for p in str(full_name).split() if p]
    if not parts:
        return "NA"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def patient_severity_banner(severity):
    s = str(severity).lower()
    if "severe" in s or "high" in s:
        return "High Concern", "🚨"
    if "moderate" in s:
        return "Moderate Concern", "⚠️"
    if "mild" in s or "low" in s or "normal" in s or "no tumor" in s:
        return "Low Concern", "✅"
    return "Clinical Review Needed", "🩺"


def patient_basic_points(modality, label, severity_clean, tumor_pct=None):
    is_ct = modality == "CT"
    label_l = str(label).lower()

    if is_ct:
        scan_text = "You had a CT brain scan, which is a fast scan used to check for bleeding or urgent brain changes."
        if "normal" in label_l:
            where_text = "The AI did not highlight a concerning area in this scan."
        else:
            where_text = "The AI highlighted an unusual area in the brain image that needs doctor review."
    else:
        scan_text = "You had an MRI brain scan, which gives detailed pictures of brain tissue."
        if tumor_pct is not None:
            where_text = (
                "The AI marked a suspicious tissue area in the analyzed MRI region "
                f"(about {float(tumor_pct):.1f}% of that image area)."
            )
        else:
            where_text = "The AI marked a suspicious tissue area that should be checked with your doctor."

    meaning_text = (
        "This result may help your doctor decide how urgent follow-up should be, "
        f"based on a {str(severity_clean).lower()} severity level."
    )
    return [scan_text, where_text, meaning_text]


def get_ct_patient_text(label):
    label_text = str(label or "").strip()
    label_l = label_text.lower()

    if "hemorrhage" in label_l or "bleed" in label_l:
        key = "Intracranial Hemorrhage Detected"
    elif "normal" in label_l or "no hemorrhage" in label_l:
        key = "Normal"
    elif "abnormal" in label_l:
        key = "Abnormality Detected"
    else:
        key = label_text

    what_found = PATIENT_CONTENT["CT"]["what_found"].get(
        key,
        "The scan flagged an area that looks different from a typical brain scan and needs doctor review.",
    )
    meaning = PATIENT_CONTENT["CT"]["plain_meaning"].get(
        key,
        "This does not confirm a final diagnosis. It is an early AI signal that should be checked by your doctor.",
    )
    return what_found, meaning


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    if st.session_state.page == "clinician":
        st.markdown("### NeuroVision AI")
        st.caption("Clinical Decision Support")
        st.markdown("""
<div class="side-nav">
    <div class="side-nav-item side-nav-active">Analysis Studio</div>
    <div class="side-nav-item side-nav-disabled">Patient Report</div>
</div>
""", unsafe_allow_html=True)
        st.divider()
        st.markdown("**Workflow**")
        st.markdown("1. Upload Scan")
        st.markdown("2. Modality Detection")
        st.markdown("3. AI Inference")
        st.markdown("4. View Results")
        st.divider()
        st.markdown("**Supported Formats**")
        st.markdown("• `.dcm` (CT)")
        st.markdown("• `.nii` / `.nii.gz` (MRI)")
        st.markdown("• `.zip`")
        st.caption("MRI requires flair, t1ce, t2 sequences")
        st.divider()
        has_result = st.session_state.last_modality is not None
        if st.button("Open Patient Report", key="goto_patient", disabled=not has_result):
            st.session_state.page = "patient"
            st.rerun()
        if not has_result:
            st.caption("Currently disabled until results are available.")
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        st.caption("v2.0 • Research Use Only")

    else:  # patient page sidebar
        st.markdown("### Patient Report")
        st.caption("Plain-language summary for patients")
        st.divider()
        if st.button("← Back to Clinician View", key="goto_clinician"):
            st.session_state.page = "clinician"
            st.rerun()
        st.divider()
        st.markdown("**Premium Full Report**")
        if not st.session_state.patient_unlocked:
            st.markdown("""<div class="sidebar-premium-box">
<div class="sidebar-premium-tag">PREMIUM</div>
<div class="sidebar-premium-title">Unlock Full Report</div>
<div class="sidebar-premium-copy">Causes, what to do next, lifestyle tips and emergency signs in plain language.</div>
<div class="sidebar-premium-price">₹500 one-time</div>
</div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔓  Unlock Full Report", key="unlock_sidebar"):
                st.session_state.patient_unlocked = True
                st.rerun()
        else:
            st.success("✅ Full report unlocked")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: CLINICIAN VIEW
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.page == "clinician":
    hero_preview_html = _hero_scan_preview_html()
    hero_header_html = textwrap.dedent(f"""
<div class="hero-wrap">
    <div class="hero-layout">
        <div>
            <div class="hero-eyebrow">NeuroVision AI</div>
            <h1 class="hero-title">Brain Imaging Analysis Studio</h1>
            <p class="hero-sub">Clinical decision support system for CT hemorrhage detection and MRI tumor segmentation with explainable AI.</p>
            <div class="pill-row">
                <span class="pill pill-blue"><span class="pill-dot"></span>CT Detection</span>
                <span class="pill pill-purple"><span class="pill-dot"></span>MRI Segmentation</span>
                <span class="pill pill-green"><span class="pill-dot"></span>Explainable AI</span>
                <span class="pill pill-amber"><span class="pill-dot"></span>Severity Scoring</span>
            </div>
        </div>
        {hero_preview_html}
    </div>
</div>
""").strip()
    st.markdown(hero_header_html, unsafe_allow_html=True)

    st.markdown("""
<div class="scan-card">
    <h3>Start a New Patient Analysis</h3>
    <p>Upload scan files to run automated modality detection, AI inference, explainability maps, and severity-focused summaries.</p>
</div>""", unsafe_allow_html=True)

    # Generate rotating tips
    tip_idx = next_tip_idx()
    tips = TIP_SETS[tip_idx]
    
    tips_html = f"""
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.8rem;margin-bottom:1.2rem;">
    <div class="upload-note-box">
        <div class="k">{tips[0][0]}</div>
        <div class="v">{tips[0][1]}</div>
    </div>
    <div class="upload-note-box">
        <div class="k">{tips[1][0]}</div>
        <div class="v">{tips[1][1]}</div>
    </div>
    <div class="upload-note-box">
        <div class="k">{tips[2][0]}</div>
        <div class="v">{tips[2][1]}</div>
    </div>
</div>"""
    st.markdown(tips_html, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload scan files for analysis",
        type=["dcm", "nii", "nii.gz", "zip"],
        key="primary_scan_upload",
        accept_multiple_files=True,
    )

    # If no new upload but we already have a result from this
    # session, re-show the last visuals so the clinician view does
    # not appear empty when navigating back from the patient page.
    if not uploaded_files and st.session_state.last_modality is not None:
        if st.session_state.last_modality == "CT" and st.session_state.last_ct_overlay is not None:
            label    = st.session_state.last_label or "Unknown"
            severity = st.session_state.last_severity or "Moderate"
            prob     = st.session_state.last_prob or 0.0

            st.markdown("""
<div class="modality-card">
    <div class="section-label">Detected Modality</div>
    <span class="modality-pill" style="border-color:#9fc2dd;background:#e8f3fb;color:#1e537a;">🧠 CT Scan</span>
</div>
""", unsafe_allow_html=True)

            st.markdown(f"""
<div class="result-grid">
    <div class="result-card">
        <h3>🩺 Prediction Result</h3>
        <div class="section-label">AI finding</div>
        <div style="font-family:'Inter',sans-serif;font-size:1.5rem;color:#16344d;margin-top:0.15rem;font-weight:800;">{label}</div>
        <div style="margin-top:0.95rem;">
            <div class="section-label">Confidence</div>
            <div style="font-family:'Space Mono',monospace;font-size:1.6rem;color:var(--accent);margin-top:0.12rem;">{prob:.1%}</div>
            <div class="ct-bullet-wrap">
                <div class="ct-bullet-track"><div class="ct-bullet-marker" style="left:{prob*100:.1f}%;"></div></div>
                <div class="ct-bullet-scale"><span>Low</span><span>Moderate</span><span>High</span></div>
            </div>
        </div>
    </div>
    <div class="result-card">
        <h3>🧠 Severity Level</h3>
        <div class="pat-sev-badge ct-sev-neutral">{severity or "N/A"}</div>
        {_ct_severity_variant_html(severity)}
        <div style="margin-top:0.9rem;padding-top:0.9rem;border-top:1px solid var(--border);font-family:'Space Mono',monospace;font-size:0.7rem;color:var(--muted);line-height:1.6;">
            AI decision-support only.<br>Clinical judgement takes precedence.
        </div>
    </div>
</div>""", unsafe_allow_html=True)

            _show_ct_visual_stage(st.session_state.last_ct_overlay, label, severity, prob)

            ct_label    = label
            ct_severity = severity or "N/A"
            ct_prob_pct = f"{prob*100:.1f}%"

            st.markdown(f"""
<div class="metric-row">
    <div class="metric-tile"><div class="k">Hemorrhage Risk</div><div class="v">{ct_prob_pct}</div><div class="m">Model-estimated probability of intracranial hemorrhage.</div></div>
    <div class="metric-tile"><div class="k">Severity Tier</div><div class="v">{ct_severity}</div><div class="m">Risk level derived from the model's confidence.</div></div>
    <div class="metric-tile"><div class="k">AI Finding</div><div class="v">{ct_label}</div><div class="m">Primary classification for this CT scan.</div></div>
</div>
""", unsafe_allow_html=True)

        elif st.session_state.last_modality == "MRI" and st.session_state.last_mri_fig is not None and st.session_state.last_mri_overlay is not None:
            # Reuse previously computed MRI panel and overlay
            _show_mri_modality_card()
            _show_mri(
                st.session_state.last_mri_fig,
                st.session_state.last_mri_overlay,
                st.session_state.last_tumor_pct or 0.0,
                st.session_state.last_overall_tumor_pct,
                st.session_state.last_dice,
                st.session_state.last_iou,
                st.session_state.last_severity or "Moderate",
            )

    if uploaded_files:
        names       = [f.name for f in uploaded_files]
        is_ct_batch = any(n.lower().endswith(".dcm") for n in names)
        zip_files   = [f for f in uploaded_files if f.name.lower().endswith(".zip")]
        report      = None
        clinical_summary_html = ""

        if zip_files and len(uploaded_files) > 1:
            st.error("Upload either one .zip or multiple MRI files — not both.")

        # ── CT ───────────────────────────────────────────────────────────────
        elif is_ct_batch:
            ct_file   = next(f for f in uploaded_files if f.name.lower().endswith(".dcm"))
            file_path = save_temp(ct_file)

            img         = preprocess_ct(file_path)
            label, prob = predict_ct(img)
            severity    = ct_severity(prob)
            overlay     = get_gradcam_overlay(img, load_ct_model())

            # New analyzed case should start with premium report locked.
            st.session_state.patient_unlocked = False

            st.session_state.last_modality  = "CT"
            st.session_state.last_label     = label
            st.session_state.last_severity  = severity
            st.session_state.last_prob      = prob
            st.session_state.last_tumor_pct = None
            st.session_state.last_ct_overlay = overlay

            st.markdown("""
<div class="modality-card">
    <div class="section-label">Detected Modality</div>
    <span class="modality-pill" style="border-color:#9fc2dd;background:#e8f3fb;color:#1e537a;">🧠 CT Scan</span>
</div>
""", unsafe_allow_html=True)

            st.markdown(f"""
<div class="result-grid">
    <div class="result-card">
        <h3>🩺 Prediction Result</h3>
        <div class="section-label">AI finding</div>
        <div style="font-family:'Inter',sans-serif;font-size:1.5rem;color:#16344d;margin-top:0.15rem;font-weight:800;">{label}</div>
        <div style="margin-top:0.95rem;">
            <div class="section-label">Confidence</div>
            <div style="font-family:'Space Mono',monospace;font-size:1.6rem;color:var(--accent);margin-top:0.12rem;">{prob:.1%}</div>
            <div class="ct-bullet-wrap">
                <div class="ct-bullet-track"><div class="ct-bullet-marker" style="left:{prob*100:.1f}%;"></div></div>
                <div class="ct-bullet-scale"><span>Low</span><span>Moderate</span><span>High</span></div>
            </div>
        </div>
    </div>
    <div class="result-card">
        <h3>🧠 Severity Level</h3>
        <div class="pat-sev-badge ct-sev-neutral">{severity or "N/A"}</div>
        {_ct_severity_variant_html(severity)}
        <div style="margin-top:0.9rem;padding-top:0.9rem;border-top:1px solid var(--border);font-family:'Space Mono',monospace;font-size:0.7rem;color:var(--muted);line-height:1.6;">
            AI decision-support only.<br>Clinical judgement takes precedence.
        </div>
    </div>
</div>""", unsafe_allow_html=True)

            _show_ct_visual_stage(overlay, label, severity, prob)

            ct_label    = label
            ct_severity = severity or "N/A"
            ct_prob_pct = f"{prob*100:.1f}%"

            st.markdown(f"""
<div class="metric-row">
    <div class="metric-tile"><div class="k">Hemorrhage Risk</div><div class="v">{ct_prob_pct}</div><div class="m">Model-estimated probability of intracranial hemorrhage.</div></div>
    <div class="metric-tile"><div class="k">Severity Tier</div><div class="v">{ct_severity}</div><div class="m">Risk level derived from the model's confidence.</div></div>
    <div class="metric-tile"><div class="k">AI Finding</div><div class="v">{ct_label}</div><div class="m">Primary classification for this CT scan.</div></div>
</div>
""", unsafe_allow_html=True)

            clinical_summary_html = f"""
<div class="clinical-card">
    <h3>Clinical Summary</h3>
    <ul>
        <li><strong>Finding:</strong> {label}</li>
        <li><strong>Severity:</strong> {severity}</li>
        <li><strong>Confidence:</strong> {prob:.1%}</li>
        <li><strong>Interpretation:</strong> Model indicates CT pattern consistent with the reported finding. Correlate clinically.</li>
    </ul>
</div>
"""
            st.session_state.last_clinical_summary_html = clinical_summary_html

            report = generate_report("CT", label, severity, prob)

        # ── MRI ──────────────────────────────────────────────────────────────
        else:
            _show_mri_modality_card()
            picked = {}

            def _run_mri(flair_obj, t1ce_obj, t2_obj, seg_obj):
                # Handle both numpy arrays and nibabel objects
                is_nib_flair = not isinstance(flair_obj, np.ndarray)

                # All modalities are expected to share the same depth
                depth = flair_obj.shape[2]

                if seg_obj is None:
                    overall_tumor_pct = None
                elif is_nib_flair:
                    overall_tumor_pct = _mri_overall_coverage(seg_obj, True)
                else:
                    overall_tumor_pct = _mri_overall_coverage(seg_obj, False)

                if is_nib_flair:
                    # Working with nibabel images: use on-demand slice loading to reduce memory
                    if seg_obj is not None:
                        tumor_slices = []
                        for k in range(depth):
                            seg_slice = np.asarray(seg_obj.dataobj[:, :, k])
                            if np.sum(seg_slice) > 500:
                                tumor_slices.append(k)
                        tumor_slices = np.array(tumor_slices, dtype=int)

                        if len(tumor_slices) > 0:
                            if len(tumor_slices) <= 5:
                                selected = tumor_slices
                            else:
                                pos = np.linspace(0, len(tumor_slices) - 1, 5, dtype=int)
                                selected = tumor_slices[pos]
                        else:
                            center = depth // 2
                            selected = np.array([max(0, center - 1), center, min(depth - 1, center + 1)])
                    else:
                        center = depth // 2
                        selected = np.array([max(0, center - 1), center, min(depth - 1, center + 1)])
                else:
                    # Working with numpy arrays (backward compatibility)
                    flair_data = lambda: flair_obj
                    t1ce_data = lambda: t1ce_obj
                    t2_data = lambda: t2_obj
                    seg_data = (lambda: seg_obj) if seg_obj is not None else None

                    if seg_obj is not None:
                        tumor_slices = np.where(np.sum(seg_obj, axis=(0, 1)) > 500)[0]
                        if len(tumor_slices) > 0:
                            if len(tumor_slices) <= 5:
                                selected = tumor_slices
                            else:
                                pos = np.linspace(0, len(tumor_slices) - 1, 5, dtype=int)
                                selected = tumor_slices[pos]
                        else:
                            center = depth // 2
                            selected = np.array([max(0, center - 1), center, min(depth - 1, center + 1)])
                    else:
                        center = depth // 2
                        selected = np.array([max(0, center - 1), center, min(depth - 1, center + 1)])

                slice_runs = []
                for slice_idx in selected:
                    if is_nib_flair:
                        # Load only this slice from each modality, as float32
                        f_slice = np.asarray(flair_obj.dataobj[:, :, slice_idx], dtype=np.float32)
                        t_slice = np.asarray(t1ce_obj.dataobj[:, :, slice_idx], dtype=np.float32)
                        t2_slice = np.asarray(t2_obj.dataobj[:, :, slice_idx], dtype=np.float32)
                    else:
                        f_slice = flair_data()[:, :, slice_idx]
                        t_slice = t1ce_data()[:, :, slice_idx]
                        t2_slice = t2_data()[:, :, slice_idx]

                    img = preprocess_slice(f_slice, t_slice, t2_slice)
                    mask, prob_map = predict_mri(img)

                    gt = None
                    if seg_obj is not None:
                        if is_nib_flair:
                            seg_slice = np.asarray(seg_obj.dataobj[:, :, slice_idx])
                        else:
                            seg_slice = seg_data()[:, :, slice_idx] if seg_data is not None else None

                        if seg_slice is not None:
                            gt = (seg_slice > 0).astype(np.float32)
                            gt = cv2.resize(gt, (128, 128), interpolation=cv2.INTER_NEAREST)

                    fig, tp, dice, iou = render_mri_panel(img, mask, prob_map, gt)
                    overlay_img = get_mri_overlay(img, mask)
                    slice_runs.append((fig, tp, dice, iou, overlay_img))

                # Use the slice with highest predicted tumor area for visualization
                # and report metrics for that same slice so the tiles match the
                # figure title and overlay.
                best_idx = int(np.argmax([x[1] for x in slice_runs]))
                fig, tp, dice, iou, best_overlay = slice_runs[best_idx]

                tp = float(tp)
                dice = float(dice) if dice is not None else None
                iou = float(iou) if iou is not None else None

                sev = _mri_severity_from_coverage(tp)

                return fig, best_overlay, tp, overall_tumor_pct, dice, iou, sev

            if len(zip_files) == 1:
                zf_upload = zip_files[0]
                with tempfile.TemporaryDirectory() as ext_dir:
                    try:
                        zf_upload.seek(0)
                        with zipfile.ZipFile(zf_upload) as zf: zf.extractall(ext_dir)
                    except zipfile.BadZipFile:
                        st.error("Invalid ZIP file.")
                    else:
                        nii_files = [p for p in Path(ext_dir).rglob("*")
                                     if p.is_file() and (p.name.lower().endswith(".nii") or p.name.lower().endswith(".nii.gz"))]
                        for nf in nii_files:
                            m = infer_mri_modality(nf.name)
                            if m and m not in picked: picked[m] = str(nf)
                        missing = [m for m in ["flair","t1ce","t2"] if m not in picked]
                        if missing:
                            st.error("ZIP missing: " + ", ".join(missing))
                        else:
                            # Load nibabel images and let _run_mri handle memory-efficient slicing
                            flair = nib.load(picked["flair"])
                            t1ce  = nib.load(picked["t1ce"])
                            t2    = nib.load(picked["t2"])
                            seg   = nib.load(picked["seg"]) if "seg" in picked else None
                            fig, best_overlay, tp, overall_tumor_pct, dice, iou, sev = _run_mri(flair, t1ce, t2, seg)

                            # New analyzed case should start with premium report locked.
                            st.session_state.patient_unlocked = False
                            report = _show_mri(fig, best_overlay, tp, overall_tumor_pct, dice, iou, sev)
            else:
                for uploaded in uploaded_files:
                    m = infer_mri_modality(uploaded.name)
                    if m and m not in picked: picked[m] = uploaded
                missing = [m for m in ["flair","t1ce","t2"] if m not in picked]
                if missing:
                    st.error("Could not find: " + ", ".join(missing))
                else:
                    fp = save_temp(picked["flair"]); t1p = save_temp(picked["t1ce"]); t2p = save_temp(picked["t2"])
                    sp = save_temp(picked["seg"]) if "seg" in picked else None
                    flair = nib.load(fp)
                    t1ce = nib.load(t1p)
                    t2 = nib.load(t2p)
                    seg = nib.load(sp) if sp else None
                    fig, best_overlay, tp, overall_tumor_pct, dice, iou, sev = _run_mri(flair, t1ce, t2, seg)

                    # New analyzed case should start with premium report locked.
                    st.session_state.patient_unlocked = False
                    report = _show_mri(fig, best_overlay, tp, overall_tumor_pct, dice, iou, sev)

        # ── Clinical Report / navigation to patient page ─────────────────────
        # Use the last stored clinical summary so results persist when
        # revisiting the clinician page.
        if st.session_state.last_modality is not None:
            if st.session_state.last_clinical_summary_html and st.session_state.last_modality != "MRI":
                st.markdown(st.session_state.last_clinical_summary_html, unsafe_allow_html=True)
            if st.button("Generate Patient Report", key="open_patient_from_main"):
                st.session_state.page = "patient"
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PATIENT-FRIENDLY REPORT
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == "patient":

    modality = st.session_state.last_modality
    label    = st.session_state.last_label or "Unknown"
    severity = st.session_state.last_severity or "Moderate"
    prob     = st.session_state.last_prob
    tumor_pct= st.session_state.last_tumor_pct

    if modality is None:
        st.warning("No analysis results found.")
        if st.button("← Go back"):
            st.session_state.page = "clinician"
            st.rerun()
        st.stop()

    # -------- FIX SEVERITY --------
    def normalize_mri_severity(sev):
        s = sev.lower()
        if "no tumor" in s: return "No Tumor"
        if "small" in s: return "Mild"
        if "medium" in s: return "Moderate"
        if "large" in s: return "Severe"
        return "Moderate"

    if modality == "MRI":
        severity_clean = normalize_mri_severity(severity)
    else:
        severity_clean = severity

    is_ct = modality == "CT"

    # -------- CONTENT --------
    if is_ct:
        found_text, meaning = get_ct_patient_text(label)
    else:
        data = PATIENT_CONTENT["MRI"]
        found_text = data["what_found_tumor"]
        meaning    = data["plain_meaning_tumor"]

    profile = st.session_state.patient_profile
    now = datetime.now()
    report_id = f"NV-RPT-{now.strftime('%Y%m%d')}-0481"

    # 1) PAGE HEADER
    st.markdown("""
    <div class="pat-section">
      <div style="font-size:0.78rem;letter-spacing:0.12em;font-weight:700;color:#50657a;">PATIENT REPORT · CONFIDENTIAL</div>
            <h2 style="margin-top:0.45rem;">Understanding Your Brain Scan</h2>
      <p style="margin-bottom:0;">A clear, plain-language summary of your scan results.</p>
    </div>
    """, unsafe_allow_html=True)

    # 2) PATIENT INFO CARD (always visible)
    st.markdown('<div class="pat-section">', unsafe_allow_html=True)
    c_left, c_right = st.columns([1, 3])

    with c_left:
        initials = patient_initials(profile["name"])
        st.markdown(
            f"""
            <div style='width:80px;height:80px;border-radius:50%;background:#e8f1ff;display:flex;align-items:center;
                        justify-content:center;font-size:30px;font-weight:700;color:#1b4769;margin-bottom:0.5rem;'>
                {initials}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(profile["name"])
        st.caption("Patient")
        st.write(profile["patient_id"])

    with c_right:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.caption("Date of Birth")
            st.write(profile["dob"])
            st.caption("Scan Date")
            st.write(now.strftime("%Y-%m-%d"))
        with m2:
            st.caption("Age / Gender")
            st.write(profile["age_gender"])
            st.caption("Modality")
            st.write(modality)
        with m3:
            st.caption("Referring Physician")
            st.write(profile["physician"])
            st.caption("Facility / Hospital")
            st.write(profile["facility"])

    p1, p2, p3 = st.columns(3)
    p1.info("Scan Quality: High")
    p2.info("Analysis: Complete")
    p3.warning("Reviewed: Pending")
    st.markdown('</div>', unsafe_allow_html=True)

    # 7) PREMIUM LOCK OVERLAY (controls sections 3-6)
    if not st.session_state.patient_unlocked:
        st.markdown("""
        <div class="pat-section">
            <h3>🔒 Premium Report Locked</h3>
            <p>Unlock the full plain-language report, recommendations, and next steps.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Unlock Full Report", key="unlock_patient_center"):
            st.session_state.patient_unlocked = True
            st.rerun()
    else:
        # 3) DIAGNOSIS SUMMARY CARD
        sev_title, sev_icon = patient_severity_banner(severity_clean)
        st.markdown('<div class="pat-section">', unsafe_allow_html=True)
        st.subheader("What We Found")
        st.success(f"{sev_icon} {sev_title}")
        st.write(found_text)
        dc1, dc2 = st.columns(2)
        with dc1:
            st.write("How sure is this result?")
            if is_ct and prob is not None:
                if prob >= 0.85:
                    conf_label = "High"
                elif prob >= 0.70:
                    conf_label = "Moderate"
                else:
                    conf_label = "Limited"
            else:
                conf_label = "Moderate"
            st.info(f"AI certainty: {conf_label}")
            st.caption("This is a support estimate, not a final diagnosis.")
        with dc2:
            st.write("How urgent is follow-up?")
            follow_window = patient_followup_window(modality, severity_clean, label)
            st.info(follow_window)
            st.caption(patient_urgency_text(modality, severity_clean))
        st.markdown('</div>', unsafe_allow_html=True)

        # 4) WHAT THIS MEANS CARD
        st.markdown('<div class="pat-section">', unsafe_allow_html=True)
        st.subheader("What This Means For You")
        st.write(f"- {meaning}")
        st.write(f"- {patient_urgency_text(modality, severity_clean)}")
        st.write("- Your doctor will combine this with symptoms, examination, and history before final decisions.")
        st.markdown('</div>', unsafe_allow_html=True)

        # 5) RECOMMENDED NEXT STEPS CARD
        st.markdown('<div class="pat-section">', unsafe_allow_html=True)
        st.subheader("Next Steps")
        next_steps = [
            ("Schedule Clinical Follow-up", f"Recommended timeline: {follow_window}."),
            ("Review Scan With Specialist", "Confirm AI findings with radiology and clinical context."),
            ("Track Symptoms Daily", "Note any new or worsening neurological symptoms."),
            ("Plan Repeat Imaging If Advised", "Follow your clinician's timeline for repeat CT/MRI."),
        ]
        for idx, (title, desc) in enumerate(next_steps, start=1):
            st.write(f"{idx}. {title}")
            st.caption(desc)
        st.markdown('</div>', unsafe_allow_html=True)

        # Basic plain-language section before doctor questions
        st.markdown('<div class="pat-section">', unsafe_allow_html=True)
        st.subheader("Your Scan, In Simple Words")
        basic_points = patient_basic_points(modality, label, severity_clean, tumor_pct)
        st.write(f"- What scan was done: {basic_points[0]}")
        st.write(f"- Where the issue appears: {basic_points[1]}")
        st.write(f"- What this means right now: {basic_points[2]}")
        st.markdown('</div>', unsafe_allow_html=True)

        # 6) QUESTIONS FOR YOUR DOCTOR CARD
        st.markdown('<div class="pat-section">', unsafe_allow_html=True)
        st.subheader("Questions to Ask Your Doctor")
        questions = [
            "How certain is this finding, and what confirms it?",
            "Do I need another scan, and when should it happen?",
            "What symptoms should make me seek urgent care?",
            "How do these results affect my treatment plan?",
            "Should I see a neurologist or another specialist now?",
        ]
        for q in questions:
            st.info(f"💬 \"{q}\"")
        st.markdown('</div>', unsafe_allow_html=True)

    # 8) REPORT FOOTER (always visible)
    st.markdown('<div class="pat-section">', unsafe_allow_html=True)
    f1, f2 = st.columns([1.3, 1.7])
    with f1:
        st.caption(f"Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        st.caption(f"Report ID: {report_id}")
    with f2:
        st.caption("This report is AI-generated and must be reviewed by a licensed clinician.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Report export remains unchanged in logic
    report_text = generate_report(
        modality,
        label,
        severity,
        prob if is_ct else None
    )

    st.download_button(
        "Download Report",
        report_text,
        file_name="brain_report.txt"
    )