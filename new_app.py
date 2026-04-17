import streamlit as st
import numpy as np
import nibabel as nib
import tempfile
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from preprocessing.preprocess_ct import preprocess_ct
from preprocessing.preprocess_mri import preprocess_slice
from models.ct_model import predict_ct, load_ct_model
from models.mri_model import predict_mri
from xai.xai_ct import get_gradcam_overlay
from xai.xai_mri import get_mri_overlay
from utils.severity import ct_severity, mri_severity
from utils.report import generate_report

st.set_page_config(page_title="NeuroVision AI V2", layout="wide")


# ================= THEME TOGGLE =================
theme = st.sidebar.radio("Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown(
        """
    <style>
    body, .stApp {background-color:#0e1117; color:white;}
    .card {background:#161b22; color:white;}
    </style>
    """,
        unsafe_allow_html=True,
    )


# ================= SESSION =================
if "page" not in st.session_state:
    st.session_state.page = "clinician"
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None
if "result" not in st.session_state:
    st.session_state.result = None
if "history" not in st.session_state:
    st.session_state.history = []
if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""
if "patient_id" not in st.session_state:
    st.session_state.patient_id = ""
if "patient_age" not in st.session_state:
    st.session_state.patient_age = 0
if "patient_gender" not in st.session_state:
    st.session_state.patient_gender = "Prefer not to say"


def save_uploaded_temp(uploaded_file):
    suffix = "".join(Path(uploaded_file.name).suffixes) or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def infer_mri_modality(filename):
    name = filename.lower()
    if "seg" in name:
        return "seg"
    if "flair" in name:
        return "flair"
    if "t1ce" in name or "t1c" in name:
        return "t1ce"
    if "t2" in name:
        return "t2"
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
    tumor_percent = (np.sum(pred_mask) / pred_mask.size) * 100
    if gt_mask is not None:
        dice = dice_score(gt_mask, pred_mask)
        iou = iou_score(gt_mask, pred_mask)
        error_map = np.abs(gt_mask.astype(np.float32) - pred_mask.astype(np.float32))
        gt_title = "Ground Truth"
        overlay_title = f"Overlay\nDice:{dice:.2f}"
        suptitle = f"Tumor Area: {tumor_percent:.2f}% | IoU:{iou:.2f}"
    else:
        dice = iou = None
        error_map = np.zeros_like(pred_mask, dtype=np.float32)
        gt_mask = np.zeros_like(pred_mask, dtype=np.float32)
        gt_title = "Ground Truth (N/A)"
        overlay_title = "Overlay"
        suptitle = f"Tumor Area: {tumor_percent:.2f}%"

    fig, ax = plt.subplots(1, 6, figsize=(16, 3), dpi=80, constrained_layout=True)
    fig.patch.set_facecolor("#f8fbff")
    for a in ax:
        a.set_facecolor("#f8fbff")

    ax[0].imshow(img[:, :, 0], cmap="gray")
    ax[0].set_title("MRI Slice", color="#5c7087", fontsize=9)
    ax[0].axis("off")

    ax[1].imshow(gt_mask, cmap="gray")
    ax[1].set_title(gt_title, color="#5c7087", fontsize=9)
    ax[1].axis("off")

    ax[2].imshow(pred_mask, cmap="gray")
    ax[2].set_title("Predicted Mask", color="#5c7087", fontsize=9)
    ax[2].axis("off")

    ax[3].imshow(prob_map, cmap="jet")
    ax[3].set_title("Tumor Probability Map", color="#5c7087", fontsize=9)
    ax[3].axis("off")

    ax[4].imshow(img[:, :, 0], cmap="gray")
    ax[4].imshow(pred_mask, cmap="Reds", alpha=0.5)
    ax[4].set_title(overlay_title, color="#5c7087", fontsize=9)
    ax[4].axis("off")

    ax[5].imshow(error_map, cmap="hot")
    ax[5].set_title("Segmentation Error", color="#5c7087", fontsize=9)
    ax[5].axis("off")

    fig.suptitle(suptitle, color="#1a3a58", fontsize=10)
    return fig, tumor_percent, dice, iou


def append_case_history(result):
    entry = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "modality": result.get("modality", "Unknown"),
        "label": result.get("label", "Unknown"),
        "severity": result.get("severity", "Unknown"),
    }
    st.session_state.history.insert(0, entry)
    st.session_state.history = st.session_state.history[:5]


PATIENT_CONTENT = {
    "CT": {
        "what_found": {
            "No Hemorrhage": "Your CT scan does not show signs of active intracranial bleeding.",
            "Mild Hemorrhage": "The CT suggests a small region suspicious for intracranial bleeding that needs timely specialist review.",
            "Moderate Hemorrhage": "The CT indicates a concerning bleed pattern that requires urgent clinical assessment.",
            "Severe Hemorrhage": "The CT indicates a potentially high-risk intracranial bleed and urgent emergency care is recommended.",
        },
        "plain_meaning": {
            "No Hemorrhage": "No clear bleed signal was detected by AI on this scan.",
            "Mild Hemorrhage": "This may represent a small bleed; many cases are manageable when treated early.",
            "Moderate Hemorrhage": "This finding can affect brain function and should be evaluated quickly by a neurologist/neurosurgeon.",
            "Severe Hemorrhage": "This can be serious and needs immediate emergency management.",
        },
        "what_to_do": {
            "No Hemorrhage": ["Share report with your doctor", "Continue routine follow-up", "Seek care if new neurologic symptoms appear"],
            "Mild Hemorrhage": ["Consult neurology soon", "Avoid blood thinners unless prescribed", "Monitor headache, weakness, speech changes"],
            "Moderate Hemorrhage": ["Seek urgent hospital evaluation", "Do not drive yourself", "Carry this report and scan details"],
            "Severe Hemorrhage": ["Go to emergency care immediately", "Call ambulance if symptoms worsen", "Avoid food/drink if intervention may be needed"],
        },
        "lifestyle": {
            "No Hemorrhage": ["Sleep well", "Hydrate", "Control blood pressure"],
            "Mild Hemorrhage": ["Strict rest", "No alcohol/smoking", "Avoid heavy exertion"],
            "Moderate Hemorrhage": ["Activity restriction", "Close family support", "Follow specialist instructions"],
            "Severe Hemorrhage": ["Emergency-first approach", "Do not delay treatment", "Continuous caregiver supervision"],
        },
    },
    "MRI": {
        "what_found_tumor": "The MRI segmentation model identified an abnormal tissue region consistent with a possible tumor area.",
        "plain_meaning_tumor": "This means the model sees a region that differs from normal brain tissue. Specialist review is required to confirm diagnosis and type.",
        "what_to_do_tumor": {
            "Mild": ["Book neurologist review this week", "Keep all MRI files ready", "Plan interval follow-up imaging"],
            "Moderate": ["Consult neurologist/neurosurgeon within 48-72 hours", "Track new symptoms daily", "Expect additional tests if advised"],
            "Severe": ["Seek urgent specialist care today", "Go to emergency for seizure, severe headache, weakness", "Do not delay treatment planning"],
        },
        "lifestyle_tumor": ["Prioritize sleep and hydration", "Avoid smoking and alcohol", "Use family/caregiver support", "Maintain a symptom diary"],
    },
}


def mri_severity_bucket(severity_text):
    s = (severity_text or "").lower()
    if "mild" in s or "small" in s:
        return "Mild"
    if "moderate" in s or "medium" in s:
        return "Moderate"
    if "severe" in s or "large" in s:
        return "Severe"
    return "Moderate"


def build_patient_questions(modality, severity, label):
    if modality == "CT":
        if severity == "No Hemorrhage":
            return [
                "Do I need any follow-up imaging or routine monitoring?",
                "Are there symptoms that should make me come back sooner?",
                "Do I need to change any medicines or blood pressure control?",
            ]
        if severity == "Mild Hemorrhage":
            return [
                "Should I see neurology soon or go to the emergency room?",
                "Do I need to avoid blood thinners or strenuous activity?",
                "What warning signs mean the bleeding may be getting worse?",
            ]
        if severity == "Moderate Hemorrhage":
            return [
                "Do I need urgent hospital review today?",
                "Should I bring this report and scan to a neurosurgeon?",
                "What symptoms mean I should seek emergency care immediately?",
            ]
        return [
            "Do I need emergency treatment right now?",
            "Should someone stay with me until I am reviewed?",
            "What is the fastest way to get specialist care?",
        ]

    if severity == "Mild Tumor":
        return [
            "Do I need a follow-up MRI or additional scans?",
            "Should I see a neurologist this week?",
            "What symptoms should I track at home?",
        ]
    if severity == "Moderate Tumor":
        return [
            "Should I see a neurosurgeon or oncologist within 48-72 hours?",
            "Do I need more testing to confirm the finding?",
            "Which symptoms would make this more urgent?",
        ]
    if severity == "Severe Tumor":
        return [
            "Do I need urgent specialist care today?",
            "Should I go to the emergency department if symptoms worsen?",
            "What is the next step for confirming treatment options?",
        ]
    return [
        "What does this MRI finding mean in plain language?",
        "Do I need a repeat scan or specialist review?",
        "Which symptoms should I watch for next?",
    ]


def build_patient_report_details(modality, label, severity, qa, found_text, meaning, next_steps, lifestyle):
    qa_notes = []

    if modality == "CT":
        confidence = qa.get("confidence")
        if confidence is not None:
            qa_notes.append(f"Model confidence: {confidence * 100:.2f}%")
        uncertainty = qa.get("uncertainty")
        if uncertainty:
            qa_notes.append(uncertainty)
    else:
        qa_notes.append(
            f"Slices evaluated: {qa.get('evaluated_slices', 'Unknown')} / {qa.get('total_slices', 'Unknown')}"
            f" (step={qa.get('step', 'Unknown')})"
        )
        qa_notes.append(f"Best slice index: {qa.get('best_slice_index', 'Unknown')}")
        qa_notes.append(f"Best slice tumor area: {qa.get('best_slice_tumor_pct', 0.0):.2f}%")
        if qa.get("has_gt"):
            dice = qa.get("dice")
            iou = qa.get("iou")
            if dice is not None and iou is not None:
                qa_notes.append(f"Dice: {dice:.3f} | IoU: {iou:.3f}")
        else:
            qa_notes.append("No ground-truth segmentation was provided for this MRI case.")

    summary = f"{modality} analysis identified {label} with {severity.lower()} severity."

    return {
        "summary": summary,
        "meaning": meaning,
        "qa_notes": qa_notes,
        "next_steps": next_steps,
        "lifestyle": lifestyle,
        "questions": build_patient_questions(modality, severity, label),
        "note": "This report is AI-assisted and should be reviewed by a qualified clinician before any medical decision.",
        "found": found_text,
    }


def severity_badge_class(severity):
    s = (severity or "").lower()
    if "severe" in s:
        return "severity-severe"
    if "mild" in s:
        return "severity-mild"
    return "severity-moderate"


def severity_risk_pct(severity):
    s = (severity or "").lower()
    if "no hemorrhage" in s:
        return 12
    if "mild" in s:
        return 34
    if "moderate" in s:
        return 66
    if "severe" in s:
        return 92
    return 58


def patient_initials(name):
    value = (name or "").strip()
    if not value:
        return "P"
    parts = [part for part in value.split() if part]
    letters = "".join(part[0] for part in parts[:2]).upper()
    return letters or "P"


def build_patient_faq(modality, severity):
    if modality == "CT":
        if severity == "No Hemorrhage":
            return [
                ("What does this mean?", "The model did not find a bleed pattern on the scan."),
                ("How serious is it?", "This result is usually lower risk, but symptoms still matter."),
                ("What should I do next?", "Share the report with your doctor and watch for new symptoms."),
            ]
        if severity == "Mild Hemorrhage":
            return [
                ("What does this mean?", "A small bleeding pattern may be present and needs doctor review."),
                ("How serious is it?", "It may not be an emergency, but it should be checked soon."),
                ("What should I do next?", "Contact a neurologist or emergency service depending on symptoms."),
            ]
        if severity == "Moderate Hemorrhage":
            return [
                ("What does this mean?", "The scan suggests a more concerning bleed pattern."),
                ("How serious is it?", "This is more urgent and should be assessed quickly."),
                ("What should I do next?", "Seek urgent hospital evaluation and bring the scan/report."),
            ]
        return [
            ("What does this mean?", "The scan suggests a high-risk bleed pattern."),
            ("How serious is it?", "This is urgent and needs immediate medical attention."),
            ("What should I do next?", "Go to emergency care right away."),
        ]

    if severity == "Mild Tumor":
        return [
            ("What does this mean?", "The model found a small abnormal area that needs follow-up."),
            ("How serious is it?", "This often needs specialist review, but it may not be immediately dangerous."),
            ("What should I do next?", "Book neurology follow-up and keep all MRI files ready."),
        ]
    if severity == "Moderate Tumor":
        return [
            ("What does this mean?", "The model sees a more noticeable abnormal region."),
            ("How serious is it?", "This should be reviewed quickly by a specialist."),
            ("What should I do next?", "Arrange specialist review within 48-72 hours."),
        ]
    if severity == "Severe Tumor":
        return [
            ("What does this mean?", "The abnormal area appears significant on the scan."),
            ("How serious is it?", "This is a high-priority finding and needs urgent care."),
            ("What should I do next?", "Seek specialist or emergency review today."),
        ]
    return [
        ("What does this mean?", "The MRI finding needs a specialist to interpret it in context."),
        ("How serious is it?", "The exact urgency depends on symptoms and clinical history."),
        ("What should I do next?", "Discuss the result with your clinician and follow their advice."),
    ]


# ================= CSS =================
st.markdown(
    """
<style>
.block-container {padding-top:2rem;}
.card {
    background: white;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 18px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.05);
}
h1,h2,h3 {font-family: Inter;}
.stButton > button {
    border-radius: 10px;
    background: #0d6efd;
    color: white;
    font-weight: 600;
}
.report-note {
    border-left: 4px solid #0d6efd;
    background: #f1f7ff;
    padding: 10px 12px;
    border-radius: 8px;
    color: #244763;
}
.report-page {
    background: linear-gradient(180deg, #f6f9fc 0%, #edf3f8 100%);
    border: 1px solid #dce7f2;
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 16px 40px rgba(14, 34, 58, 0.08);
}
.report-hero {
    display: flex;
    justify-content: space-between;
    gap: 18px;
    align-items: flex-start;
    padding: 18px 20px;
    border-radius: 20px;
    background: linear-gradient(135deg, #0f2740 0%, #183c5f 50%, #215d86 100%);
    color: #eef6fb;
    margin-bottom: 18px;
}
.report-hero h1 {
    margin: 0 0 6px;
    font-size: 2rem;
    letter-spacing: -0.03em;
}
.report-hero p {
    margin: 0;
    color: rgba(238, 246, 251, 0.88);
}
.report-hero-meta {
    text-align: right;
    min-width: 180px;
}
.report-pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
}
.report-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.16);
    color: #ffffff;
    font-size: 0.82rem;
}
.patient-card,
.summary-card,
.media-card,
.recommend-card,
.explain-card,
.faq-card,
.download-card {
    background: #ffffff;
    border: 1px solid #d8e3ee;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 8px 22px rgba(19, 42, 68, 0.06);
}
.section-title {
    margin: 0 0 10px;
    color: #15324c;
    font-size: 1.1rem;
}
.section-lead {
    margin: 0 0 12px;
    color: #587086;
    line-height: 1.6;
}
.patient-meta {
    background: #f7fbff;
    border: 1px solid #d8e3ee;
    border-radius: 14px;
    padding: 12px;
}
.patient-meta .k {
    font-size: 0.78rem;
    color: #6a7f92;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.patient-meta .v {
    font-size: 1rem;
    color: #17334e;
    font-weight: 600;
}
.avatar-box {
    width: 100%;
    min-height: 210px;
    border-radius: 18px;
    border: 1px dashed #b7c7d9;
    background: linear-gradient(180deg, #f8fbff, #eef5fb);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    padding: 16px;
}
.avatar-circle {
    width: 96px;
    height: 96px;
    border-radius: 50%;
    display: grid;
    place-items: center;
    background: linear-gradient(135deg, #0f2740, #215d86);
    color: white;
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 10px;
    box-shadow: 0 10px 24px rgba(15, 39, 64, 0.22);
}
.summary-hero {
    display: flex;
    justify-content: space-between;
    gap: 14px;
    align-items: flex-start;
}
.diagnosis-text {
    font-size: 1.55rem;
    font-weight: 700;
    color: #14324c;
    margin: 4px 0 6px;
    line-height: 1.25;
}
.severity-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border-radius: 999px;
    padding: 8px 14px;
    font-size: 0.85rem;
    font-weight: 700;
}
.severity-mild {
    background: #e8f8ef;
    color: #1f7a44;
}
.severity-moderate {
    background: #fff5df;
    color: #8a6410;
}
.severity-severe {
    background: #fde8e6;
    color: #b13b2e;
}
.risk-meter {
    height: 12px;
    border-radius: 999px;
    background: #e4ecf4;
    overflow: hidden;
    margin-top: 10px;
}
.risk-meter-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #1f9d62, #f2b73d, #d94e41);
}
.mini-panel {
    background: #f8fbff;
    border: 1px solid #d8e3ee;
    border-radius: 14px;
    padding: 14px;
}
.mini-panel ul {
    margin: 8px 0 0;
    padding-left: 18px;
}
.media-frame {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid #d8e3ee;
    background: #f8fbff;
}
.media-note {
    font-size: 0.95rem;
    color: #4f677d;
    line-height: 1.6;
}
.faq-item {
    border: 1px solid #d8e3ee;
    border-radius: 14px;
    background: #f8fbff;
    padding: 12px 14px;
    margin-bottom: 10px;
}
.faq-item h5 {
    margin: 0 0 4px;
    color: #18334d;
    font-size: 0.96rem;
}
.faq-item p {
    margin: 0;
    color: #516579;
    line-height: 1.55;
}
.download-hint {
    color: #5b7084;
    font-size: 0.9rem;
}
@media (max-width: 900px) {
    .report-hero,
    .summary-hero {
        grid-template-columns: 1fr;
        display: grid;
    }
    .report-hero-meta {
        text-align: left;
        min-width: 0;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# ================= SIDEBAR =================
with st.sidebar:
    st.title("🧠 NeuroVision AI")
    st.caption("Clinical Decision Support System")

    st.markdown("### 🧭 Workflow")
    st.markdown(
        """
    1. Upload Scan  
    2. Detect Modality  
    3. Run AI Analysis  
    4. View Clinical Results  
    5. Generate Patient Report  
    """
    )

    st.markdown("---")

    if st.button("📊 Clinical Dashboard"):
        st.session_state.page = "clinician"

    if st.session_state.result:
        st.markdown("### 🔒 Patient Report")
        if st.button("Unlock Patient Report"):
            st.session_state.page = "patient"

    st.markdown("---")

    with st.expander("Recent Cases"):
        if not st.session_state.history:
            st.caption("No cases yet")
        for h in st.session_state.history:
            st.write(f"{h['modality']} | {h['severity']}")


# ================= CLINICIAN =================
if st.session_state.page == "clinician":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title("Brain Imaging Analysis Studio")
    st.caption("Upload scan, run AI analysis, and generate patient report")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload Scan", type=["dcm", "nii", "nii.gz"], accept_multiple_files=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded:
        st.session_state.uploaded_files = uploaded

    if uploaded:
        uploaded = st.session_state.uploaded_files
        names = [f.name for f in uploaded]

        has_ct = any(n.lower().endswith(".dcm") for n in names)
        has_mri = any(n.lower().endswith(".nii") or n.lower().endswith(".nii.gz") for n in names)

        if has_ct and has_mri:
            st.error("Upload either CT (.dcm) or MRI (.nii/.nii.gz) files, not both together.")
            st.stop()

        # ================= CT =================
        if has_ct:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Detected Modality")
            st.success("CT Scan")
            st.markdown("</div>", unsafe_allow_html=True)

            file = uploaded[0]
            path = save_uploaded_temp(file)

            img = preprocess_ct(path)
            if img is None:
                st.error("Could not preprocess this CT file.")
                st.stop()

            label, prob = predict_ct(img)
            severity = ct_severity(prob)
            overlay = get_gradcam_overlay(img, load_ct_model())

            uncertainty = "High certainty"
            if 0.35 <= prob <= 0.45 or 0.55 <= prob <= 0.65:
                uncertainty = "Borderline confidence - clinical review strongly recommended"

            st.session_state.result = {
                "modality": "CT",
                "label": label,
                "prob": prob,
                "severity": severity,
                "overlay": overlay,
                "qa": {
                    "confidence": float(prob),
                    "uncertainty": uncertainty,
                },
            }
            append_case_history(st.session_state.result)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Prediction")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Prediction", label)

                with col2:
                    st.metric("Confidence", f"{prob*100:.1f}%")

                with col3:
                    st.metric("Severity", severity)

                st.progress(prob)
                st.markdown("</div>", unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("Severity")
                    st.write(severity)
                    st.caption(uncertainty)
                    st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Explainability")
            st.image(overlay)
            st.caption("Highlighted regions influence prediction")
            st.markdown("</div>", unsafe_allow_html=True)

        # ================= MRI =================
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Detected Modality")
            st.success("MRI Scan")
            st.markdown("</div>", unsafe_allow_html=True)

            picked = {}
            for f in uploaded:
                modality = infer_mri_modality(f.name)
                if modality and modality not in picked:
                    picked[modality] = save_uploaded_temp(f)

            missing = [m for m in ["flair", "t1ce", "t2"] if m not in picked]
            if missing:
                st.error("Could not find required MRI files: " + ", ".join(missing))
                st.stop()

            flair = nib.load(picked["flair"]).get_fdata()
            t1ce = nib.load(picked["t1ce"]).get_fdata()
            t2 = nib.load(picked["t2"]).get_fdata()
            seg = nib.load(picked["seg"]).get_fdata() if "seg" in picked else None

            if flair.shape[2] < 5:
                st.error("MRI volume has too few slices for reliable segmentation.")
                st.stop()

            depth = flair.shape[2]
            max_eval_slices = 48
            step = max(1, int(np.ceil(depth / max_eval_slices)))
            indices = list(range(0, depth, step))
            center = depth // 2
            if indices[-1] != depth - 1:
                indices.append(depth - 1)
            if center not in indices:
                indices.append(center)
            indices = sorted(set(indices))

            slice_runs = []
            for idx in indices:
                img = preprocess_slice(flair[:, :, idx], t1ce[:, :, idx], t2[:, :, idx])
                mask, prob_map = predict_mri(img)
                mask = (mask > 0.5).astype(np.uint8)

                gt = None
                if seg is not None:
                    gt = (seg[:, :, idx] > 0).astype(np.float32)
                    gt = cv2.resize(gt, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)

                fig, tp, dice, iou = render_mri_panel(img, mask, prob_map, gt)
                overlay_img = get_mri_overlay(img, mask)
                slice_runs.append((fig, tp, dice, iou, overlay_img, mask, img, prob_map, gt, idx))

            best_idx = int(np.argmax([x[1] for x in slice_runs]))
            fig, tp, dice, iou, overlay, best_mask, best_img, best_prob_map, best_gt, best_slice = slice_runs[best_idx]

            severity = mri_severity(best_mask)
            tumor_ratio = (float(np.sum(best_mask)) / float(best_mask.size)) if best_mask.size else 0.0

            st.session_state.result = {
                "modality": "MRI",
                "label": "Tumor Segmentation",
                "severity": severity,
                "overlay": overlay,
                "mri_panel": {
                    "img": best_img,
                    "mask": best_mask,
                    "prob_map": best_prob_map,
                    "gt": best_gt,
                },
                "qa": {
                    "evaluated_slices": len(indices),
                    "total_slices": depth,
                    "step": step,
                    "best_slice_index": int(best_slice),
                    "best_slice_tumor_pct": float(tp),
                    "tumor_ratio": float(tumor_ratio),
                    "dice": None if dice is None else float(dice),
                    "iou": None if iou is None else float(iou),
                    "has_gt": bool(seg is not None),
                },
            }
            append_case_history(st.session_state.result)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("MRI Segmentation")
            st.caption(f"Evaluated {len(indices)} of {depth} slices (step={step})")
            st.pyplot(fig, clear_figure=True, width="stretch")
            plt.close(fig)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Severity")
            st.write(severity)
            st.caption(f"Severity basis: tumor ratio = {tumor_ratio:.4f} ({tumor_ratio*100:.2f}%)")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Clinical Summary")
        st.write(f"Finding: {st.session_state.result['label']}")
        st.write(f"Severity: {st.session_state.result['severity']}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
### 🧾 Quick Summary
- AI detected abnormal region  
- Severity estimated based on model output  
- Visual explanation available above  
"""
        )

        if st.button("Generate Patient Report"):
            st.session_state.page = "patient"
            st.rerun()

    elif st.session_state.result:
        r = st.session_state.result

        if r.get("modality") == "MRI" and r.get("mri_panel") is not None:
            panel = r["mri_panel"]
            saved_fig, _, _, _ = render_mri_panel(panel["img"], panel["mask"], panel["prob_map"], panel.get("gt"))

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("MRI Segmentation")
            st.pyplot(saved_fig, clear_figure=True, width="stretch")
            plt.close(saved_fig)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Severity")
            st.write(r.get("severity", "Unknown"))
            st.markdown("</div>", unsafe_allow_html=True)
        elif r.get("modality") == "CT":
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Detected Modality")
            st.success("CT Scan")
            st.markdown("</div>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Prediction")
                st.write(r.get("label", "Unknown"))
                if r.get("prob") is not None:
                    st.write(f"{r['prob']*100:.2f}% confidence")
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Severity")
                st.write(r.get("severity", "Unknown"))
                st.markdown("</div>", unsafe_allow_html=True)

            if r.get("overlay") is not None:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Explainability")
                st.image(r["overlay"])
                st.caption("Highlighted regions influence prediction")
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Clinical Summary")
        st.write(f"Finding: {r.get('label', 'Unknown')}")
        st.write(f"Severity: {r.get('severity', 'Unknown')}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
### 🧾 Quick Summary
- AI detected abnormal region  
- Severity estimated based on model output  
- Visual explanation available above  
"""
        )

        if st.button("Generate Patient Report", key="generate_saved_report"):
            st.session_state.page = "patient"
            st.rerun()


# ================= PATIENT =================
else:
    if not st.session_state.result:
        st.warning("No saved analysis found. Please run an analysis first.")
        if st.button("Back to Analysis", key="back_no_result"):
            st.session_state.page = "clinician"
            st.rerun()
        st.stop()

    r = st.session_state.result

    modality = r.get("modality", "Unknown")
    label = r.get("label", "Unknown")
    severity = r.get("severity", "Unknown")
    qa = r.get("qa", {})

    if modality == "CT":
        found_text = PATIENT_CONTENT["CT"]["what_found"].get(severity, "AI detected a finding that needs doctor review.")
        meaning = PATIENT_CONTENT["CT"]["plain_meaning"].get(severity, "Please consult your doctor for interpretation.")
        next_steps = PATIENT_CONTENT["CT"]["what_to_do"].get(severity, ["Consult your doctor with this report"])
        lifestyle = PATIENT_CONTENT["CT"]["lifestyle"].get(severity, ["Follow your doctor's advice"])
    else:
        found_text = PATIENT_CONTENT["MRI"]["what_found_tumor"]
        meaning = PATIENT_CONTENT["MRI"]["plain_meaning_tumor"]
        sev_key = mri_severity_bucket(severity)
        next_steps = PATIENT_CONTENT["MRI"]["what_to_do_tumor"].get(sev_key, ["Consult your neurologist"])
        lifestyle = PATIENT_CONTENT["MRI"]["lifestyle_tumor"]
        visual_note = "Highlighted areas show where the model detected abnormal tissue probability."

    patient_name = st.session_state.get("patient_name", "").strip() or "Patient Name"
    patient_id = st.session_state.get("patient_id", "").strip() or "Not provided"
    patient_age = st.session_state.get("patient_age", 0)
    patient_gender = st.session_state.get("patient_gender", "Prefer not to say")
    patient_initial = patient_initials(patient_name if patient_name != "Patient Name" else "Patient")

    confidence_value = r.get("prob")
    if confidence_value is None and qa.get("tumor_ratio") is not None:
        confidence_value = qa.get("tumor_ratio")

    severity_class = severity_badge_class(severity)
    risk_pct = severity_risk_pct(severity)

    if modality == "CT":
        summary_line = found_text
        explanation_line = meaning
        visual_note = "Highlighted regions indicate where the CT attention map focused during prediction."
    else:
        summary_line = f"The segmentation model found an abnormal region consistent with tumor-like tissue."
        explanation_line = meaning

    report_details = build_patient_report_details(
        modality,
        label,
        severity,
        qa,
        found_text,
        meaning,
        next_steps,
        lifestyle,
    )

    report_text = generate_report(
        modality,
        label,
        severity,
        r.get("prob", None),
        details=report_details,
    )

    faq_items = build_patient_faq(modality, severity)

    st.markdown('<div class="report-page">', unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="report-hero">
    <div>
        <h1>Patient Report</h1>
        <p>NeuroVision AI patient summary and clinical guidance.</p>
        <div class="report-pill-row">
            <span class="report-pill">Hospital / System: NeuroVision AI V2</span>
            <span class="report-pill">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
            <span class="report-pill">Scan Type: {modality}</span>
        </div>
    </div>
    <div class="report-hero-meta">
        <div class="severity-badge {severity_class}">{severity}</div>
        <div style="margin-top:10px;color:rgba(238,246,251,0.9);font-size:0.92rem;">Risk level: {risk_pct}%</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    patient_col, avatar_col = st.columns([1.35, 0.65])
    with patient_col:
        st.markdown('<div class="patient-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Patient Information</div>', unsafe_allow_html=True)
        info_cols = st.columns(3)
        with info_cols[0]:
            st.session_state.patient_name = st.text_input("Patient name", value=st.session_state.patient_name, placeholder="Enter patient name")
            st.session_state.patient_id = st.text_input("Patient ID", value=st.session_state.patient_id, placeholder="Enter patient ID")
        with info_cols[1]:
            st.session_state.patient_age = st.number_input("Age", min_value=0, max_value=120, value=int(st.session_state.patient_age or 0), step=1)
            st.session_state.patient_gender = st.selectbox(
                "Gender",
                ["Prefer not to say", "Female", "Male", "Other"],
                index=["Prefer not to say", "Female", "Male", "Other"].index(st.session_state.patient_gender)
                if st.session_state.patient_gender in ["Prefer not to say", "Female", "Male", "Other"]
                else 0,
            )
        with info_cols[2]:
            st.markdown(
                f"""
<div class="patient-meta" style="height:100%;display:flex;flex-direction:column;justify-content:center;">
    <div class="k">Exam type</div>
    <div class="v">{modality}</div>
    <div style="margin-top:8px;color:#5b7084;font-size:0.9rem;">AI-assisted patient summary</div>
</div>
""",
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with avatar_col:
        st.markdown(
            f"""
<div class="patient-card">
    <div class="section-title">Profile</div>
    <div class="avatar-box">
        <div class="avatar-circle">{patient_initial}</div>
        <div style="font-weight:700;color:#18334d;">{patient_name or 'Patient'}</div>
        <div style="color:#60778b;margin-top:4px;">Profile image optional</div>
        <div style="color:#60778b;font-size:0.9rem;">No image uploaded</div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="summary-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Diagnosis Summary</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="summary-hero">
    <div style="flex:1;">
        <div class="diagnosis-text">{label}</div>
        <div class="severity-badge {severity_class}">{severity} Severity</div>
        <div style="margin-top:10px;color:#516579;line-height:1.65;">{summary_line}</div>
        <div class="risk-meter"><div class="risk-meter-fill" style="width:{risk_pct}%;"></div></div>
        <div style="display:flex;justify-content:space-between;margin-top:6px;color:#62778a;font-size:0.84rem;">
            <span>Low</span><span>Moderate</span><span>High</span>
        </div>
    </div>
    <div class="mini-panel" style="min-width:220px;">
        <div class="k" style="font-size:0.78rem;color:#6a7f92;text-transform:uppercase;letter-spacing:0.04em;">Confidence</div>
        <div style="font-size:1.8rem;font-weight:800;color:#17334e;margin-top:4px;">{(confidence_value * 100):.0f}%</div>
        <div style="color:#5a7084;line-height:1.55;margin-top:4px;">{report_details['note']}</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    media_col, rec_col = st.columns([1.2, 0.8])
    with media_col:
        st.markdown('<div class="media-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Visual Explanation</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-lead">The visual panel highlights the region that influenced the model. Use it as supporting evidence, not a standalone diagnosis.</div>', unsafe_allow_html=True)
        left_media, right_media = st.columns(2)
        with left_media:
            st.markdown('<div class="media-frame">', unsafe_allow_html=True)
            if r.get("overlay") is not None:
                st.image(r["overlay"], use_container_width=True)
            else:
                st.markdown('<div style="padding:28px 16px;text-align:center;color:#5b7084;">No visual overlay available for this case.</div>', unsafe_allow_html=True)
            st.markdown(
                f"<div style='padding:0 12px 12px 12px;color:#5b7084;line-height:1.6;'>Highlighted regions indicate model attention for {modality} interpretation.</div>",
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)
        with right_media:
            st.markdown('<div class="media-frame">', unsafe_allow_html=True)
            st.markdown(f"<div style='padding:14px 14px 6px 14px;'><div class='section-title' style='margin-bottom:8px;'>Clinical Context</div><div class='media-note'>{explanation_line}</div></div>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='padding:0 14px 14px 14px;'><div class='mini-panel' style='margin:0;'><div class='k' style='font-size:0.78rem;color:#6a7f92;text-transform:uppercase;letter-spacing:0.04em;'>Short note</div><div style='margin-top:6px;color:#17334e;font-weight:600;'>{found_text}</div></div></div>",
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with rec_col:
        st.markdown('<div class="recommend-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Recommendations</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-lead">Next steps and everyday guidance are separated so the report stays easy to scan.</div>', unsafe_allow_html=True)
        st.markdown("<strong>Medical suggestions</strong>", unsafe_allow_html=True)
        for item in next_steps:
            st.markdown(f"- {item}")
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        st.markdown("<strong>Lifestyle tips</strong>", unsafe_allow_html=True)
        for item in lifestyle:
            st.markdown(f"- {item}")
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
<div class="mini-panel">
    <div class="k" style="font-size:0.78rem;color:#6a7f92;text-transform:uppercase;letter-spacing:0.04em;">Doctor note</div>
    <div style="margin-top:6px;color:#17334e;line-height:1.6;">AI output should be correlated with symptoms, history, and specialist review.</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    explain_col, faq_col = st.columns([1.05, 0.95])
    with explain_col:
        st.markdown('<div class="explain-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Detailed Explanation</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
<div class="section-lead">{explanation_line}</div>
<div class="mini-panel">
    <strong>What it means:</strong>
    <div style="margin-top:6px;color:#516579;line-height:1.6;">{found_text}</div>
</div>
<div style="height:10px;"></div>
<div class="mini-panel">
    <strong>Why this matters:</strong>
    <div style="margin-top:6px;color:#516579;line-height:1.6;">{report_details['summary']}</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with faq_col:
        st.markdown('<div class="faq-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">FAQ</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-lead">Short answers only. This section is for quick patient understanding, not a repeat of the summary.</div>', unsafe_allow_html=True)
        for question, answer in faq_items:
            st.markdown(
                f"""

            st.markdown("### 📊 Tumor Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Tumor Area %", f"{tp:.2f}%")

            with col2:
                st.metric("Severity", severity)

            st.progress(tp / 100)
<div class="faq-item">
    <h5>{question}</h5>
    <p>{answer}</p>
</div>
""",
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="download-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Download / Export</div>', unsafe_allow_html=True)
    st.markdown('<div class="download-hint">Use the export below for sharing with a clinician or saving a copy.</div>', unsafe_allow_html=True)
    download_cols = st.columns([0.45, 0.55])
    with download_cols[0]:
        st.download_button(
            "Download Full Report",
            data=report_text,
            file_name="patient_report.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with download_cols[1]:
        st.markdown('<div class="mini-panel">Print: use your browser print option for a paper copy. Share: forward the downloaded report to your clinician.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="report-note" style="margin-top:14px;">This is an AI-assisted decision support report and is not a final diagnosis. Please consult a qualified neurologist or radiologist for medical decisions.</div>', unsafe_allow_html=True)

    if st.button("Back to Analysis"):
        st.session_state.page = "clinician"
        st.rerun()
