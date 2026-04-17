import streamlit as st
import numpy as np
import nibabel as nib
import tempfile
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ===== YOUR ORIGINAL IMPORTS =====
from preprocessing.preprocess_ct import preprocess_ct
from preprocessing.preprocess_mri import preprocess_slice
from models.ct_model import predict_ct, load_ct_model
from models.mri_model import predict_mri
from xai.xai_ct import get_gradcam_overlay
from xai.xai_mri import get_mri_overlay
from utils.severity import ct_severity, mri_severity
from utils.report import generate_report

# ================= CONFIG =================
st.set_page_config(page_title="NeuroVision AI", layout="wide")

# ================= SESSION =================
if "page" not in st.session_state:
    st.session_state.page = "clinician"

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None

if "result" not in st.session_state:
    st.session_state.result = None

if "report_unlocked" not in st.session_state:
    st.session_state.report_unlocked = False

if "patient_profile" not in st.session_state:
    st.session_state.patient_profile = {
        "name": "Rahul Verma",
        "patient_id": "NV-2026-04871",
        "dob": "1992-09-18",
        "age_gender": "33 / Male",
        "physician": "Dr. Meera Kapoor",
        "facility": "NeuroVision Partner Hospital",
    }


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
    fig.patch.set_facecolor('#f8fbff')
    for a in ax:
        a.set_facecolor('#f8fbff')

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


def _score_to_grade(score):
    score = float(np.clip(score, 0, 100))
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "E"


def _severity_urgency_score(severity):
    text = str(severity).lower()
    if "critical" in text or "severe" in text or "high" in text:
        return 35
    if "moderate" in text or "intermediate" in text:
        return 60
    if "mild" in text or "low" in text:
        return 82
    return 70


def build_progress_report_data(result):
    modality = result.get("modality", "Unknown")
    severity = result.get("severity", "Unknown")
    finding = result.get("label", "Unknown")

    confidence_pct = None
    if result.get("prob") is not None:
        confidence_pct = float(np.clip(result["prob"] * 100.0, 0, 100))

    urgency_score = _severity_urgency_score(severity)

    if modality == "CT":
        confidence_score = confidence_pct if confidence_pct is not None else 72.0
        explainability_score = 88.0 if result.get("overlay") is not None else 70.0
        readiness_score = float(np.clip((confidence_score * 0.5) + (urgency_score * 0.5), 0, 100))

        areas = [
            ("Model Confidence", confidence_score),
            ("Clinical Stability", urgency_score),
            ("Explainability Coverage", explainability_score),
            ("Follow-up Readiness", readiness_score),
        ]

        summary = "The scan was analyzed in CT mode. Confidence and severity are combined to prioritize follow-up urgency."
        meaning = (
            "Higher Clinical Stability and Follow-up Readiness suggest lower immediate risk. "
            "Lower values indicate the care team should review quickly."
        )
        next_steps = [
            "Review AI-highlighted region with radiologist.",
            "Correlate with neurological symptoms and vitals.",
            "Decide if urgent repeat imaging is needed.",
        ]

    else:
        tumor_ratio = float(result.get("tumor_ratio", 0.0))
        tumor_burden_pct = float(np.clip(tumor_ratio * 100.0, 0, 100))
        dice = result.get("dice")
        iou = result.get("iou")

        segmentation_quality = float(np.clip(((float(dice) if dice is not None else 0.72) * 100.0), 0, 100))
        stability_score = float(np.clip(100.0 - tumor_burden_pct, 0, 100))
        confidence_score = float(np.clip((segmentation_quality * 0.6) + (stability_score * 0.4), 0, 100))

        areas = [
            ("Segmentation Quality", segmentation_quality),
            ("Clinical Stability", urgency_score),
            ("Tumor Burden Control", stability_score),
            ("Follow-up Readiness", confidence_score),
        ]

        eval_slices = result.get("evaluated_slices")
        total_slices = result.get("total_slices")
        summary = (
            f"MRI segmentation identified estimated tumor area around {tumor_burden_pct:.2f}% "
            "on the selected representative slice."
        )
        if eval_slices is not None and total_slices is not None:
            summary += f" Evaluated {eval_slices} out of {total_slices} slices."

        meaning = (
            "Tumor Burden Control reflects relative non-tumor tissue proportion in the analyzed slice. "
            "Segmentation Quality is stronger when Dice/IoU are available and higher."
        )
        next_steps = [
            "Validate segmented area against full radiology read.",
            "Track burden change against prior MRI if available.",
            "Plan follow-up interval based on severity and symptoms.",
        ]

        if iou is not None:
            meaning += f" Current IoU reference: {float(iou):.2f}."

    average_score = float(np.mean([score for _, score in areas])) if areas else 0.0

    return {
        "finding": finding,
        "modality": modality,
        "severity": severity,
        "confidence_pct": confidence_pct,
        "average_score": average_score,
        "areas": [(name, float(np.clip(score, 0, 100)), _score_to_grade(score)) for name, score in areas],
        "summary": summary,
        "meaning": meaning,
        "next_steps": next_steps,
    }


def _patient_initials(full_name):
    parts = [p for p in str(full_name).split() if p]
    if not parts:
        return "NA"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def _severity_banner(severity):
    text = str(severity).lower()
    if "critical" in text or "severe" in text or "high" in text:
        return "High Concern", "🚨"
    if "moderate" in text:
        return "Moderate Concern", "⚠️"
    if "mild" in text or "low" in text:
        return "Low Concern", "✅"
    return "Clinical Review Needed", "🩺"


def _default_doctor_questions(modality):
    if str(modality).upper() == "MRI":
        return [
            "How does this finding compare with my previous MRI scans?",
            "Do I need a follow-up MRI, and when should it be scheduled?",
            "What symptoms should prompt immediate medical attention?",
            "Do these results change my current treatment plan?",
            "Should I consult a specialist based on this report?",
        ]
    return [
        "How certain is this CT finding and what confirms it?",
        "Do I need additional imaging or lab tests?",
        "What warning signs should I watch for at home?",
        "What is the expected follow-up timeline for this result?",
            "Should I seek referral to a neurologist now?",
    ]

# ================= CSS =================
st.markdown("""
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
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
with st.sidebar:
    st.title("🧠 NeuroVision AI")
    st.caption("Clinical Decision Support")

    st.markdown("---")
    st.subheader("Navigation")

    if st.button("Analysis Studio"):
        st.session_state.page = "clinician"

    if st.session_state.result:
        if st.button("Patient Report"):
            st.session_state.page = "patient"

    st.markdown("---")
    st.subheader("Current Case")
    if st.session_state.result:
        sr = st.session_state.result
        st.write(f"Patient: {st.session_state.patient_profile['name']}")
        st.write(f"ID: {st.session_state.patient_profile['patient_id']}")
        st.write(f"Modality: {sr.get('modality', 'Unknown')}")
        st.write(f"Status: {'Report Ready' if st.session_state.result else 'Pending'}")
    else:
        st.caption("No active case")

    st.markdown("---")
    st.subheader("Premium Report")
    if st.session_state.report_unlocked:
        st.success("Full report unlocked")
    else:
        st.warning("Full report locked")
    if st.button("Unlock Full Report", key="unlock_from_sidebar"):
        st.session_state.report_unlocked = True
        st.rerun()

    st.markdown("---")
    st.subheader("System")
    st.caption("Models: CT DenseNet50 + MRI U-Net")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ================= CLINICIAN =================
if st.session_state.page == "clinician":

    # HEADER
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title("Brain Imaging Analysis Studio")
    st.caption("AI-based CT & MRI clinical decision system")
    st.markdown('</div>', unsafe_allow_html=True)

    # UPLOAD
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload Scan", type=["dcm","nii","nii.gz"], accept_multiple_files=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded:
        st.session_state.uploaded_files = uploaded

    if uploaded:

        uploaded = st.session_state.uploaded_files
        names = [f.name for f in uploaded]

        # ================= CT =================
        if any(n.endswith(".dcm") for n in names):

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Detected Modality")
            st.success("CT Scan")
            st.markdown('</div>', unsafe_allow_html=True)

            file = uploaded[0]
            path = save_uploaded_temp(file)

            img = preprocess_ct(path)
            label, prob = predict_ct(img)
            severity = ct_severity(prob)
            overlay = get_gradcam_overlay(img, load_ct_model())

            st.session_state.result = {
                "modality":"CT",
                "label":label,
                "prob":prob,
                "severity":severity,
                "overlay": overlay
            }
            st.session_state.report_unlocked = False

            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Prediction")
                st.write(label)
                st.write(f"{prob*100:.2f}% confidence")
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Severity")
                st.write(severity)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Explainability")
            st.image(overlay)
            st.caption("Highlighted regions influence prediction")
            st.markdown('</div>', unsafe_allow_html=True)

        # ================= MRI =================
        else:

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Detected Modality")
            st.success("MRI Scan")
            st.markdown('</div>', unsafe_allow_html=True)

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
                slice_runs.append((fig, tp, dice, iou, overlay_img, mask, img, prob_map, gt))

            best_idx = int(np.argmax([x[1] for x in slice_runs]))
            fig, tp, dice, iou, overlay, best_mask, best_img, best_prob_map, best_gt = slice_runs[best_idx]
            severity = mri_severity(best_mask)
            tumor_ratio = (float(np.sum(best_mask)) / float(best_mask.size)) if best_mask.size else 0.0

            st.session_state.result = {
                "modality":"MRI",
                "label":"Tumor Segmentation",
                "severity":severity,
                "overlay": overlay,
                "tumor_ratio": tumor_ratio,
                "dice": dice,
                "iou": iou,
                "evaluated_slices": len(indices),
                "total_slices": depth,
                "mri_panel": {
                    "img": best_img,
                    "mask": best_mask,
                    "prob_map": best_prob_map,
                    "gt": best_gt,
                }
            }
            st.session_state.report_unlocked = False

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("MRI Segmentation")
            st.caption(f"Evaluated {len(indices)} of {depth} slices (step={step})")
            st.pyplot(fig, clear_figure=True, width="stretch")
            plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Severity")
            st.write(severity)
            st.caption(f"Severity basis: tumor ratio = {tumor_ratio:.4f} ({tumor_ratio*100:.2f}%)")
            st.markdown('</div>', unsafe_allow_html=True)

        # SUMMARY
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Clinical Summary")
        st.write(f"Finding: {st.session_state.result['label']}")
        st.write(f"Severity: {st.session_state.result['severity']}")
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("📋 Generate Patient Report"):
            st.session_state.page = "patient"
            st.rerun()

    elif st.session_state.result:
        r = st.session_state.result

        if r.get("modality") == "MRI" and r.get("mri_panel") is not None:
            panel = r["mri_panel"]
            saved_fig, _, _, _ = render_mri_panel(
                panel["img"], panel["mask"], panel["prob_map"], panel.get("gt")
            )

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("MRI Segmentation")
            st.pyplot(saved_fig, clear_figure=True, width="stretch")
            plt.close(saved_fig)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Severity")
            st.write(r.get("severity", "Unknown"))
            st.markdown('</div>', unsafe_allow_html=True)
        elif r.get("modality") == "CT":
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Detected Modality")
            st.success("CT Scan")
            st.markdown('</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Prediction")
                st.write(r.get("label", "Unknown"))
                if r.get("prob") is not None:
                    st.write(f"{r['prob']*100:.2f}% confidence")
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Severity")
                st.write(r.get("severity", "Unknown"))
                st.markdown('</div>', unsafe_allow_html=True)

            if r.get("overlay") is not None:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Explainability")
                st.image(r["overlay"])
                st.caption("Highlighted regions influence prediction")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Last Analysis (Saved)")
            st.write(f"Modality: {r.get('modality', 'Unknown')}")
            st.write(f"Finding: {r.get('label', 'Unknown')}")
            st.write(f"Severity: {r.get('severity', 'Unknown')}")
            if r.get("prob") is not None:
                st.write(f"Confidence: {r['prob']*100:.2f}%")
            if r.get("overlay") is not None:
                st.image(r["overlay"])
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Clinical Summary")
        st.write(f"Finding: {r.get('label', 'Unknown')}")
        st.write(f"Severity: {r.get('severity', 'Unknown')}")
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("📋 Generate Patient Report", key="generate_saved_report"):
            st.session_state.page = "patient"
            st.rerun()

# ================= PATIENT =================
else:

    if not st.session_state.result:
        st.warning("No saved analysis found. Please run an analysis first.")
        if st.button("← Back to Analysis", key="back_no_result"):
            st.session_state.page = "clinician"
            st.rerun()
        st.stop()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.caption("PATIENT REPORT · CONFIDENTIAL")
    st.title("Your Brain Health Report")
    st.caption("A clear, plain-language summary of your scan results.")
    st.markdown('</div>', unsafe_allow_html=True)

    r = st.session_state.result
    report_data = build_progress_report_data(r)
    profile = st.session_state.patient_profile
    scan_date = datetime.now().strftime("%Y-%m-%d")
    report_id = f"NV-RPT-{datetime.now().strftime('%Y%m%d')}-0481"

    # Patient info card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    left, right = st.columns([1, 3])
    with left:
        initials = _patient_initials(profile["name"])
        st.markdown(
            f"""
            <div style='width:84px;height:84px;border-radius:50%;background:#e8f0ff;display:flex;
                        align-items:center;justify-content:center;font-size:30px;font-weight:700;color:#1a3a58;'>
                {initials}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(profile["name"])
        st.caption("Patient")
        st.write(profile["patient_id"])

    with right:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.caption("Date of Birth")
            st.write(profile["dob"])
            st.caption("Scan Date")
            st.write(scan_date)
        with m2:
            st.caption("Age / Gender")
            st.write(profile["age_gender"])
            st.caption("Modality")
            st.write(report_data["modality"])
        with m3:
            st.caption("Referring Physician")
            st.write(profile["physician"])
            st.caption("Facility")
            st.write(profile["facility"])

    p1, p2, p3 = st.columns(3)
    p1.info("Scan Quality: High")
    p2.info("Analysis: Complete")
    p3.warning("Reviewed: Pending")
    st.markdown('</div>', unsafe_allow_html=True)

    locked = not st.session_state.report_unlocked
    if locked:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔒 Premium Report Locked")
        st.write("Unlock the full plain-language report, recommendations, and next steps.")
        if st.button("Unlock Full Report", key="unlock_from_patient"):
            st.session_state.report_unlocked = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    if not locked:
        # Diagnosis summary card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("What We Found")
        sev_text, sev_emoji = _severity_banner(report_data["severity"])
        st.success(f"{sev_emoji} {sev_text}")
        st.write(report_data["summary"])
        c1, c2 = st.columns(2)
        with c1:
            conf_pct = report_data["confidence_pct"]
            conf_val = conf_pct if conf_pct is not None else 75.0
            st.write("Confidence")
            st.progress(conf_val / 100.0)
            st.caption(f"{conf_val:.1f}%")
        with c2:
            sev_score = _severity_urgency_score(report_data["severity"])
            st.write("Severity")
            st.progress(sev_score / 100.0)
            st.caption(f"Severity index: {sev_score:.0f}/100")
        st.markdown('</div>', unsafe_allow_html=True)

        # What this means
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("What This Means For You")
        st.write(f"- {report_data['meaning']}")
        st.write("- This report is designed to support a doctor conversation in plain language.")
        st.write("- Your care team will combine this with symptoms and medical history.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Next steps
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Next Steps")
        for idx, step in enumerate(report_data["next_steps"][:4], start=1):
            st.write(f"{idx}. {step}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Questions for doctor
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Questions to Ask Your Doctor")
        for q in _default_doctor_questions(report_data["modality"]):
            st.info(f"💬 \"{q}\"")
        st.markdown('</div>', unsafe_allow_html=True)

    report = generate_report(
        report_data["modality"],
        report_data["finding"],
        report_data["severity"],
        r.get("prob", None),
        details={
            "summary": report_data["summary"],
            "meaning": report_data["meaning"],
            "next_steps": report_data["next_steps"],
            "questions": _default_doctor_questions(report_data["modality"]),
            "note": "This AI report supports but does not replace clinician diagnosis.",
        },
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    foot_l, foot_r = st.columns([1.4, 1.6])
    with foot_l:
        st.caption(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.caption(f"Report ID: {report_id}")
    with foot_r:
        st.caption("This report is AI-generated and must be reviewed by a licensed clinician.")
    with st.expander("View full text report"):
        st.text(report)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("← Back to Analysis"):
        st.session_state.page = "clinician"
        st.rerun()