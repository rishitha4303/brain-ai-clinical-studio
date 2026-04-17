import cv2
import numpy as np

__all__ = ["get_mri_overlay", "get_mri_xai_views"]


def get_mri_overlay(img, mask):
    """
    img: (128, 128, 3), normalized to [0, 1]
    mask: (128, 128), binary segmentation mask
    """
    base = img[:, :, 0]
    base = (base * 255).astype(np.uint8)
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)

    overlay = base.copy()
    overlay[mask == 1] = [255, 0, 0]
    return overlay


def get_mri_xai_views(img, mask, prob_map):
    """
    Returns multiple MRI explainability views for UI display.
    - original: grayscale MRI slice in RGB
    - mask: binary segmentation mask
    - probability: probability heatmap
    - attention: probability blended with MRI slice
    - overlay: red mask on MRI slice
    """
    base = img[:, :, 0]
    base = (np.clip(base, 0, 1) * 255).astype(np.uint8)
    original = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)

    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_rgb = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2RGB)

    prob_u8 = (np.clip(prob_map, 0, 1) * 255).astype(np.uint8)
    probability = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)
    probability = cv2.cvtColor(probability, cv2.COLOR_BGR2RGB)

    attention = cv2.addWeighted(original, 0.65, probability, 0.35, 0)
    overlay = get_mri_overlay(img, mask)

    return {
        "original": original,
        "mask": mask_rgb,
        "probability": probability,
        "attention": attention,
        "overlay": overlay,
    }
