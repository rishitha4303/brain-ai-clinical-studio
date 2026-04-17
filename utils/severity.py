import numpy as np

__all__ = ["ct_severity", "mri_severity"]


def ct_severity(prob):
    if prob < 0.4:
        return "No Hemorrhage"
    if prob < 0.6:
        return "Mild Hemorrhage"
    if prob < 0.8:
        return "Moderate Hemorrhage"
    return "Severe Hemorrhage"


def mri_severity(mask):
    ratio = float(np.sum(mask)) / float(mask.size) if mask.size else 0.0

    if ratio < 0.01:
        return "No Tumor"
    if ratio < 0.05:
        return "Mild Tumor"
    if ratio < 0.15:
        return "Moderate Tumor"
    return "Severe Tumor"