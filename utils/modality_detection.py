import re

__all__ = ["detect_modality", "infer_mri_modality"]


def detect_modality(filename):
    """Detects the general imaging modality (CT or MRI) from the filename."""
    name = filename.lower()
    if name.endswith(".dcm"):
        return "CT"
    if name.endswith(".nii") or name.endswith(".nii.gz"):
        return "MRI"
    return "Unknown"


def infer_mri_modality(filename: str) -> str:
    """
    Infers the specific MRI modality from the filename (e.g., flair, t1ce, t2).
    """
    filename_lower = filename.lower()
    
    if "flair" in filename_lower:
        return "flair"
    if "t1ce" in filename_lower or "t1_ce" in filename_lower:
        return "t1ce"
    if "t1" in filename_lower:
        return "t1"
    if "t2" in filename_lower:
        return "t2"
    if "seg" in filename_lower:
        return "seg"
        
    # Fallback for patterns like _t1.nii.gz
    match = re.search(r"_(t1|t2|t1ce|flair|seg)\.", filename_lower)
    if match:
        return match.group(1)
        
    return ""
