import cv2
import numpy as np

def normalize(img):
    """Normalizes a single image channel to the [0, 1] range."""
    denom = img.max() - img.min()
    if denom != 0:
        return (img - img.min()) / denom
    return np.zeros_like(img)

def preprocess_slice(flair, t1ce, t2):
    """
    Preprocesses a single MRI slice by normalizing each modality
    and resizing to 128x128.
    """
    # Normalize each modality independently
    flair = normalize(flair)
    t1ce = normalize(t1ce)
    t2 = normalize(t2)

    # Resize to model input size
    flair = cv2.resize(flair, (128, 128))
    t1ce = cv2.resize(t1ce, (128, 128))
    t2 = cv2.resize(t2, (128, 128))

    # Stack modalities to create a 3-channel image
    img = np.stack([flair, t1ce, t2], axis=-1)
    
    return img.astype(np.float32)