import pydicom
import numpy as np
import cv2

def preprocess_ct(path):
    """
    Preprocesses a CT scan from a DICOM file.
    - Reads the DICOM file.
    - Applies rescale slope and intercept.
    - Clips the intensity to the brain window (0-80 HU).
    - Normalizes the image to [0, 1].
    - Resizes to 224x224.
    - Converts to 3 channels for the model.
    """
    try:
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.int16)
        
        # Apply rescale slope and intercept
        img = img * ds.RescaleSlope + ds.RescaleIntercept

        # Windowing for brain (0-80 HU)
        img = np.clip(img, 0, 80)

        # Normalize to [0, 1]
        if img.max() != img.min():
            img = (img - img.min()) / (img.max() - img.min())
        else:
            img = np.zeros_like(img)

        # Resize and convert to 3 channels
        img = cv2.resize(img, (224, 224))
        img = np.repeat(img[..., np.newaxis], 3, axis=-1)

        return img.astype(np.float32)
    except Exception as e:
        print(f"Error preprocessing DICOM file {path}: {e}")
        return None