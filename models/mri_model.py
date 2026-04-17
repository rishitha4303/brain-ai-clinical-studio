import numpy as np
import tensorflow as tf

MODEL_PATH = "models/brats_unet_model.h5"

__all__ = ["load_mri_model", "predict_mri"]

_model = None


def load_mri_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _model


def predict_mri(img):
    """
    img: preprocessed image with shape (128, 128, 3)
    returns: (binary_mask, probability_map)
    """
    model = load_mri_model()

    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0]

    prob_map = pred[:, :, 0]
    mask = (prob_map > 0.5).astype(np.uint8)
    return mask, prob_map
