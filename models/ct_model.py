import tensorflow as tf
import numpy as np

IMG_SIZE = 224

def build_model():
    base = tf.keras.applications.DenseNet121(
        weights=None,
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=base.input, outputs=output)

    return model


# lazy load
model = None

def load_ct_model():
    global model
    if model is None:
        model = build_model()
        model.load_weights("models/CT_DenseNet50K_weights.h5")
        print("✅CT Model Loaded")
    return model


def predict_ct(img, threshold=0.4):
    m = load_ct_model()

    img = np.expand_dims(img, axis=0)

    prob = m.predict(img)[0][0]
    pred = 1 if prob > threshold else 0

    label = "Hemorrhage" if pred == 1 else "Normal"

    return label, prob